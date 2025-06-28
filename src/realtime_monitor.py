# realtime_monitor.py
import numpy as np
import websocket
import json
import time
import threading
from queue import Queue, Empty as QueueEmpty, Full as QueueFull
from typing import Dict, Any, Optional, List, Tuple, Set, Callable # Thêm Callable
import math
import warnings
import logging
import os
from collections import deque
import ssl
import pandas as pd
import asyncio

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
logger = logging.getLogger("RealtimeMonitor")

# --- Lấy URL WebSocket từ Environment Variable hoặc dùng Default ---
DEFAULT_BINANCE_WSS_URL = "wss://stream.binance.com:9443/stream?streams="
WSS_URL_TEMPLATE = os.getenv("WSS_URL_TEMPLATE", DEFAULT_BINANCE_WSS_URL)

# --- Cấu hình mặc định cho từng Symbol ---
DEFAULT_SYMBOL_CONFIG = {
    "vol_check_interval": 1.0,   # Giây
    "vol_threshold_pct": 0.2,    # Phần trăm
    "event_throttle_ms": 500,    # Milliseconds
    "rr_crossing_buffer_pct": 0.0005 # 0.05% buffer để xác nhận cắt qua RR
}

class RealtimeMonitor:
    def __init__(self,
                 symbols: List[str],
                 event_loop: asyncio.AbstractEventLoop,
                 # Thay thế queue bằng callbacks hoặc giữ cả hai
                 wakeup_queue: Optional[Queue] = None, # Queue trở thành tùy chọn
                 event_callbacks: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None, # <<< Callbacks tùy chỉnh
                 valid_symbols: Optional[Set[str]] = None,
                 per_symbol_config: Optional[Dict[str, Dict[str, Any]]] = None, # <<< Config cho từng symbol
                 wss_url_template: str = WSS_URL_TEMPLATE,
                 reconnect_delay_s: int = 5,
                 max_reconnect_attempts: int = 5,
                 price_history_len: int = 5,
                 ping_interval: int = 60,
                 ping_timeout: int = 10):
        if not websocket:
            raise ImportError("websocket-client library is required for RealtimeMonitor.")
        if not wakeup_queue and not event_callbacks:
             raise ValueError("Must provide either a wakeup_queue or at least one event_callback.")

        # --- Validate Symbols ---
        self.symbols_to_monitor = set(s.upper() for s in symbols)
        if valid_symbols:
            invalid_symbols = self.symbols_to_monitor - valid_symbols
            if invalid_symbols:
                logger.warning(f"Ignoring invalid symbols: {invalid_symbols}")
                self.symbols_to_monitor -= invalid_symbols
        if not self.symbols_to_monitor:
            raise ValueError("No valid symbols provided to monitor.")
        logger.info(f"Monitoring symbols: {list(self.symbols_to_monitor)}")
        self.main_event_loop = event_loop

        self.wakeup_queue = wakeup_queue
        self.event_callbacks = event_callbacks or {} # Đảm bảo là dict
        self.per_symbol_config_input = per_symbol_config or {}

        streams_path = "/".join([f"{s.lower()}@aggTrade" for s in self.symbols_to_monitor])
        self.wss_url = f"{wss_url_template}{streams_path}"
        logger.info(f"Monitor will connect to: {self.wss_url}")

        # Các tham số chung khác
        self.reconnect_delay_s = reconnect_delay_s
        self.max_reconnect_attempts = max_reconnect_attempts
        self.price_history_len = max(1, price_history_len)
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        self.position_states: Dict[str, Dict[str, Any]] = {}
        self._initialize_position_states() # Khởi tạo state ban đầu

        self.position_states_lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.ws_thread: Optional[threading.Thread] = None
        self.ws: Optional[websocket.WebSocketApp] = None
        self.reconnect_attempts = 0
        self._connected = threading.Event()
        self._last_pong_ts = time.time()

    def _initialize_position_states(self):
        """Khởi tạo cấu trúc state ban đầu cho các symbols."""
        for symbol in self.symbols_to_monitor:
            # Kết hợp config mặc định và config riêng của symbol (nếu có)
            symbol_specific_config = self.per_symbol_config_input.get(symbol, {})
            final_config = {**DEFAULT_SYMBOL_CONFIG, **symbol_specific_config}

            self.position_states[symbol] = {
                "active": False,
                "config": final_config, # <<< Lưu config của symbol
                "price_history": deque(maxlen=self.price_history_len),
                "last_price": None,
                "last_vol_check_ts": 0,
                "last_event_ts": 0, # <<< Last event timestamp cho throttling
                # Trạng thái RR crossing mới
                "last_crossed_up_level": float('-inf'), # Mức RR cuối cùng giá đã vượt lên trên
                "last_crossed_down_level": float('inf'), # Mức RR cuối cùng giá đã giảm xuống dưới
                "thresholds": {},
                "position_type": None,
                "current_trailing_tp_level": 0,
                "min_locked_tp_level": 0,
                "rr_step_price": None,
            }

    # --- WebSocket Callbacks ---
    def _on_message(self, ws, message):
        try:
            data_outer = json.loads(message)
            if 'stream' not in data_outer or 'data' not in data_outer: return

            data = data_outer['data']
            stream_name = data_outer['stream']
            symbol = stream_name.split('@')[0].upper()
            if symbol not in self.symbols_to_monitor: return

            if data.get('e') == 'aggTrade' and 'p' in data and 'T' in data:
                current_price = float(data['p'])
                timestamp_ms = data['T']
                if current_price <= 0: return

                # --- Biến cục bộ để tránh race condition nhỏ ---
                state_copy = None
                symbol_config = None
                thresholds_copy = None
                position_type_copy = None
                is_active = False
                current_tp_level_num_copy = 0
                min_tp_floor_price_copy = None
                sl_price_copy = None
                last_event_ts_copy = 0
                last_crossed_up_copy = float('-inf')
                last_crossed_down_copy = float('inf')
                rr_step_price_copy = None

                with self.position_states_lock:
                    state = self.position_states.get(symbol)
                    if not state: return # Nên được khởi tạo rồi, nhưng kiểm tra cho chắc

                    # Cập nhật giá và lịch sử giá
                    state['price_history'].append(current_price)
                    state['last_price'] = current_price
                    is_active = state.get('active', False)

                    if is_active:
                        # Sao chép các giá trị cần thiết vào biến cục bộ trong lock
                        state_copy = state.copy() # Sao chép nông là đủ vì các giá trị cần là immutable hoặc dict/list sẽ copy sau
                        symbol_config = state_copy.get('config', DEFAULT_SYMBOL_CONFIG).copy() # Copy config dict
                        thresholds_copy = state_copy.get('thresholds', {}).copy() # Copy thresholds dict
                        position_type_copy = state_copy.get('position_type')
                        current_tp_level_num_copy = state_copy.get('current_trailing_tp_level', 0)
                        sl_price_copy = thresholds_copy.get('SL')
                        min_tp_floor_price_copy = thresholds_copy.get('MIN_TP_FLOOR')
                        last_event_ts_copy = state_copy.get('last_event_ts', 0)
                        last_crossed_up_copy = state_copy.get('last_crossed_up_level', float('-inf'))
                        last_crossed_down_copy = state_copy.get('last_crossed_down_level', float('inf'))
                        rr_step_price_copy = state_copy.get('rr_step_price')
                    else:
                        # Nếu không active, không cần xử lý gì thêm
                        return

                # --- Xử lý logic bên ngoài lock bằng các giá trị đã sao chép ---
                if not is_active or not state_copy or not symbol_config or not thresholds_copy or not position_type_copy:
                    return # Thoát nếu không active hoặc thiếu thông tin

                time_now = time.time()
                event_throttle_ms = symbol_config.get('event_throttle_ms', DEFAULT_SYMBOL_CONFIG['event_throttle_ms'])

                # --- 1. Check Volatility ---
                vol_check_interval = symbol_config.get('vol_check_interval', DEFAULT_SYMBOL_CONFIG['vol_check_interval'])
                last_vol_check_ts = state_copy.get('last_vol_check_ts', 0)

                if (time_now - last_vol_check_ts) >= vol_check_interval:
                    price_history_copy = list(state_copy['price_history']) # Lấy snapshot lịch sử giá
                    # Cập nhật last_vol_check_ts trong state gốc
                    with self.position_states_lock:
                         self.position_states[symbol]['last_vol_check_ts'] = time_now

                    if len(price_history_copy) >= 2:
                        oldest_price = price_history_copy[0]
                        if oldest_price > 0:
                            price_change_percent = abs(current_price - oldest_price) / oldest_price * 100
                            vol_threshold_pct = symbol_config.get('vol_threshold_pct', DEFAULT_SYMBOL_CONFIG['vol_threshold_pct'])
                            if price_change_percent >= vol_threshold_pct:
                                if timestamp_ms > last_event_ts_copy + event_throttle_ms:
                                    logger.warning(f"[{symbol}] Abnormal Volatility: {price_change_percent:.2f}%")
                                    signal_data = {"type":"ABNORMAL_VOLATILITY", "symbol":symbol, "price":current_price, "change_percent":price_change_percent, "timestamp":timestamp_ms }
                                    self._dispatch_signal(signal_data)
                                    # Cập nhật last_event_ts trong state gốc
                                    with self.position_states_lock:
                                        self.position_states[symbol]['last_event_ts'] = timestamp_ms
                                    return # Gửi 1 loại tín hiệu mỗi lần

                # --- 2. Check Thresholds (Exit conditions) ---
                triggered_info: Optional[Tuple[str, float | int]] = None # Có thể là giá hoặc level

                def check_hit(price, threshold, p_type):
                     if threshold is None: return False
                     if p_type == 'long': return price <= threshold
                     if p_type == 'short': return price >= threshold
                     return False

                # Lấy giá TP di động hiện tại
                current_tp_key = f'RR_{current_tp_level_num_copy:.1f}'.replace('.0', '') if current_tp_level_num_copy > 0 else None
                current_tp_price_copy = thresholds_copy.get(current_tp_key) if current_tp_key else None

                # Kiểm tra SL
                if check_hit(current_price, sl_price_copy, position_type_copy):
                    triggered_info = ("SL", sl_price_copy)
                    logger.warning(f"[{symbol}] SL HIT ({current_price:.4f} <= {sl_price_copy:.4f} for {position_type_copy})")
                # Kiểm tra MIN_TP_FLOOR
                elif check_hit(current_price, min_tp_floor_price_copy, position_type_copy):
                    triggered_info = ("MIN_TP_FLOOR", min_tp_floor_price_copy)
                    logger.warning(f"[{symbol}] MIN TP FLOOR HIT ({current_price:.4f} vs {min_tp_floor_price_copy:.4f} for {position_type_copy})")
                # Kiểm tra TRAILING_TP (phải khác MIN_TP_FLOOR nếu có)
                elif (
                    current_tp_price_copy is not None and
                    (min_tp_floor_price_copy is None or abs(current_tp_price_copy - min_tp_floor_price_copy) > 1e-9) and
                    check_hit(current_price, current_tp_price_copy, position_type_copy)
                ):
                    triggered_info = ("TRAILING_TP", current_tp_level_num_copy) # Gửi level thay vì giá
                    logger.info(f"[{symbol}] TRAILING TP HIT (Level {current_tp_level_num_copy}, Price {current_price:.4f} vs {current_tp_price_copy:.4f} for {position_type_copy})")

                # Gửi tín hiệu nếu chạm ngưỡng
                if triggered_info:
                    if timestamp_ms > last_event_ts_copy + event_throttle_ms:
                        logger.info(f"[{symbol}] Sending EXIT Threshold Event: {triggered_info[0]}")
                        signal_data = {
                            "type": "THRESHOLD_HIT", "symbol": symbol,
                            "threshold_type": triggered_info[0],
                            "level_or_price": triggered_info[1], # Giá cho SL/Floor, Level cho TrailingTP
                            "current_price": current_price, "timestamp": timestamp_ms
                        }
                        self._dispatch_signal(signal_data)
                        # Cập nhật last_event_ts trong state gốc
                        with self.position_states_lock:
                            self.position_states[symbol]['last_event_ts'] = timestamp_ms
                        return # Gửi 1 loại tín hiệu mỗi lần

                # --- 3. Check Optimized RR Crossing ---
                # Lấy tất cả các mức RR và sắp xếp
                rr_levels = []
                for k, v in thresholds_copy.items():
                    if k.startswith('RR_') and v is not None:
                        try:
                            level_num = float(k.split('_')[1])
                            rr_levels.append({'name': k, 'price': v, 'level': level_num})
                        except ValueError:
                            logger.warning(f"[{symbol}] Could not parse RR level: {k}")
                rr_levels.sort(key=lambda x: x['level']) # Sắp xếp theo level tăng dần

                # Tìm mức RR ngay dưới và ngay trên giá hiện tại
                lower_rr: Optional[Dict] = None
                upper_rr: Optional[Dict] = None
                for i, rr in enumerate(rr_levels):
                    if rr['price'] < current_price:
                        lower_rr = rr # Cập nhật lower_rr liên tục khi giá tăng
                    if rr['price'] > current_price:
                        upper_rr = rr # Lấy upper_rr đầu tiên tìm thấy
                        break # Không cần tìm nữa

                rr_crossing_buffer_pct = symbol_config.get('rr_crossing_buffer_pct', DEFAULT_SYMBOL_CONFIG['rr_crossing_buffer_pct'])
                crossed_signal_data = None
                new_last_crossed_up = last_crossed_up_copy
                new_last_crossed_down = last_crossed_down_copy

                # Kiểm tra cắt lên trên (UPWARD CROSSING)
                if upper_rr and upper_rr['level'] > last_crossed_up_copy:
                    buffer = upper_rr['price'] * rr_crossing_buffer_pct
                    if current_price > upper_rr['price'] + buffer:
                        logger.info(f"[{symbol}] Price crossed UP {upper_rr['name']} (Lvl {upper_rr['level']:.1f} > LastUp {last_crossed_up_copy:.1f}) at {current_price:.4f} (Thresh: {upper_rr['price']:.4f})")
                        crossed_signal_data = {
                            "type": "RR_CROSSED", "symbol": symbol, "level": upper_rr['level'],
                            "direction": "up", "threshold_price": upper_rr['price'],
                            "current_price": current_price, "timestamp": timestamp_ms
                        }
                        new_last_crossed_up = upper_rr['level']
                        # Cập nhật cả down nếu cần (ví dụ giá nhảy vọt qua nhiều mức)
                        if lower_rr and lower_rr['level'] > last_crossed_down_copy:
                             new_last_crossed_down = lower_rr['level']


                # Kiểm tra cắt xuống dưới (DOWNWARD CROSSING) - Chỉ xảy ra nếu không có tín hiệu cắt lên
                elif lower_rr and lower_rr['level'] < last_crossed_down_copy:
                    buffer = lower_rr['price'] * rr_crossing_buffer_pct
                    if current_price < lower_rr['price'] - buffer:
                        logger.info(f"[{symbol}] Price crossed DOWN {lower_rr['name']} (Lvl {lower_rr['level']:.1f} < LastDown {last_crossed_down_copy:.1f}) at {current_price:.4f} (Thresh: {lower_rr['price']:.4f})")
                        crossed_signal_data = {
                            "type": "RR_CROSSED", "symbol": symbol, "level": lower_rr['level'],
                            "direction": "down", "threshold_price": lower_rr['price'],
                            "current_price": current_price, "timestamp": timestamp_ms
                        }
                        new_last_crossed_down = lower_rr['level']
                         # Cập nhật cả up nếu cần
                        if upper_rr and upper_rr['level'] < new_last_crossed_up:
                             new_last_crossed_up = upper_rr['level'] if upper_rr else float('-inf')


                # Gửi tín hiệu nếu có crossing và đủ throttle
                if crossed_signal_data:
                    if timestamp_ms > last_event_ts_copy + event_throttle_ms:
                        self._dispatch_signal(crossed_signal_data)
                        # Cập nhật trạng thái crossing và event ts trong state gốc
                        with self.position_states_lock:
                             self.position_states[symbol]['last_crossed_up_level'] = new_last_crossed_up
                             self.position_states[symbol]['last_crossed_down_level'] = new_last_crossed_down
                             self.position_states[symbol]['last_event_ts'] = timestamp_ms
                        # Không return ở đây, cho phép các logic khác chạy nếu cần (hiện tại thì không)

        except json.JSONDecodeError: logger.debug("Received non-JSON message.")
        except KeyError as ke: logger.warning(f"Monitor: Missing key in trade data: {ke}")
        except Exception as e: logger.error(f"Error processing message: {e}", exc_info=True)

    def _on_error(self, ws, error): logger.error(f"Monitor WebSocket Error: {error}")
    def _on_close(self, ws, close_status_code, close_msg):
        self._connected.clear(); logger.warning(f"### Monitor WebSocket Closed (Code: {close_status_code}, Msg: {close_msg}) ###")

    def _on_open(self, ws):
        self.reconnect_attempts = 0; self._connected.set(); self._last_pong_ts = time.time()
        logger.info("### Monitor WebSocket Opened ###")

    def _on_ping(self, ws, message): logger.debug("Received WebSocket Ping")
    def _on_pong(self, ws, message): logger.debug("Received WebSocket Pong"); self._last_pong_ts = time.time()

    def _dispatch_signal(self, signal_data: Dict[str, Any]):
        signal_type = signal_data.get("type")
        if not signal_type:
            logger.warning("Dispatch signal called with missing 'type'.")
            return

        callback = self.event_callbacks.get(signal_type)

        # Kiểm tra xem callback có tồn tại và event loop đã được cung cấp khi khởi tạo không
        if callback and hasattr(self, 'main_event_loop') and self.main_event_loop:
            try:
                self.main_event_loop.call_soon_threadsafe(
                    self._execute_callback_threadsafe, # Hàm sẽ chạy trong event loop
                    callback,                         # Hàm callback thực tế (async hoặc sync)
                    signal_data                       # Dữ liệu tín hiệu
                )
            except RuntimeError as e:
                 # Lỗi này thường xảy ra nếu event loop đã bị đóng
                 logger.error(f"RuntimeError scheduling callback for {signal_type}: {e}. Loop might be closed.")
            except Exception as e:
                # Các lỗi khác khi cố gắng lên lịch
                logger.error(f"Failed to schedule callback for {signal_type}: {e}", exc_info=True)
        # Chỉ log cảnh báo nếu không tìm thấy callback phù hợp
        elif not callback:
             logger.warning(f"No callback configured for signal type '{signal_type}'. Signal dropped.")
        # Trường hợp có callback nhưng không có main_event_loop (lỗi khởi tạo)
        elif callback and (not hasattr(self, 'main_event_loop') or not self.main_event_loop):
             logger.error(f"Cannot dispatch signal '{signal_type}': Callback exists but main_event_loop is missing or invalid.")
    
    def _execute_callback_threadsafe(self, callback, signal_data):
        signal_type = signal_data.get("type", "UNKNOWN") # Lấy type để log lỗi tốt hơn
        callback_name = getattr(callback, '__name__', 'N/A') # Lấy tên callback để log

        try:
            if asyncio.iscoroutinefunction(callback):
                task = asyncio.create_task(callback(signal_data), name=f"Callback_{signal_type}_{callback_name}")
                def task_done_callback(fut: asyncio.Future):
                    try:
                        # Kiểm tra xem task có gây exception không
                        exception = fut.exception()
                        if exception:
                            logger.error(f"Exception in async callback task '{fut.get_name()}': {exception}", exc_info=exception)
                    except asyncio.CancelledError:
                         logger.warning(f"Async callback task '{fut.get_name()}' was cancelled.")
                    except Exception as e:
                        logger.error(f"Error in task done callback for '{fut.get_name()}': {e}")

                task.add_done_callback(task_done_callback)
            else:
                callback(signal_data)

        except Exception as e:
            logger.error(f"Error initiating callback '{callback_name}' for signal type {signal_type}: {e}", exc_info=True)


    def _run_websocket_thread(self):
         while not self.stop_flag.is_set():
            logger.info(f"Attempting to connect WebSocket (Attempt: {self.reconnect_attempts + 1})...")
            sslopt = {"cert_reqs": ssl.CERT_NONE} if self.wss_url.startswith("wss:") else {}
            self.ws = websocket.WebSocketApp(self.wss_url,
                                             on_open=self._on_open,
                                             on_message=self._on_message,
                                             on_error=self._on_error,
                                             on_close=self._on_close,
                                             on_ping=self._on_ping,
                                             on_pong=self._on_pong)

            ws_run_thread = threading.Thread(target=self.ws.run_forever, kwargs={
                "sslopt": sslopt,
                "ping_interval": self.ping_interval,
                "ping_timeout": self.ping_timeout
            }, daemon=True)
            ws_run_thread.start()

            while ws_run_thread.is_alive() and not self.stop_flag.is_set():
                if self._connected.is_set() and (time.time() - self._last_pong_ts > self.ping_interval + self.ping_timeout + 5):
                     logger.warning(f"Pong timeout detected (last pong {time.time() - self._last_pong_ts:.1f}s ago). Closing connection.")
                     if self.ws: self.ws.close()
                     break
                time.sleep(self.ping_interval // 2)

            ws_run_thread.join(timeout=1.0)

            if self.stop_flag.is_set(): logger.info("Stop flag set, exiting websocket loop."); break
            self.reconnect_attempts += 1
            if self.reconnect_attempts >= self.max_reconnect_attempts: logger.error(f"Max reconnect attempts ({self.max_reconnect_attempts}) reached. Stopping monitor."); self.stop_flag.set(); break
            logger.info(f"WebSocket connection lost/closed. Reconnecting in {self.reconnect_delay_s} seconds...")
            time.sleep(self.reconnect_delay_s)
         logger.info("Websocket thread finished.")


    def update_thresholds(self, symbol: str, active: bool, position_type: Optional[str] = None, thresholds: Optional[Dict[str, float]] = None, current_trailing_tp_level: int = 0, min_locked_tp_level: int = 0):
        """ Updates the price thresholds and state for a symbol. """
        symbol_upper = symbol.upper()
        if symbol_upper not in self.symbols_to_monitor:
             logger.warning(f"Monitor: Symbol {symbol_upper} not in the monitoring list. Ignoring update.")
             return

        with self.position_states_lock:
            # Lấy state hiện tại hoặc tạo mới nếu cần (dù đã có init)
            state = self.position_states.get(symbol_upper)
            if not state:
                logger.error(f"State for {symbol_upper} not found during update, should have been initialized.")
                # Tạo lại state với config mặc định nếu bị thiếu vì lý do nào đó
                symbol_specific_config = self.per_symbol_config_input.get(symbol_upper, {})
                final_config = {**DEFAULT_SYMBOL_CONFIG, **symbol_specific_config}
                state = {
                    "active": False, "config": final_config,
                    "price_history": deque(maxlen=self.price_history_len),
                    # ... khởi tạo lại các trường khác ...
                    "last_crossed_up_level": float('-inf'),
                    "last_crossed_down_level": float('inf'),
                    "last_event_ts": 0,
                }
                self.position_states[symbol_upper] = state


            if active:
                if not position_type or thresholds is None: # thresholds có thể rỗng
                    logger.warning(f"Monitor: Cannot activate {symbol_upper}: missing position_type or thresholds dict.")
                    state['active'] = False # Đảm bảo là false nếu thiếu thông tin
                    return

                # Tính rr_step
                rr_step = None
                entry = thresholds.get('entry_price')
                sl = thresholds.get('SL')
                if entry is not None and sl is not None:
                    rr_step = abs(entry - sl)

                # Reset trạng thái khi kích hoạt
                state.update({
                    "active": True,
                    "position_type": position_type,
                    "thresholds": thresholds.copy(),
                    "current_trailing_tp_level": current_trailing_tp_level,
                    "min_locked_tp_level": min_locked_tp_level,
                    "rr_step_price": rr_step,
                    # Giữ lại last_price và history nếu có
                    "last_price": state.get('last_price'),
                    # Reset trạng thái sự kiện và crossing
                    "last_vol_check_ts": time.time(), # <<< Reset để kiểm tra vol ngay
                    "last_event_ts": 0,
                    "last_crossed_up_level": float('-inf'),
                    "last_crossed_down_level": float('inf'),
                })
                logger.info(f"Monitor: Activated/Updated thresholds for {symbol_upper} (TP: {current_trailing_tp_level}, Floor: {min_locked_tp_level})")

            else: # Deactivate
                if state.get('active'):
                    state['active'] = False
                    # Có thể reset một số trường khác nếu cần khi deactivate
                    state['position_type'] = None
                    state['thresholds'] = {}
                    logger.info(f"Monitor: Deactivated {symbol_upper}")
                # else: logger.debug(f"Monitor: {symbol_upper} was already inactive.")


    def start(self):
        if not websocket: logger.error("Monitor Error: websocket-client not installed."); return
        if self.ws_thread and self.ws_thread.is_alive(): logger.warning("Monitor already running."); return
        self.stop_flag.clear(); self.reconnect_attempts = 0
        # Đảm bảo states được khởi tạo trước khi start thread
        if not self.position_states:
            self._initialize_position_states()
        self.ws_thread = threading.Thread(target=self._run_websocket_thread, daemon=True)
        self.ws_thread.start(); logger.info("Monitor thread starting.")

    def stop(self):
        logger.info("Monitor attempting to stop...")
        self.stop_flag.set()
        if self.ws:
            try: self.ws.close(); logger.info("Monitor WebSocket close signal sent.")
            except Exception as e: logger.error(f"Error closing monitor websocket: {e}")
        if self.ws_thread and self.ws_thread.is_alive():
            logger.info("Waiting for monitor thread to join...")
            self.ws_thread.join(timeout=self.reconnect_delay_s + 2)
            if self.ws_thread.is_alive(): logger.warning("Monitor thread did not stop gracefully.")
            else: logger.info("Monitor thread stopped.")
        else: logger.info("Monitor thread was not running or already stopped.")
        self.ws = None; self.ws_thread = None; self._connected.clear()

    # Helper static method (không thay đổi)
    @staticmethod
    def _calculate_rr_price(entry: float, sl: float, level: float, p_type: str) -> Optional[float]:
        risk = 0.0
        if p_type == 'long': risk = entry - sl
        elif p_type == 'short': risk = sl - entry
        else: return None
        if risk <= 1e-9: return None
        target = entry + level * risk if p_type == 'long' else entry - level * risk
        return max(0.0, target) if target is not None else None


# --- Ví dụ kiểm thử riêng module Monitor ---
if __name__ == "__main__":
    print("--- Testing RealtimeMonitor Module (Callback/PerSymbol/OptimizedRR) ---")

    # --- Định nghĩa các hàm callback mẫu ---
    def handle_threshold_hit(signal_data):
        logger.info(f"[CALLBACK] Threshold Hit: {signal_data}")
        symbol = signal_data.get("symbol")
        # Trong ứng dụng thực, bạn sẽ gọi hàm để đóng vị thế ở đây
        # và sau đó gọi monitor_test.update_thresholds(symbol, active=False)
        print(f" >>> ACTION: Close position for {symbol} due to {signal_data.get('threshold_type')} <<<")
        # Tạm thời deactivate ngay trong callback để test
        if symbol in monitor_test.position_states: # Kiểm tra monitor còn chạy không
             monitor_test.update_thresholds(symbol, active=False)


    def handle_rr_crossed(signal_data):
         logger.info(f"[CALLBACK] RR Crossed: {signal_data}")
         # Bot có thể cập nhật trailing TP/floor ở đây dựa trên tín hiệu này
         print(f" >>> INFO: {signal_data.get('symbol')} crossed RR {signal_data.get('level')} {signal_data.get('direction')} <<<")

    def handle_volatility(signal_data):
         logger.warning(f"[CALLBACK] Abnormal Volatility: {signal_data}")
         # Bot có thể giảm size vị thế hoặc gửi cảnh báo
         print(f" >>> ALERT: High volatility detected for {signal_data.get('symbol')} <<<")

    test_callbacks = {
        "THRESHOLD_HIT": handle_threshold_hit,
        "RR_CROSSED": handle_rr_crossed,
        "ABNORMAL_VOLATILITY": handle_volatility,
    }

    # --- Cấu hình cho từng symbol (ví dụ) ---
    test_per_symbol_config = {
        "BTCUSDT": {
            "vol_check_interval": 0.8,
            "vol_threshold_pct": 0.1, # Nhạy hơn với BTC
            "event_throttle_ms": 400,
            "rr_crossing_buffer_pct": 0.0003
        },
        "ETHUSDT": {
            "vol_check_interval": 1.2,
            "vol_threshold_pct": 0.15,
            "event_throttle_ms": 600,
            "rr_crossing_buffer_pct": 0.0006 # Buffer lớn hơn cho ETH
        }
        # Các symbol khác sẽ dùng DEFAULT_SYMBOL_CONFIG
    }

    # --- Khởi tạo Monitor ---
    monitor_symbols = ["BTCUSDT", "ETHUSDT"] # Thêm ETH để test config khác nhau
    monitor_test = RealtimeMonitor(
        symbols=monitor_symbols,
        # wakeup_queue=None, # Không dùng queue nữa nếu đã có callback
        event_callbacks=test_callbacks,
        per_symbol_config=test_per_symbol_config,
        # Các tham số khác giữ nguyên hoặc tùy chỉnh
    )

    monitor_test.start()
    time.sleep(5) # Đợi kết nối

    # --- Kích hoạt theo dõi cho BTCUSDT ---
    print("\nSimulating setting initial thresholds for BTCUSDT...")
    btc_entry = 68000.0; btc_sl = 67000.0; btc_type = "long"
    btc_thresholds = {"SL": btc_sl, "entry_price": btc_entry}
    for rr in np.arange(0.5, 3.1, 0.5): # Tạo RR levels
        p = RealtimeMonitor._calculate_rr_price(btc_entry, btc_sl, rr, btc_type)
        if p: btc_thresholds[f'RR_{rr:.1f}'] = p
    monitor_test.update_thresholds("BTCUSDT", True, btc_type, btc_thresholds, current_trailing_tp_level=0, min_locked_tp_level=0)

     # --- Kích hoạt theo dõi cho ETHUSDT ---
    print("\nSimulating setting initial thresholds for ETHUSDT...")
    eth_entry = 3800.0; eth_sl = 3750.0; eth_type = "long"
    eth_thresholds = {"SL": eth_sl, "entry_price": eth_entry}
    for rr in np.arange(0.5, 3.1, 0.5): # Tạo RR levels
        p = RealtimeMonitor._calculate_rr_price(eth_entry, eth_sl, rr, eth_type)
        if p: eth_thresholds[f'RR_{rr:.1f}'] = p
    monitor_test.update_thresholds("ETHUSDT", True, eth_type, eth_thresholds, current_trailing_tp_level=0, min_locked_tp_level=0)

    print("\nWaiting for signals (handled by callbacks)... Press Ctrl+C to stop.")

    try:
        # Vòng lặp chính giờ chỉ cần chạy để giữ chương trình sống
        # Hoặc có thể làm việc khác
        while True:
            # Kiểm tra xem tất cả symbols đã bị deactive chưa
            all_inactive = True
            with monitor_test.position_states_lock:
                 for symbol in monitor_symbols:
                     if monitor_test.position_states.get(symbol, {}).get('active', False):
                         all_inactive = False
                         break
            if all_inactive and len(monitor_symbols)>0: # Kiểm tra len > 0 phòng trường hợp list rỗng ban đầu
                 print("\nAll monitored symbols are inactive. Stopping test.")
                 break

            print(".", end="", flush=True)
            time.sleep(2) # Chờ lâu hơn vì callback xử lý rồi

    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        monitor_test.stop()
        print("--- Monitor Test Finished ---")