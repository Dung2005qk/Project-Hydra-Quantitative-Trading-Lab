from __future__ import annotations
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import joblib
import pickle # Thêm pickle để có thể tải file .pkl cũ nếu joblib lỗi
import logging
from datetime import datetime, timedelta, timezone
import asyncio
import os
import sys
from typing import Dict, Optional, List, Tuple, Any
import talib
import aiohttp
from scipy.signal import find_peaks
from typing import TypeAlias, Type
EnhancedTradingBot: TypeAlias = Type[Any]

CONFIG = {
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframes": ["1m", "5m", "15m", "1h", "4h"],  # Cập nhật để hỗ trợ 5 khung thời gian
    "days_to_fetch": 365,
    "initial_balance": 10000,
    "leverage": 5,
    "risk_per_trade": 0.06,
    "fee_rate": 0.0005,
    "max_trades_per_day": 15,
    "max_account_risk": 0.07,
    "min_position_size": 0.02,
    "max_position_size": 0.5,
    "prob_threshold": 0.75,
    "sentiment_window": 24,
    "min_samples": 50,
    "lookback_window": 10,
    "hmm_warmup_min": 50,
    "hmm_warmup_max": 200,
    "vah_bin_range": (20, 60),
    "fib_levels": [0, 0.236, 0.382, 0.5, 0.618, 1],
    "max_gap_storage": 3,
    "circuit_breaker_threshold": 4.0,
    "circuit_breaker_cooldown": 60,
    "max_exposure": 0.75,
    "volume_spike_multiplier": 4.0,
    "volume_sustained_multiplier": 1.5,
    "adx_threshold": 20,
    "adx_thresholds": {
        "1h": 22,      
        "4h": 25,     
        "15m": 25,     
        "default": 20   
    },
     "symbol_risk_modifiers": {
        "BTC": 1.25, 
        "ETH": 1.10,  
        "DEFAULT": 0.90 
    },
    "volume_ma20_threshold": 0.8,
    "api_key": os.getenv("BINANCE_API_KEY", "your_api_key"),
    "api_secret": os.getenv("BINANCE_API_SECRET", "your_api_secret"),
    "enable_rate_limit": True,
    "rate_limit_delay": 0.2,
    "websocket_url": "wss://stream.binance.com:9443/ws",
    "xgb_long_threshold": 0.65, 
    "xgb_short_threshold": 0.35,
    "min_xgb_feature_dim": 20,
}

class MinimalBot:
            def __init__(self):
                self.exchange = None # Sẽ được khởi tạo
                self.regime_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
                self.gap_levels = {}
                self.temporal_features = [] # Placeholder
                self.vah_optimizer = None # Placeholder

            async def _initialize_exchange(self):
                try:
                    exchange_config = {
                        'apiKey': CONFIG.get("api_key"), # Dùng get để tránh KeyError
                        'secret': CONFIG.get("api_secret"),
                        'enableRateLimit': CONFIG.get("enable_rate_limit", True),
                        'options': {
                            'adjustForTimeDifference': True, # Tự động điều chỉnh time diff
                            'defaultType': 'future', # Chỉ định loại tài khoản (nếu cần)
                        },
                        # 'rateLimit': 1200, # ccxt tự quản lý rate limit nếu enableRateLimit=True
                    }
                    # Xử lý testnet nếu có
                    if CONFIG.get("use_testnet", False):
                        exchange_config['options']['defaultType'] = 'future' # Testnet thường là future
                        # Hoặc exchange_config['urls'] = {'api': 'https://testnet.binancefuture.com'} tùy ccxt version

                    self.exchange = ccxt.binance(exchange_config)

                    # Kiểm tra API keys (nếu có)
                    if not exchange_config['apiKey'] or not exchange_config['secret']:
                        logging.warning("API key or secret is missing. Exchange functionality will be limited.")
                    else:
                        # Thử gọi một hàm yêu cầu xác thực nhẹ nhàng để kiểm tra keys
                        await self.exchange.fetch_balance()
                        logging.info("API keys verified successfully.")

                    await self.exchange.load_markets()
                    logging.info(f"Successfully initialized Binance exchange and loaded {len(self.exchange.markets)} markets.")

                except ccxt.AuthenticationError as e:
                    logging.critical(f"Exchange Authentication Error: {e}. Check API keys.")
                    raise
                except ccxt.ExchangeNotAvailable as e:
                    logging.critical(f"Exchange Not Available: {e}.")
                    raise
                except Exception as e:
                    logging.critical(f"Failed to initialize or test exchange connection: {e}", exc_info=True)
                    raise

            async def _fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None) -> Optional[List]: # Sửa gợi ý kiểu trả về thành List
                limit = 1000  # Max candles per request (Binance default)
                max_retries = 3
                # Tăng nhẹ delay để giảm khả năng bị rate limit liên tục
                base_retry_delay = 7 # giây
                all_ohlcv = [] # List để tích lũy kết quả

                # Log thời điểm bắt đầu fetch một cách rõ ràng
                since_dt_str = pd.to_datetime(since, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S %Z') if since else 'beginning of available history'
                logging.debug(f"Attempting to fetch OHLCV for {symbol} {timeframe} since {since_dt_str}...")

                for attempt in range(max_retries):
                    current_since = since
                    try:
                        while True:
                            # Log chi tiết hơn cho từng request nhỏ bên trong
                            current_since_dt_str = pd.to_datetime(current_since, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S %Z') if current_since else 'beginning'
                            logging.debug(f"Fetching up to {limit} candles for {symbol} {timeframe} starting from {current_since_dt_str} (Attempt {attempt + 1}/{max_retries})")

                            # Thực hiện fetch
                            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)

                            # Xử lý kết quả fetch
                            if not ohlcv:
                                logging.debug(f"No more data returned for {symbol} {timeframe} after {current_since_dt_str}.")
                                break  # Hết dữ liệu cho lần fetch này

                            # Thêm dữ liệu mới vào list tổng
                            all_ohlcv.extend(ohlcv)
                            last_ts = ohlcv[-1][0] # Timestamp của nến cuối cùng nhận được

                            # Tính toán timestamp cho lần fetch tiếp theo
                            timeframe_duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
                            current_since = last_ts + timeframe_duration_ms

                            # Kiểm tra xem đã đến hiện tại chưa để dừng sớm
                            if pd.to_datetime(current_since, unit='ms', utc=True) > datetime.now(timezone.utc):
                                logging.debug(f"Reached current time for {symbol} {timeframe}.")
                                break

                            # Delay nhỏ giữa các request liên tiếp trong cùng một lần thử
                            await asyncio.sleep(CONFIG.get("rate_limit_delay", 0.2))

                        # Nếu vòng lặp while kết thúc mà không có lỗi (break),
                        # nghĩa là đã fetch thành công cho lần thử 'attempt' này.
                        logging.info(f"Successfully fetched {len(all_ohlcv)} raw candles cumulatively for {symbol} {timeframe} in attempt {attempt + 1}.")
                        # --- TRẢ VỀ LIST THÔ HOẶC NONE ---
                        if not all_ohlcv:
                            return None # Trả về None nếu list rỗng sau khi fetch thành công (ít xảy ra)
                        else:
                            return all_ohlcv # <<< ĐÚNG: Trả về list thô all_ohlcv

                    # --- Xử lý lỗi ---
                    except ccxt.RateLimitExceeded as e:
                        wait_time = base_retry_delay * (attempt + 1) # Simple linear backoff
                        logging.warning(f"Rate limit exceeded fetching {symbol} {timeframe} (Attempt {attempt+1}/{max_retries}): {e}. Retrying after {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        # Vòng lặp for sẽ tự động chuyển sang lần thử tiếp theo
                    except (ccxt.NetworkError, ccxt.RequestTimeout, aiohttp.ClientError) as e:
                        wait_time = base_retry_delay * (attempt + 1)
                        logging.warning(f"Network/Timeout error fetching {symbol} {timeframe} (Attempt {attempt+1}/{max_retries}): {e}. Retrying after {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    except ccxt.ExchangeError as e:
                        # Lỗi từ sàn thường không nên retry nhiều lần
                        logging.error(f"Exchange error fetching {symbol} {timeframe}: {e}. Stopping fetch for this timeframe.")
                        return None # Trả về None ngay lập tức
                    except Exception as e:
                        logging.error(f"Unexpected error fetching {symbol} {timeframe} (Attempt {attempt+1}/{max_retries}): {e}", exc_info=True)
                        if attempt < max_retries - 1:
                            wait_time = base_retry_delay * (attempt + 1)
                            await asyncio.sleep(wait_time)
                        else:
                            # Lỗi không xác định sau nhiều lần thử
                            logging.error(f"Max retries reached for {symbol} {timeframe} after unexpected error. Fetch failed.")
                            return None # Trả về None

                # Nếu vòng lặp for kết thúc mà không return thành công (hết số lần thử)
                logging.error(f"Failed to fetch data for {symbol} {timeframe} after {max_retries} attempts.")
                return None

            def _process_ohlcv(self, ohlcv: List) -> pd.DataFrame:
                # ... (Xử lý lỗi tốt hơn nếu ohlcv rỗng) ...
                if not ohlcv:
                    return pd.DataFrame() # Trả về DataFrame rỗng
                try:
                    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True) # Thêm utc=True
                    # Loại bỏ các dòng có timestamp trùng lặp, giữ dòng cuối
                    df = df.drop_duplicates(subset="timestamp", keep="last")
                    df = df.set_index("timestamp")
                    # Chuyển đổi kiểu dữ liệu sang số để tính toán
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce') # coerce sẽ biến lỗi thành NaN
                    # Optional: Xử lý NaN nếu cần thiết ngay tại đây
                    # df.dropna(subset=["open", "high", "low", "close"], inplace=True) # Xóa dòng có giá NaN
                    # df['volume'].fillna(0, inplace=True) # Fill volume NaN bằng 0
                    return df
                except Exception as e:
                    logging.error(f"Error processing OHLCV data: {e}", exc_info=True)
                    return pd.DataFrame() # Trả về DataFrame rỗng nếu lỗi

            def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
                # ... (Xử lý lỗi, kiểm tra NaN, dùng copy()) ...
                if not isinstance(df, pd.DataFrame) or df.empty:
                    # logging.warning("calculate_advanced_indicators: Input DataFrame is empty.")
                    return df # Trả về df rỗng/None
                if len(df) < 2:
                    # logging.warning("calculate_advanced_indicators: DataFrame too small for most indicators.")
                    return df # Trả về nếu quá nhỏ

                # Sử dụng copy để tránh cảnh báo SettingWithCopyWarning
                df = df.copy()

                # Kiểm tra các cột cơ bản
                required_cols = ["open", "high", "low", "close", "volume"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logging.error(f"Missing required columns in DataFrame for indicators: {missing_cols}")
                    return df # Trả về df gốc nếu thiếu cột

                # Kiểm tra NaN trong các cột cơ bản
                if df[required_cols].isnull().values.any():
                    initial_len = len(df)
                    # Chỉ xóa các dòng có NaN trong các cột giá, volume có thể fill
                    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                    df['volume'].fillna(0, inplace=True)
                    if len(df) < 2: # Kiểm tra lại sau khi xóa NaN
                        logging.warning(f"DataFrame too small after dropping NaNs in price columns.")
                        return df
                    logging.warning(f"Dropped {initial_len - len(df)} rows due to NaNs in core columns during indicator calculation.")


                # --- Tính toán chỉ báo (dùng try-except cho từng nhóm nếu cần) ---
                try:
                    # Sử dụng .loc để gán giá trị một cách an toàn
                    df.loc[:, "EMA_20"] = talib.EMA(df["close"], 20)
                    df.loc[:, "EMA_50"] = talib.EMA(df["close"], 50)
                    df.loc[:, "EMA_200"] = talib.EMA(df["close"], 200)
                    df.loc[:, "RSI"] = talib.RSI(df["close"], 14)
                    df.loc[:, "ATR"] = talib.ATR(df["high"], df["low"], df["close"], 14)
                    df.loc[:, "volatility"] = df["close"].rolling(20).std() # Biến động giá đóng cửa

                    macd, macd_signal, macd_hist = talib.MACD(df["close"], 12, 26, 9)
                    df.loc[:, "MACD"] = macd
                    df.loc[:, "MACD_signal"] = macd_signal
                    # df.loc[:, "MACD_hist"] = macd_hist # Thêm nếu cần

                    df.loc[:, "ADX"] = talib.ADX(df["high"], df["low"], df["close"], 14)
                    df.loc[:, "OBV"] = talib.OBV(df["close"], df["volume"])

                    bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"], 20)
                    df.loc[:, "BB_upper"] = bb_upper
                    df.loc[:, "BB_middle"] = bb_middle
                    df.loc[:, "BB_lower"] = bb_lower

                    # VWAP (cần tính lại nếu df bị cắt) - Cách tính này đúng nếu df là toàn bộ dữ liệu từ đầu session
                    # Nếu df chỉ là một phần, cần truyền dữ liệu tích lũy hoặc tính VWAP rolling
                    # Tạm thời giữ nguyên cách tính đơn giản
                    df.loc[:, "VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan) # Tránh chia cho 0

                    # Các chỉ báo tùy chỉnh
                    df = self.detect_regime_change(df) # Tính regime trước nếu các chỉ báo khác cần
                    df.loc[:, "divergence"] = self.detect_divergence(df)
                    df = self.detect_price_gaps(df)
                    df = self.create_temporal_features(df) # Tính features thời gian
                    df = self.detect_swing_points(df)
                    df.loc[:, "rejection"] = self.detect_rejection_candle(df)

                    # Volume Profile (có thể gây lỗi nếu df quá ngắn sau khi dropna)
                    if len(df) >= 100: # Kiểm tra độ dài tối thiểu cho volume profile
                        volume_profile = self.calculate_volume_profile(df)
                        df.loc[:, "VAH"] = volume_profile.get("VAH") # Dùng get để tránh lỗi nếu thiếu key
                        # Gán HVN/LVN: Có thể là list, cần xử lý phù hợp (ví dụ: lấy giá trị đầu tiên)
                        df.loc[:, "HVN"] = volume_profile.get("HVN", [np.nan])[0] if volume_profile.get("HVN") else np.nan
                        df.loc[:, "LVN"] = volume_profile.get("LVN", [np.nan])[0] if volume_profile.get("LVN") else np.nan
                    else:
                        df.loc[:, ["VAH", "HVN", "LVN"]] = np.nan

                    df.loc[:, "momentum"] = df["close"].pct_change(5)
                    # Tính order imbalance an toàn hơn
                    vol_roll_mean = df["volume"].rolling(5).mean().replace(0, np.nan)
                    df.loc[:, "order_imbalance"] = (df["volume"] - df["volume"].shift(1)) / vol_roll_mean

                except Exception as e:
                    logging.error(f"Error during indicator calculation: {e}", exc_info=True)
                    # Không trả về df nếu lỗi nghiêm trọng, hoặc trả về df với các cột đã tính được
                    # return df # Trả về df với những gì đã tính được

                # --- Xử lý NaN cuối cùng ---
                try:
                    filled_df = df.ffill().bfill()

                    result_df = filled_df.infer_objects(copy=True) # Hoặc để mặc định copy=True

                    # Bước 3: Kiểm tra và fill NaN còn sót lại (nếu có) bằng 0
                    if result_df.isnull().values.any():
                        logging.warning("NaNs still present after ffill/bfill/infer_objects. Filling remaining with 0.")
                        result_df = result_df.fillna(0)

                except Exception as e:
                    logging.error(f"Error during final NaN filling/inference in indicators: {e}", exc_info=True)
                    result_df = df.ffill().bfill().fillna(0)

                # Lưu vào cache (logic cache có thể đơn giản hóa)
                # if not df.empty: self.indicator_cache[str(df.index[-1])] = result_df

                return result_df
            
            def detect_regime_change(self, df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                # Đảm bảo cột 'regime' tồn tại trước khi cố gắng fit HMM
                if 'regime' not in df.columns: df['regime'] = 0.0 # Khởi tạo nếu chưa có

                try:
                    # Kiểm tra độ dài tối thiểu cho warmup và fit
                    min_len_hmm = CONFIG.get("hmm_warmup_max", 200) + 50 # Cần thêm data để fit
                    if len(df) < min_len_hmm:
                        # logging.warning(f"detect_regime_change: Not enough data ({len(df)}) for HMM. Min required: {min_len_hmm}. Using existing/default regime.")
                        df['regime'] = df['regime'].fillna(0).astype(int)
                        return df

                    warm_up = self.get_adaptive_warmup_period(df)
                    features = ["EMA_20", "RSI", "volatility"]
                    missing_features = [f for f in features if f not in df.columns]
                    if missing_features:
                        logging.error(f"detect_regime_change: Missing features {missing_features}. Using existing/default regime.")
                        df['regime'] = df['regime'].fillna(0).astype(int)
                        return df

                    # Chọn dữ liệu và xử lý NaN chỉ cho HMM
                    X_hmm_df = df[features].dropna()
                    if len(X_hmm_df) <= warm_up:
                        # logging.warning(f"detect_regime_change: Not enough valid data points ({len(X_hmm_df)}) after dropna for HMM warmup ({warm_up}). Using existing/default regime.")
                        df['regime'] = df['regime'].fillna(0).astype(int)
                        return df

                    X_hmm = X_hmm_df.values

                    # Huấn luyện và dự đoán HMM
                    try:
                        # Chỉ fit trên dữ liệu sau warmup
                        self.regime_model.fit(X_hmm[warm_up:])
                        # Dự đoán trên toàn bộ dữ liệu hợp lệ để có thể fill vào df gốc
                        regimes_pred = self.regime_model.predict(X_hmm)
                        # Tạo Series với index gốc
                        predicted_regime_series = pd.Series(regimes_pred, index=X_hmm_df.index)
                    except Exception as hmm_e:
                        logging.error(f"Error during HMM fit/predict: {hmm_e}. Using existing/default regime.")
                        # Không thay đổi cột regime hiện có nếu HMM lỗi
                        df['regime'] = df['regime'].fillna(0).astype(int)
                        return df

                    # Cập nhật cột 'regime' trong df gốc, ưu tiên giá trị mới từ HMM
                    # Dùng update để ghi đè chỉ số khớp
                    df.update(predicted_regime_series.rename('regime'))
                    # Fill các giá trị còn thiếu (thường ở đầu do warmup) bằng 0
                    df["regime"] = df["regime"].fillna(0).astype(int)
                    return df

                except Exception as e:
                    logging.error(f"Unexpected error in detect_regime_change: {e}", exc_info=True)
                    # Đảm bảo cột regime tồn tại và fill NaN
                    if 'regime' not in df.columns: df['regime'] = 0.0
                    df['regime'] = df['regime'].fillna(0).astype(int)
                    return df
                
            def get_adaptive_warmup_period(self, df: pd.DataFrame) -> int:
                default_warmup = CONFIG.get("hmm_warmup_max", 200)
                if 'close' not in df.columns or len(df) < 20: return default_warmup # Cần đủ data cho rolling

                try:
                    # Lấy giá trị cuối cùng, xử lý NaN
                    last_close = df["close"].iloc[-1]
                    mean_close = df["close"].mean() # Mean trên toàn bộ df có sẵn
                    volatility = df["close"].rolling(20).std().iloc[-1]

                    if pd.isna(volatility) or pd.isna(last_close) or pd.isna(mean_close) or mean_close == 0:
                        # logging.warning("NaN detected in warmup calculation inputs. Using default warmup.")
                        return default_warmup

                    # Tính toán warmup
                    warmup_min = CONFIG.get("hmm_warmup_min", 50)
                    warmup_max = CONFIG.get("hmm_warmup_max", 200)
                    # Sử dụng tỷ lệ biến động so với giá trung bình
                    vol_ratio = volatility / mean_close
                    # Công thức điều chỉnh warmup: cao hơn vol -> warmup ngắn hơn (ít data hơn)
                    warmup = int(warmup_max - vol_ratio * (warmup_max - warmup_min) * 5) # Nhân thêm hệ số để thay đổi rõ hơn
                    # Giới hạn trong khoảng min/max
                    return np.clip(warmup, warmup_min, warmup_max)
                except Exception as e:
                    logging.error(f"Error calculating adaptive warmup: {e}. Using default {default_warmup}.")
                    return default_warmup
                
            def detect_divergence(self, df: pd.DataFrame, swing_lookback: int = 5, max_bar_diff: int = 5) -> pd.Series:
                required_cols = ['high', 'low', 'close', 'RSI']
                if not all(col in df.columns for col in required_cols):
                    logging.error("detect_divergence: Missing required columns.")
                    return pd.Series(0.0, index=df.index)
                if len(df) < swing_lookback * 2 + 2:
                    logging.warning("detect_divergence: Not enough data.")
                    return pd.Series(0.0, index=df.index)

                df = df.copy()
                divergence = pd.Series(0.0, index=df.index)
                df_index = df.index # Lưu index đầy đủ để dùng get_indexer

                # 1. Xác định tất cả các điểm Swing
                df = self.detect_swing_points(df, lookback=swing_lookback)
                # Xác định swing cho RSI
                rsi_roll_max = df["RSI"].rolling(2 * swing_lookback + 1, center=True, min_periods=swing_lookback+1).max()
                rsi_roll_min = df["RSI"].rolling(2 * swing_lookback + 1, center=True, min_periods=swing_lookback+1).min()
                df['rsi_swing_high'] = (df['RSI'] == rsi_roll_max).fillna(False)
                df['rsi_swing_low'] = (df['RSI'] == rsi_roll_min).fillna(False)

                # Lấy indices của các điểm swing
                price_low_indices = df.index[df['swing_low']]
                price_high_indices = df.index[df['swing_high']]
                rsi_low_indices = df.index[df['rsi_swing_low']]
                rsi_high_indices = df.index[df['rsi_swing_high']]

                # --- 2. Phát hiện Phân Kỳ Dương (Bullish) ---
                # Lặp qua các cặp đáy giá liên tiếp
                for i in range(1, len(price_low_indices)):
                    prev_price_low_idx = price_low_indices[i-1]
                    current_price_low_idx = price_low_indices[i]

                    # Điều kiện 1: Giá tạo đáy thấp hơn
                    if df.loc[current_price_low_idx, 'low'] < df.loc[prev_price_low_idx, 'low']:
                        # Tìm điểm đáy RSI tương ứng gần nhất
                        prev_price_loc = df_index.get_loc(prev_price_low_idx)
                        current_price_loc = df_index.get_loc(current_price_low_idx)

                        prev_rsi_low_match_idx = self.find_closest_swing_idx(prev_price_loc, rsi_low_indices, df_index, max_bar_diff)
                        current_rsi_low_match_idx = self.find_closest_swing_idx(current_price_loc, rsi_low_indices, df_index, max_bar_diff)

                        # Điều kiện 2: Tìm thấy cả hai điểm RSI tương ứng và RSI tạo đáy cao hơn
                        if prev_rsi_low_match_idx is not None and current_rsi_low_match_idx is not None and \
                        current_rsi_low_match_idx > prev_rsi_low_match_idx and \
                        df.loc[current_rsi_low_match_idx, 'RSI'] > df.loc[prev_rsi_low_match_idx, 'RSI']:

                            # Đánh dấu phân kỳ dương tại điểm đáy giá hiện tại
                            divergence.loc[current_price_low_idx] = 1.0
                            logging.debug(f"Bullish Divergence detected at {current_price_low_idx}: Price Low {df.loc[current_price_low_idx, 'low']:.2f} < {df.loc[prev_price_low_idx, 'low']:.2f}, RSI Low {df.loc[current_rsi_low_match_idx, 'RSI']:.2f} > {df.loc[prev_rsi_low_match_idx, 'RSI']:.2f}")


                # --- 3. Phát hiện Phân Kỳ Âm (Bearish) ---
                # Lặp qua các cặp đỉnh giá liên tiếp
                for i in range(1, len(price_high_indices)):
                    prev_price_high_idx = price_high_indices[i-1]
                    current_price_high_idx = price_high_indices[i]

                    # Điều kiện 1: Giá tạo đỉnh cao hơn
                    if df.loc[current_price_high_idx, 'high'] > df.loc[prev_price_high_idx, 'high']:
                        # Tìm điểm đỉnh RSI tương ứng gần nhất
                        prev_price_loc = df_index.get_loc(prev_price_high_idx)
                        current_price_loc = df_index.get_loc(current_price_high_idx)

                        prev_rsi_high_match_idx = self.find_closest_swing_idx(prev_price_loc, rsi_high_indices, df_index, max_bar_diff)
                        current_rsi_high_match_idx = self.find_closest_swing_idx(current_price_loc, rsi_high_indices, df_index, max_bar_diff)

                        # Điều kiện 2: Tìm thấy cả hai điểm RSI tương ứng và RSI tạo đỉnh thấp hơn
                        if prev_rsi_high_match_idx is not None and current_rsi_high_match_idx is not None and \
                        current_rsi_high_match_idx > prev_rsi_high_match_idx and \
                        df.loc[current_rsi_high_match_idx, 'RSI'] < df.loc[prev_rsi_high_match_idx, 'RSI']:

                            # Đánh dấu phân kỳ âm tại điểm đỉnh giá hiện tại
                            divergence.loc[current_price_high_idx] = -1.0
                            logging.debug(f"Bearish Divergence detected at {current_price_high_idx}: Price High {df.loc[current_price_high_idx, 'high']:.2f} > {df.loc[prev_price_high_idx, 'high']:.2f}, RSI High {df.loc[current_rsi_high_match_idx, 'RSI']:.2f} < {df.loc[prev_rsi_high_match_idx, 'RSI']:.2f}")

                return divergence.fillna(0.0)
            
            def detect_swing_points(self, df: pd.DataFrame, lookback: int = 7, prominence_factor: float = 0.0015) -> pd.DataFrame:
                df = df.copy()
                required_cols = ['high', 'low', 'close'] # Thêm close để tính prominence tương đối
                if not all(col in df.columns for col in required_cols) or len(df) <= lookback:
                    logging.warning("detect_swing_points: Thiếu cột hoặc không đủ dữ liệu.")
                    df['swing_high'] = False
                    df['swing_low'] = False
                    return df

                min_prominence = df['close'].rolling(lookback * 2 + 1).mean().bfill().ffill() * prominence_factor
                # Đảm bảo prominence không phải NaN và có giá trị tối thiểu rất nhỏ
                min_prominence = min_prominence.fillna(1e-9)
                min_prominence = np.maximum(min_prominence, 1e-9)


                # --- Tìm Swing Highs (Peaks) ---
                try:
                    # find_peaks cần giá trị prominence cho từng điểm
                    high_peaks_indices, properties_high = find_peaks(
                        df['high'].values,             # Dữ liệu cần tìm đỉnh
                        distance=lookback,             # Khoảng cách tối thiểu giữa các đỉnh
                        prominence=min_prominence.values # Ngưỡng nổi bật tối thiểu cho từng điểm
                    )
                    df['swing_high'] = False
                    if len(high_peaks_indices) > 0:
                        # Sử dụng iloc để gán giá trị tại các vị trí số nguyên
                        df.iloc[high_peaks_indices, df.columns.get_loc('swing_high')] = True
                except Exception as e:
                    logging.error(f"Error finding high peaks: {e}", exc_info=True)
                    df['swing_high'] = False

                try:
                    # find_peaks cần giá trị prominence cho từng điểm (dùng cùng min_prominence)
                    low_peaks_indices, properties_low = find_peaks(
                        -df['low'].values,             # Dùng giá trị âm của low
                        distance=lookback,             # Khoảng cách tối thiểu giữa các đáy
                        prominence=min_prominence.values # Ngưỡng nổi bật tối thiểu
                    )
                    df['swing_low'] = False
                    if len(low_peaks_indices) > 0:
                        df.iloc[low_peaks_indices, df.columns.get_loc('swing_low')] = True
                except Exception as e:
                    logging.error(f"Error finding low peaks: {e}", exc_info=True)
                    df['swing_low'] = False

                df["swing_high"] = df["swing_high"].fillna(False)
                df["swing_low"] = df["swing_low"].fillna(False)

                return df
            
            def find_closest_swing_idx(self, target_idx_loc: int, swing_indices: pd.DatetimeIndex, df_index: pd.DatetimeIndex, max_diff: int) -> Optional[pd.Timestamp]:
                if swing_indices.empty:
                    return None

                # Lấy vị trí số nguyên của các điểm swing RSI
                try:
                    swing_idx_locs = df_index.get_indexer(swing_indices)
                    valid_swing_idx_locs = swing_idx_locs[swing_idx_locs != -1] # Loại bỏ các index không tìm thấy
                    if len(valid_swing_idx_locs) == 0:
                        return None
                except Exception as e:
                    logging.error(f"Error getting indexer for swings: {e}")
                    return None


                # Tính khoảng cách tuyệt đối về vị trí
                diffs = np.abs(valid_swing_idx_locs - target_idx_loc)

                # Tìm index có khoảng cách nhỏ nhất
                min_diff_idx = np.argmin(diffs)

                # Kiểm tra nếu khoảng cách nhỏ nhất nằm trong giới hạn max_diff
                if diffs[min_diff_idx] <= max_diff:
                    # Trả về DatetimeIndex tương ứng với vị trí tìm được
                    return df_index[valid_swing_idx_locs[min_diff_idx]]
                else:
                    return None
                
            def detect_rejection_candle(self, df: pd.DataFrame) -> pd.Series:
                df = df.copy()
                body = abs(df["close"] - df["open"])
                upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
                lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
                # Xử lý body = 0
                body_safe = body.replace(0, 1e-9)
                rejection = ((upper_wick > 2 * body) | (lower_wick > 2 * body)).astype(float).fillna(0.0)
                return rejection
            
            def detect_price_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
                # ... (Xử lý NaN cho ATR) ...
                df = df.copy()
                if 'ATR' not in df.columns or len(df) < 2: return df # Cần ATR và shift(1)

                df.loc[:, 'prev_close'] = df['close'].shift(1)
                df.loc[:, 'gap_size'] = df['open'] - df['prev_close']
                # Tính ngưỡng gap dựa trên ATR, xử lý ATR NaN
                atr_filled = df['ATR'].ffill().bfill().fillna(0)
                significant_threshold = 2 * atr_filled
                df.loc[:, 'significant_gap'] = abs(df['gap_size']) > significant_threshold

                # Logic lưu gap_levels giữ nguyên
                gap_indices = df.index[df['significant_gap'].fillna(False)] # fillna cho mask
                for idx in gap_indices[-CONFIG.get("max_gap_storage", 3):]: # Lấy từ CONFIG
                    row = df.loc[idx]
                    if pd.notna(row['prev_close']) and pd.notna(row['open']):
                        # Đảm bảo low < high cho Fib
                        fib_low = min(row['prev_close'], row['open'])
                        fib_high = max(row['prev_close'], row['open'])
                        if fib_high > fib_low:
                            price_levels = self._calculate_fib_levels(fib_low, fib_high)
                            self.gap_levels[idx] = {'type': 'Bullish' if row['gap_size'] > 0 else 'Bearish', 'levels': price_levels}
                return df
            
            def _calculate_fib_levels(self, low: float, high: float) -> dict:
                diff = high - low
                if diff <= 0: return {} # Trả về dict rỗng nếu high <= low
                levels_config = CONFIG.get("fib_levels", [0, 0.236, 0.382, 0.5, 0.618, 1])
                return {f'{level*100:.1f}%': high - diff * level for level in levels_config}
            
            def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                # Đảm bảo index là DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        original_index = df.index
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        if df.index.isnull().any():
                            logging.warning("Failed to convert index to DatetimeIndex in create_temporal_features. Reverting.")
                            df.index = original_index # Khôi phục index cũ nếu lỗi
                            # Gán giá trị mặc định cho các cột temporal
                            for col in self.temporal_features: df[col] = 0.0
                            return df
                    except Exception as e:
                        logging.error(f"Error converting index to DatetimeIndex: {e}")
                        for col in self.temporal_features: df[col] = 0.0
                        return df

                df.loc[:, 'hour'] = df.index.hour
                df.loc[:, 'day_of_week'] = df.index.dayofweek
                df.loc[:, 'is_weekend'] = df['day_of_week'].isin([5, 6]).astype(float)
                if 'high' in df.columns:
                    df.loc[:, 'time_since_last_high'] = self._time_since_last_event(df, 'high', compare_max=True).astype(float) # So sánh với max rolling
                else: df.loc[:, 'time_since_last_high'] = 0.0

                if 'significant_gap' in df.columns:
                    df.loc[:, 'time_since_last_gap'] = self._time_since_last_event(df, 'significant_gap').astype(float)
                else: df.loc[:, 'time_since_last_gap'] = 0.0

                if 'volatility' in df.columns:
                    df.loc[:, 'volatility_regime_change'] = self._detect_volatility_regime(df).astype(float)
                else: df.loc[:, 'volatility_regime_change'] = 0.0

                return df
            
            def _time_since_last_event(self, df: pd.DataFrame, col: str, compare_max: bool = False, window: int = 10) -> pd.Series:
                """Tính số nến kể từ lần cuối một sự kiện xảy ra (True trong cột `col` hoặc giá trị là max)."""
                time_since = pd.Series(0, index=df.index)
                event_indices = None

                if compare_max:
                    # Tìm index nơi giá trị bằng max trong cửa sổ trước đó
                    rolling_max = df[col].rolling(window=window, closed='left').max()
                    event_mask = (df[col] >= rolling_max) & (df[col] > df[col].shift(1)) # Là max và tăng so với trước
                    event_indices = df.index[event_mask]
                elif col in df.columns and df[col].dtype == 'bool':
                    event_indices = df.index[df[col]]
                elif col in df.columns: # Nếu cột không phải bool, coi giá trị > 0 là event
                    event_indices = df.index[df[col] > 0]

                if event_indices is None or event_indices.empty:
                    # Nếu không có event nào, trả về số thứ tự tăng dần (hoặc 0?)
                    return pd.Series(np.arange(len(df)), index=df.index)

                # Tính khoảng cách đến event gần nhất trước đó
                last_event_time = pd.Series(event_indices, index=event_indices)
                time_since = df.index.to_series().apply(lambda x: (x - last_event_time[last_event_time <= x].max()).days if not last_event_time[last_event_time <= x].empty else np.inf)
                event_marker = df[col] if not compare_max else event_mask
                if event_marker is None or event_marker.dtype != 'bool': event_marker = event_marker > 0 # Chuyển sang bool nếu cần
                counter = (~event_marker).cumsum()
                time_since = counter - counter[event_marker].ffill().fillna(0)

                return time_since.fillna(0)
            
            def _detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
                if 'volatility' not in df.columns: return pd.Series(0.0, index=df.index)
                vol_col = df['volatility'].dropna()
                if len(vol_col) < 50: return pd.Series(0.0, index=df.index) # Cần đủ dữ liệu cho rolling
                volatility_ewma = vol_col.ewm(span=40, adjust=False).mean()
                volatility_rolling_mean = volatility_ewma.rolling(50, min_periods=20).mean() # Cần min_periods
                regime = (volatility_ewma > volatility_rolling_mean).astype(float)
                # Tính diff, fillna(0), abs và reindex về index gốc của df
                regime_change = regime.diff().fillna(0).abs().reindex(df.index).fillna(0.0)
                return regime_change
            
            def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
                if len(df) < 100:
                    logging.warning("Not enough data for VAH state calculation in volume profile.")
                    return {"VAH": np.nan, "HVN": [np.nan], "LVN": [np.nan], "bin_size": CONFIG.get("vah_bin_range", (20, 60))[0]}

                state = self._get_vah_state(df.iloc[-100:]) # Lấy state từ 100 dòng cuối
                bin_size_default = CONFIG.get("vah_bin_range", (20, 60))[0]
                bin_size = bin_size_default

                if hasattr(self, "vah_optimizer") and self.vah_optimizer is not None:
                    try:
                        # predict trả về action (index) và state (thường là None cho DQN)
                        action, _ = self.vah_optimizer.predict(state, deterministic=True)
                        # Chuyển action index thành bin size
                        vah_bin_range = CONFIG.get("vah_bin_range", (20, 60))
                        num_choices = self.vah_optimizer.action_space.n
                        step = (vah_bin_range[1] - vah_bin_range[0]) / (num_choices - 1) if num_choices > 1 else 0
                        calculated_bin_size = vah_bin_range[0] + action * step
                        bin_size = max(2, int(calculated_bin_size)) # Đảm bảo >= 2
                        logging.debug(f"VAH Optimizer action: {action} -> Bin size: {bin_size}")
                    except Exception as e:
                        logging.error(f"Error in VAH optimizer prediction: {e}. Using default bin size {bin_size_default}.")
                        bin_size = bin_size_default
                # else: # Không có optimizer, đã dùng default

                # Tính toán volume profile với bin_size đã chọn
                try:
                    # Chỉ dùng dữ liệu gần đây để tính profile, ví dụ 200 nến
                    df_slice = df.tail(200)
                    low_min = df_slice['low'].min()
                    high_max = df_slice['high'].max()

                    if pd.isna(low_min) or pd.isna(high_max) or high_max <= low_min:
                        logging.warning("Invalid min/max prices for volume profile bins.")
                        raise ValueError("Invalid prices for bins")

                    bins = np.linspace(low_min, high_max, bin_size)
                    hist, bin_edges = np.histogram(df_slice['close'], bins=bins, weights=df_slice['volume'])

                    if hist.sum() == 0: # Không có volume trong slice
                        logging.warning("Zero volume in the slice for volume profile calculation.")
                        raise ValueError("Zero volume")

                    poc_index = np.argmax(hist) # Point of Control (bin có volume cao nhất)
                    vah = bin_edges[poc_index + 1] # Cạnh trên của bin POC

                    # Xác định HVN/LVN (ví dụ: dùng percentile)
                    volume_threshold_high = np.percentile(hist[hist > 0], 70) # 70th percentile của volume > 0
                    volume_threshold_low = np.percentile(hist[hist > 0], 30) # 30th percentile

                    hvn_indices = np.where(hist >= volume_threshold_high)[0]
                    lvn_indices = np.where(hist <= volume_threshold_low)[0]

                    # Lấy giá trị trung tâm của các bin HVN/LVN
                    hvn_prices = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in hvn_indices] if hvn_indices.size > 0 else [vah] # Fallback về VAH nếu không có HVN
                    lvn_prices = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in lvn_indices] if lvn_indices.size > 0 else [low_min] # Fallback về đáy nếu không có LVN

                    return { "VAH": vah, "HVN": hvn_prices, "LVN": lvn_prices, "bin_size": bin_size }

                except Exception as e:
                    logging.error(f"Error calculating volume profile: {e}")
                    return {"VAH": np.nan, "HVN": [np.nan], "LVN": [np.nan], "bin_size": bin_size}
            
            def _get_market_info(self, symbol: str) -> Optional[dict]:
                # Thêm cache đơn giản để tránh gọi API lặp lại nếu có thể
                if not hasattr(self, '_market_info_cache'):
                    self._market_info_cache = {} # Khởi tạo cache nếu chưa có

                if symbol in self._market_info_cache:
                    logging.debug(f"Using cached market info for {symbol}")
                    return self._market_info_cache[symbol]

                # Nếu không có trong cache, lấy từ exchange
                try:
                    # Đảm bảo exchange đã được khởi tạo và load markets
                    if hasattr(self, 'exchange') and self.exchange and \
                    hasattr(self.exchange, 'markets') and self.exchange.markets and \
                    len(self.exchange.markets) > 0: # Thêm kiểm tra markets có dữ liệu không
                        if symbol in self.exchange.markets:
                            market = self.exchange.market(symbol)
                            self._market_info_cache[symbol] = market # Lưu vào cache
                            logging.debug(f"Fetched and cached market info for {symbol}")
                            return market
                        else:
                            logging.warning(f"_get_market_info: Symbol '{symbol}' not found in loaded markets.")
                            return None # Trả về None nếu symbol không tồn tại
                    else:
                        logging.error(f"_get_market_info: Cannot get market info for {symbol}. Exchange not ready or markets not loaded.")
                        return None
                except ccxt.BadSymbol:
                    logging.error(f"_get_market_info: ccxt.BadSymbol error for '{symbol}'.")
                    return None
                except Exception as e:
                    logging.error(f"Error getting market info for {symbol}: {e}", exc_info=True)
                    return None
            
            def _safe_get_precision_digits(self, symbol: str, precision_type: str) -> int:
                """Helper để lấy số chữ số thập phân từ market info."""
                default_digits = 8 # Mặc định khá cao
                try:
                    market = self._get_market_info(symbol) # Dùng hàm helper đã có
                    if market and 'precision' in market and isinstance(market['precision'], dict):
                        precision_value = market['precision'].get(precision_type) # 'amount' hoặc 'price'
                        if precision_value is not None:
                            # Precision có thể là số tick (vd 0.01) hoặc số chữ số thập phân (vd 8)
                            # Nếu là số nhỏ hơn 1, tính số chữ số thập phân
                            if 0 < precision_value < 1:
                                # Tính số chữ số thập phân từ tick size (ví dụ 0.01 -> 2)
                                # Dùng log10, cẩn thận với giá trị cực nhỏ
                                if precision_value > 1e-12: # Tránh log(0)
                                    return max(0, -int(np.floor(np.log10(precision_value))))
                                else: return default_digits # Fallback nếu quá nhỏ
                            elif isinstance(precision_value, int) and precision_value >= 0:
                                return precision_value # Trả về trực tiếp nếu là số nguyên
                            else:
                                self.logger.warning(f"Invalid precision value '{precision_value}' for {symbol} {precision_type}. Using default.")
                                return default_digits
                        else:
                            # self.logger.debug(f"Precision type '{precision_type}' not found for {symbol}. Using default.")
                            return default_digits
                    else:
                        # self.logger.debug(f"Market info or precision not found for {symbol}. Using default.")
                        return default_digits
                except Exception as e:
                    self.logger.error(f"Error getting precision digits for {symbol} {precision_type}: {e}")
                    return default_digits

            def _safe_get_amount_precision_digits(self, symbol: str) -> int:
                """Lấy số chữ số thập phân cho số lượng (amount)."""
                return self._safe_get_precision_digits(symbol, 'amount')

            def _safe_get_price_precision_digits(self, symbol: str) -> int:
                """Lấy số chữ số thập phân cho giá (price)."""
                return self._safe_get_precision_digits(symbol, 'price')
            
            def _get_vah_state(self, df: pd.DataFrame) -> np.ndarray:
                try:
                    vol = df['volatility'].mean()
                    volume = df['volume'].mean()
                    std_pct = df['close'].pct_change().std()
                    # Tính CCI, ADOSC với lookback phù hợp (14 cho CCI, 3/10 cho ADOSC)
                    cci = talib.CCI(df['high'], df['low'], df['close'], 14).iloc[-1]
                    adosc = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10).iloc[-1]
                    regime = df['regime'].iloc[-1]

                    state_array = np.array([vol, volume, std_pct, cci, adosc, regime], dtype=np.float32)
                    # Xử lý NaN/Inf
                    if not np.isfinite(state_array).all():
                        # logging.warning("NaN/Inf detected in VAH state. Replacing with 0.")
                        state_array = np.nan_to_num(state_array, nan=0.0, posinf=0.0, neginf=0.0)
                    return state_array
                except Exception as e:
                    logging.error(f"Error calculating VAH state: {e}. Returning zeros.")
                    return np.zeros(6, dtype=np.float32)
                

try:
    from api import EnhancedTradingBot as ImportedEnhancedTradingBot, CONFIG
    print("Successfully imported EnhancedTradingBot and CONFIG from api.py")
    EnhancedTradingBot = ImportedEnhancedTradingBot
except ImportError as e:
    print(f"ImportError: {e}")
    print("Could not import from api.py.")
    print("Please ensure:")
    print("1. The bot code is in a file named 'api.py' in the same directory.")
    print("2. The necessary CONFIG dictionary is defined within that file.")
    if 'CONFIG' not in globals():
        print("Defining minimal CONFIG as fallback...")
        CONFIG = {
            "timeframes": ["1m", "5m", "15m", "1h", "4h"],
            "api_key": os.getenv("BINANCE_API_KEY"),
            "api_secret": os.getenv("BINANCE_API_SECRET"),
            "enable_rate_limit": True,
            "rate_limit_delay": 0.2,
        }
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        GaussianHMM = None
        logging.error("Thư viện hmmlearn không tìm thấy. Việc phát hiện regime có thể không khả dụng trong fallback.")
    print("Using MinimalBot as fallback.")
    EnhancedTradingBot = MinimalBot
    
# --- Cấu hình logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler("data_updater.log", encoding="utf-8"), logging.StreamHandler()]
)


# --- Thông tin cần cập nhật ---
SYMBOLS_TO_UPDATE = ["BTC/USDT", "ETH/USDT"]
PKL_FILES = {
    "BTC/USDT": "BTC_USDT_data.pkl",
    "ETH/USDT": "ETH_USDT_data.pkl",
}

async def update_data_file(symbol: str, pkl_filepath: str, bot: EnhancedTradingBot):
    """Cập nhật dữ liệu cho một symbol và lưu vào file pkl."""
    logging.info(f"--- Processing symbol: {symbol} ---")
    logging.info(f"Target file: {pkl_filepath}")

    # 1. Tải dữ liệu hiện có từ file pkl
    existing_data: Dict[str, Optional[pd.DataFrame]] = {}
    if os.path.exists(pkl_filepath):
        try:
            # Thử tải bằng joblib trước
            existing_data = joblib.load(pkl_filepath)
            logging.info(f"Loaded existing data for {symbol} using joblib.")
        except Exception as e_joblib:
            logging.warning(f"Failed to load {pkl_filepath} with joblib: {e_joblib}. Trying pickle...")
            try:
                with open(pkl_filepath, 'rb') as f:
                    existing_data = pickle.load(f)
                logging.info(f"Loaded existing data for {symbol} using pickle.")
            except Exception as e_pickle:
                logging.error(f"Failed to load {pkl_filepath} with pickle: {e_pickle}. Starting fresh fetch for affected timeframes.")
                existing_data = {} # Reset nếu không tải được
        # Kiểm tra xem có phải là dict không
        if not isinstance(existing_data, dict):
            logging.error(f"Data in {pkl_filepath} is not a dictionary. Starting fresh fetch.")
            existing_data = {}
    else:
        logging.warning(f"File {pkl_filepath} not found. Will fetch all data.")

    updated_data: Dict[str, Optional[pd.DataFrame]] = {} # Lưu kết quả cuối cùng
    fetch_errors = False

    # 2. Lặp qua các timeframe cần thiết từ CONFIG
    timeframes = CONFIG.get("timeframes", [])
    if not timeframes:
        logging.error("No timeframes defined in CONFIG. Cannot update.")
        return

    for tf in timeframes:
        logging.info(f"Processing timeframe: {tf} for {symbol}")
        df_old = existing_data.get(tf)
        last_timestamp_ms: Optional[int] = None

        # Xác định điểm bắt đầu fetch ('since')
        if isinstance(df_old, pd.DataFrame) and not df_old.empty:
            try:
                # Đảm bảo index là DatetimeIndex
                if not isinstance(df_old.index, pd.DatetimeIndex):
                    df_old.index = pd.to_datetime(df_old.index)
                # Lấy timestamp của dòng cuối cùng
                last_timestamp = df_old.index[-1]
                if pd.notna(last_timestamp):
                    # Chuyển sang milliseconds UTC
                    last_timestamp_ms = int(last_timestamp.timestamp() * 1000)
                    logging.debug(f"Last timestamp found for {symbol} {tf}: {last_timestamp} ({last_timestamp_ms})")

                    # Tính 'since' cho lần fetch tiếp theo (thời điểm bắt đầu của NẾN TIẾP THEO)
                    timeframe_duration_ms = bot.exchange.parse_timeframe(tf) * 1000
                    fetch_since_ms = last_timestamp_ms + timeframe_duration_ms
                    logging.info(f"Will fetch new data for {symbol} {tf} since {pd.to_datetime(fetch_since_ms, unit='ms', utc=True)}")
                else:
                    logging.warning(f"Could not get valid last timestamp for {symbol} {tf}. Will fetch recent data.")
                    fetch_since_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000) # Lấy 30 ngày gần nhất nếu lỗi

            except Exception as e:
                logging.error(f"Error getting last timestamp for {symbol} {tf}: {e}. Will fetch recent data.")
                fetch_since_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000) # Lấy 30 ngày gần nhất nếu lỗi
        else:
            logging.warning(f"No existing data or empty DataFrame for {symbol} {tf}. Fetching last ~5 years.")
            # Lấy dữ liệu từ thời điểm xa hơn nếu chưa có gì
            days_back = 1825 # ~5 năm
            fetch_since_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

        # 3. Fetch dữ liệu mới
        try:
            # Sử dụng hàm _fetch_ohlcv của bot (đã được copy/adapt ở phần fallback)
            new_ohlcv_list_raw = await bot._fetch_ohlcv(symbol, tf, since=fetch_since_ms)
            df_new = None
            if new_ohlcv_list_raw: # Kiểm tra xem list có rỗng không
                # CHỈ GỌI _process_ohlcv KHI CÓ DỮ LIỆU THÔ
                df_new = bot._process_ohlcv(new_ohlcv_list_raw)

                # KIỂM TRA DataFrame SAU KHI XỬ LÝ
                if df_new is None or df_new.empty: # Kiểm tra xem df_new có rỗng hoặc None không
                    logging.info(f"No processable new data fetched for {symbol} {tf}.")
                    df_new = None # Đảm bảo là None nếu rỗng/lỗi xử lý
                else:
                    logging.info(f"Fetched and processed {len(df_new)} new candles for {symbol} {tf}.")
            else:
                logging.info(f"No new data returned by fetch for {symbol} {tf}.")
        except Exception as e:
            logging.error(f"Error fetching new data for {symbol} {tf}: {e}", exc_info=True)
            df_new = None
            fetch_errors = True # Đánh dấu có lỗi

        # 4. Kết hợp dữ liệu cũ và mới
        df_combined = None
        if isinstance(df_old, pd.DataFrame) and not df_old.empty and isinstance(df_new, pd.DataFrame) and not df_new.empty:
            try:
                # Đảm bảo cả hai đều có DatetimeIndex
                if not isinstance(df_old.index, pd.DatetimeIndex):
                     df_old.index = pd.to_datetime(df_old.index)
                if not isinstance(df_new.index, pd.DatetimeIndex):
                     df_new.index = pd.to_datetime(df_new.index)

                df_combined = pd.concat([df_old, df_new])
                # Loại bỏ trùng lặp, giữ bản ghi mới nhất (last)
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined = df_combined.sort_index() # Sắp xếp lại theo thời gian
                logging.info(f"Combined old and new data for {symbol} {tf}. Total rows: {len(df_combined)}")
            except Exception as e:
                logging.error(f"Error combining data for {symbol} {tf}: {e}. Using only old data if available.")
                df_combined = df_old # Fallback về dữ liệu cũ
        elif isinstance(df_new, pd.DataFrame) and not df_new.empty:
            df_combined = df_new # Chỉ có dữ liệu mới
            logging.info(f"Using newly fetched data for {symbol} {tf}.")
        elif isinstance(df_old, pd.DataFrame) and not df_old.empty:
            df_combined = df_old # Chỉ có dữ liệu cũ (nếu fetch lỗi)
            logging.info(f"Using only existing data for {symbol} {tf} (fetch failed or no new data).")
        else:
            logging.warning(f"No valid data (old or new) available for {symbol} {tf}.")
            df_combined = None # Không có dữ liệu gì cả

        # 5. Tính toán lại chỉ báo trên dữ liệu kết hợp
        df_final = None
        if isinstance(df_combined, pd.DataFrame) and not df_combined.empty:
            logging.info(f"Calculating indicators for {symbol} {tf} on {len(df_combined)} rows...")
            try:
                # QUAN TRỌNG: Gọi hàm tính chỉ báo của bot
                df_final = bot.calculate_advanced_indicators(df_combined)
                if isinstance(df_final, pd.DataFrame) and not df_final.empty:
                     # Lưu thông tin precision vào attrs của DataFrame (ví dụ)
                    try:
                        market_info = bot._get_market_info(symbol) # Lấy market info
                        if market_info:
                             df_final.attrs['price_precision_digits'] = bot._safe_get_price_precision_digits(symbol)
                             df_final.attrs['amount_precision_digits'] = bot._safe_get_amount_precision_digits(symbol)
                    except Exception as attr_e:
                         logging.warning(f"Could not get/set precision attributes for {symbol}: {attr_e}")

                    logging.info(f"Indicators calculated for {symbol} {tf}. Final rows: {len(df_final)}")
                else:
                    logging.error(f"Indicator calculation failed or returned empty for {symbol} {tf}.")
                    df_final = None # Đặt là None nếu lỗi
            except Exception as e:
                logging.error(f"Error calculating indicators for {symbol} {tf}: {e}", exc_info=True)
                df_final = None # Đặt là None nếu lỗi
        else:
            logging.warning(f"Skipping indicator calculation for {symbol} {tf} due to no combined data.")

        # Lưu kết quả vào dict cuối cùng
        updated_data[tf] = df_final

    # 6. Lưu dữ liệu đã cập nhật vào file pkl
    # Chỉ lưu nếu không có lỗi nghiêm trọng khi fetch (có thể xem xét lại)
    if not fetch_errors and updated_data:
        try:
            logging.info(f"Saving updated data for {symbol} to {pkl_filepath}...")
            joblib.dump(updated_data, pkl_filepath)
            logging.info(f"Successfully saved updated data for {symbol}.")
        except Exception as e:
            logging.error(f"Failed to save updated data for {symbol} to {pkl_filepath}: {e}", exc_info=True)
    elif fetch_errors:
         logging.error(f"Skipping save for {symbol} due to fetch errors.")
    else:
         logging.warning(f"No data to save for {symbol}.")

    logging.info(f"--- Finished processing symbol: {symbol} ---")


async def main_update():
    """Hàm chính để chạy quá trình cập nhật."""
    bot = None
    try:
        # Khởi tạo Bot (hoặc ít nhất là phần exchange và các hàm cần thiết)
        # Đảm bảo rằng lớp EnhancedTradingBot và CONFIG đã được định nghĩa/import
        if 'EnhancedTradingBot' not in globals():
             logging.critical("EnhancedTradingBot class not found. Cannot proceed.")
             return
        bot = EnhancedTradingBot() # Tạo instance
        await bot._initialize_exchange() # Chỉ cần khởi tạo exchange

        tasks = []
        for symbol in SYMBOLS_TO_UPDATE:
            if symbol in PKL_FILES:
                tasks.append(update_data_file(symbol, PKL_FILES[symbol], bot))
            else:
                logging.warning(f"No PKL file defined for symbol: {symbol}. Skipping.")

        if tasks:
            await asyncio.gather(*tasks) # Chạy cập nhật song song cho các symbol

    except Exception as e:
        logging.critical(f"Critical error during data update process: {e}", exc_info=True)
    finally:
        if bot and bot.exchange:
            logging.info("Closing exchange connection...")
            await bot.exchange.close()
            logging.info("Exchange connection closed.")
        logging.info("Data update script finished.")

if __name__ == "__main__":
    # Thiết lập policy event loop cho Windows nếu cần
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_update())
    except KeyboardInterrupt:
        logging.info("Update script interrupted by user.")