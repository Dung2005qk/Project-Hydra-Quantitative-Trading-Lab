import ccxt.async_support as ccxt  
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, RegressorMixin
from imblearn.over_sampling import RandomOverSampler
import joblib
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
import talib
import matplotlib.pyplot as plt
import os
import asyncio
import sys
import math
import aiohttp
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from transformers import pipeline
from scipy.optimize import minimize
from scipy.signal import find_peaks
from hmmlearn.hmm import GaussianHMM
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
import time
import json
import copy
import optuna
import traceback
from torch_geometric.nn import GCNConv
from news_sentiment_analyzer import NewsSentimentAnalyzer
from dotenv import load_dotenv 
from pathlib import Path
from realtime_monitor import RealtimeMonitor

load_dotenv()
fmp_api_key = os.getenv("FMP_API_KEY")
av_api_key = os.getenv("AV_API_KEY")
binance_api_key = os.getenv("BINANCE_API_KEY")
binance_api_secret = os.getenv("BINANCE_API_SECRET")
if not binance_api_key or not binance_api_secret:
    logging.warning("BINANCE_API_KEY or BINANCE_API_SECRET not set. Exchange functionality limited.")
if not fmp_api_key:
    logging.warning("FMP_API_KEY not set. Sentiment features using FMP might fail.")
if not av_api_key:
    logging.warning("AV_API_KEY not set. Sentiment/Market data using Alpha Vantage might fail.")

try:
    from torch_geometric.nn import GATConv as GraphConv # Dùng GAT như trong huấn luyện
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool, global_max_pool
    _torch_geometric_available = True
    logging.info("Successfully imported torch_geometric for api.py.")
except ImportError:
    logging.error("torch_geometric not available. GNN anomaly detection will not function.")
    # Định nghĩa lớp giả lập để tránh lỗi nếu không có PyG
    class GraphConv:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x, edge_index, **kwargs): return x # Trả về input
    class GATConv: # Thêm cả GATConv giả
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x, edge_index, **kwargs): return x
    def global_mean_pool(x, batch): return torch.zeros(batch.max().item()+1, x.size(1), device=x.device)
    def global_max_pool(x, batch): return torch.zeros(batch.max().item()+1, x.size(1), device=x.device)
    _torch_geometric_available = False

SCRIPT_DIR_API = Path(__file__).resolve().parent 
GNN_TRAINED_DIR = SCRIPT_DIR_API / "final_trained_gnn" 
GNN_MODEL_PATH = GNN_TRAINED_DIR / "final_iss_gnn_anomaly.pth"
GNN_HPARAMS_PATH = GNN_TRAINED_DIR / "best_gnn_hparams.json"
GNN_SCALER_PATH = GNN_TRAINED_DIR / "final_iss_gnn_scaler.pkl"
THRESHOLD_SAVE_PATH = GNN_TRAINED_DIR / "best_gnn_threshold.json"
LOCK_ACQUIRE_TIMEOUT = 10.0

# Kiểm tra sự tồn tại của các file cần thiết 
if not GNN_MODEL_PATH.exists():
    logging.warning(f"Trained GNN model not found at: {GNN_MODEL_PATH}. Anomaly detection might be limited.")
if not GNN_HPARAMS_PATH.exists():
    logging.warning(f"GNN hyperparameters file not found at: {GNN_HPARAMS_PATH}. Using default GNN structure.")
if not GNN_SCALER_PATH.exists():
    logging.warning(f"GNN scaler file not found at: {GNN_SCALER_PATH}. GNN input scaling might fail.")
if not THRESHOLD_SAVE_PATH.exists():
    logging.warning(f"Best GNN threshold file not found at: {THRESHOLD_SAVE_PATH}. Using default threshold 0.5.")
    

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", 
    handlers=[logging.FileHandler("enhanced_bot.log", encoding="utf-8"), logging.StreamHandler()]
)
try:
    from TP_SL_logic import (
        CryptoDecisionSupportModel, 
        CryptoTradingModel,         
        CombinedTradingAgent,       
        PositionalEncoding,        
        apply_basic_feature_engineering,
        apply_mta_advanced_feature_engineering, 
        DEFAULT_NUM_REGIMES,      
        REGIME_MAP
    )
    _new_models_available = True
    logging.info("Successfully imported new models (Decision, Hybrid, Agent).")
except ImportError as import_err:
    logging.error(f"Could not import new models module: {import_err}. CombinedTradingAgent functionality disabled.")
    # Tạo lớp/hàm giả lập để tránh lỗi runtime nếu không import được
    class CryptoDecisionSupportModel: pass
    class CryptoTradingModel: pass
    class CombinedTradingAgent: pass
    class PositionalEncoding: pass
    # Hàm giả lập cần trả về shape đúng hoặc giá trị mặc định
    def apply_basic_feature_engineering(x, input_dim): return x, x.shape[-1]
    def apply_mta_advanced_feature_engineering(x_dict, **kwargs):
        key = list(x_dict.keys())[0] # Lấy key đầu tiên làm fallback
        return x_dict[key], x_dict[key].shape[-1]
    DEFAULT_NUM_REGIMES = 3
    REGIME_MAP = {0: "TREND_UP", 1: "SIDEWAYS", 2: "TREND_DOWN", -1: "UNKNOWN"} # Map fallback
    _new_models_available = False

# Cấu hình tối ưu cho PNL và tỷ lệ thắng
CONFIG = {
    # --- General Bot Settings ---
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframes": ["15m", "1h", "4h"],       # <<< Sử dụng '15m'
    "primary_tf": "15m",                    # <<< Sử dụng '15m'
    "days_to_fetch": 1825,                   # Số ngày dữ liệu lịch sử cần lấy
    "initial_balance": 10000,
    "leverage": 5,                           # Đòn bẩy (cẩn thận khi sử dụng)
    "fee_rate": 0.0005,                      # Tỷ lệ phí giao dịch (ví dụ 0.05%)
    "max_trades_per_day": 15,                # Giới hạn số lệnh mỗi ngày
    "enable_rate_limit": True,               # Bật giới hạn tỷ lệ gọi API ccxt
    "rate_limit_delay": 0.2,                 # Thời gian chờ (giây) giữa các lệnh gọi API nếu rate limit bật
    "websocket_url": "wss://stream.binance.com:9443/ws", # URL WebSocket (ít dùng nếu có Monitor riêng)
    "api_key": os.getenv("BINANCE_API_KEY", "YOUR_API_KEY"), # Lấy từ biến môi trường
    "api_secret": os.getenv("BINANCE_API_SECRET", "YOUR_SECRET"), # Lấy từ biến môi trường
    "use_testnet": False,                    # Đặt thành True nếu dùng testnet Binance

    # --- Data Handling & Caching ---
    "embeddings_cache_dir": "embeddings_cache", # Thư mục lưu cache embedding
    "precompute_embeddings_on_start": True,  # Có tính embedding khi khởi động không
    "use_data_cache": True,                  # Có sử dụng cache dữ liệu OHLCV không
    "save_data_cache": True,                 # Có lưu dữ liệu OHLCV vào cache không
    "min_samples": 150,                      # Số mẫu tối thiểu cho huấn luyện ML
    "lookback_window": 10,                   # Cửa sổ lookback chung (ít dùng?)
    "hmm_warmup_min": 50,                    # Warmup tối thiểu cho các tính toán phụ thuộc độ dài
    "hmm_warmup_max": 200,                   # Warmup tối đa

    # --- Risk & Position Sizing ---
    "risk_per_trade": 0.06,                  # Rủi ro cơ sở cho mỗi giao dịch (tỷ lệ balance)
    "max_account_risk": 0.07,                # Tổng rủi ro tối đa tài khoản cho phép tại một thời điểm
    "min_position_size": 0.02,               # Kích thước vị thế tối thiểu (tỷ lệ balance)
    "max_position_size": 0.5,                # Kích thước vị thế tối đa (tỷ lệ balance)
    "max_exposure": 0.75,                    # Tổng giá trị danh nghĩa vị thế tối đa (tỷ lệ balance)
    "symbol_risk_modifiers": {               # Hệ số điều chỉnh rủi ro theo symbol
        "BTC": 1.25,
        "ETH": 1.10,
        "DEFAULT": 0.90
    },

    # --- Feature & Indicator Settings ---
    "input_dim": 5,                          # OHLCV
    "vah_bin_range": (20, 60),               # Khoảng bin size cho VAH Optimizer
    "fib_levels": [0, 0.236, 0.382, 0.5, 0.618, 1], # Các mức Fibonacci sử dụng
    "max_gap_storage": 3,                    # Số lượng gap giá gần nhất để lưu
    "volume_spike_multiplier": 4.0,          # Hệ số xác định volume spike
    "volume_sustained_multiplier": 1.5,      # Hệ số xác định volume duy trì
    "volume_ma20_threshold": 0.8,            # Ngưỡng volume so với MA20
    "temporal_features": ['hour', 'day_of_week', 'is_weekend'], # Các feature thời gian được thêm tự động

    # --- Circuit Breaker ---
    "circuit_breaker_threshold": 4.0,        # Ngưỡng ATR multiplier để kích hoạt CB
    "circuit_breaker_cooldown": 60,          # Thời gian (phút) chờ để reset CB

    # --- External Model Integration (XGB/SAC/DQN) ---
    "xgb_long_threshold": 0.65,
    "xgb_short_threshold": 0.35,
    "min_xgb_feature_dim": 20,               # Số features tối thiểu để chạy XGBoost
    "xgboost_feature_columns": [             # <<< Cần khớp với features khi train XGB >>>
        "EMA_diff", "RSI", "ATR", "MACD", "MACD_signal", "ADX", "OBV",
        "BB_width", "volatility", "log_volume", "volume_anomaly",
        "regime", "VWAP_ADX_interaction", "BB_EMA_sync",
        "VAH", "divergence", "swing_high", "swing_low", "rejection", "momentum", "order_imbalance",
        "hybrid_regime" # <<< Thêm regime từ Hybrid làm feature >>>
        # Temporal features sẽ được thêm động khi chuẩn bị dữ liệu ML
    ],
    "sac_long_threshold": 0.1,
    "sac_short_threshold": -0.1,
    "dqn_size_adj_enabled": True,            # Bật/tắt điều chỉnh size bằng DQN
    "win_prob_dqn_override": 0.80,           # Win prob cao để ghi đè DQN=0
    "win_prob_dqn_boost_thresh": 0.70,       # Win prob cao để DQN=2 tăng risk mạnh hơn

    # --- Sentiment ---
    "sentiment_window": 24,                  # Cửa sổ (giờ) lấy tin tức/sentiment
    "sentiment_veto_threshold": 0.65,        # Ngưỡng sentiment để phủ quyết tín hiệu
    "sentiment_influence_factor": 0.15,      # Hệ số ảnh hưởng của sentiment lên risk factor

    # --- LOB Transformer ---
    "lob_input_dim": 20,                     # Số features từ LOB (ví dụ: 5 mức giá/vol bid/ask)
    "lob_d_model": 64,
    "lob_nhead": 4,
    "lob_num_layers": 3,

    # --- Intelligent Stop System (ISS) ---
    "use_intelligent_stop_system": True,     # Bật/tắt hệ thống SL thông minh
    "iss_transformer_d_model": 256,          # Hyperparameter cho ISS Transformer
    "iss_transformer_nhead": 16,
    "iss_transformer_num_layers": 2,
    "iss_transformer_dropout": 0.11418798847562715,
    # Lưu ý: Các tham số GNN của ISS được load từ file hparams riêng

    # --- MLP Action Analyzer ---
    "mlp_action_input_dim": 16,              # Số features đầu vào cho MLPAction
    "mlp_action_output_dim": 1,
    "mlp_action_hidden_dims_default": [245, 242, 234], # Cấu trúc mặc định (từ Optuna)
    "mlp_action_dropout_p_default": 0.1508988103823329, # Dropout mặc định (từ Optuna)
    "mlp_action_activation_default": "GELU", # Activation mặc định (từ Optuna)

    # --- Decision Support Model (Transformer for SL/RR) ---
    "DECISION_SEQ_LEN": 60,                  # Độ dài chuỗi input
    "decision_d_model": 64,                  # Kích thước embedding
    "decision_nhead": 4,                     # Số lượng attention heads
    "decision_num_encoder_layers": 3,        # Số lớp encoder
    "decision_dim_feedforward": 128,         # Kích thước lớp feedforward
    "decision_num_rr_levels": 10,            # Số mức RR dự đoán
    "decision_dropout_rate": 0.1,            # Tỷ lệ dropout

    # --- Hybrid Model (CNN-LSTM for Regime/Forecast) ---
    "required_tfs_hybrid": ('15m', '1h'),     # <<< Sử dụng '15m' >>>
    "HYBRID_SEQ_LEN_MAP": {'15m': 60, '1h': 15, '4h': 4 }, # <<< Sử dụng '15m', thêm '4h' nếu cần bởi classify_regime >>>
    "hybrid_hidden_dim": 128,                # Kích thước ẩn CNN/LSTM
    "hybrid_num_regimes": DEFAULT_NUM_REGIMES, # Số regimes (ví dụ: 3)
    "hybrid_cnn_kernel_size": 3,             # Kích thước kernel CNN
    "hybrid_lstm_layers": 2,                 # Số lớp LSTM
    "hybrid_use_attention": False,           # Có dùng attention sau LSTM không
    "hybrid_dropout_rate": 0.1,              # Tỷ lệ dropout

    # ===========================================================
    # --- CombinedTradingAgent Specific Settings - START        ---
    # ===========================================================

    # --- Market Classification Thresholds (Used by Agent's classify_market_regime) ---
    "adx_threshold":22,
    "adx_thresholds": { "15m": 22, "1h": 22, "4h": 25, "default": 20 }, # <<< Sử dụng '15m' >>> # Ngưỡng ADX theo TF (Agent dùng để check strong signal)
    "adx_strong_thresh": 25,                 # Ngưỡng ADX xác định trend mạnh trong classify_market_regime
    "adx_very_strong_thresh": 35,            # Ngưỡng ADX xác định trend rất mạnh trong classify_market_regime
    "sideway_adx_thresh": 20,                # Ngưỡng ADX xác định sideways trong classify_market_regime
    "ma_slope_thresh_strong": 0.005,         # Ngưỡng độ dốc MA mạnh trong classify_market_regime
    "ma_slope_thresh_normal": 0.002,         # Ngưỡng độ dốc MA thường trong classify_market_regime
    "trend_volume_change_thresh": 1.15,      # Ngưỡng thay đổi volume cho trend trong classify_market_regime
    "sideway_volume_cv_thresh": 0.35,        # Ngưỡng hệ số biến thiên volume cho sideways trong classify_market_regime
    "bb_width_vol_threshold": 0.05,          # <<< Dùng để chọn adaptive weights >>>
    "sideway_bb_width_thresh": 0.04,         # Ngưỡng BBW hẹp cho sideways trong classify_market_regime
    "trend_bb_width_expand_thresh": 0.06,    # Ngưỡng BBW rộng cho trend trong classify_market_regime
    "trend_very_strong_thresh": 4.0,         # Ngưỡng điểm số để phân loại trend rất mạnh
    "trend_strong_thresh": 2.5,              # Ngưỡng điểm số để phân loại trend mạnh
    "sideway_strong_thresh": 3.0,            # Ngưỡng điểm số để phân loại sideways mạnh
    "sideway_normal_thresh": 1.5,            # Ngưỡng điểm số để phân loại sideways thường

    # --- Combined Signal Weights & Thresholds (Used by Agent's _generate_entry_signal) ---
    "ADAPTIVE_WEIGHTS": {                    # <<< Bổ sung keys cho logic mới >>>
        'trending_strong': {'sac': 0.3, 'xgb': 0.7},
        'trending': {'sac': 0.4, 'xgb': 0.6}, # Bao gồm cả trending normal/weak
        'sideways_high_vol': {'sac': 0.7, 'xgb': 0.3},
        'sideways_low_vol': {'sac': 0.5, 'xgb': 0.5}, # Bao gồm cả sideways normal/weak/strong nếu vol thấp
        'default': {'sac': 0.5, 'xgb': 0.5}  # Fallback và UNKNOWN
    },
    "signal_strong_threshold": 0.75,         # Ngưỡng điểm tổng hợp cho tín hiệu mạnh
    "signal_medium_threshold": 0.65,         # Ngưỡng điểm tổng hợp cho tín hiệu trung bình
    "signal_weak_threshold": 0.55,           # Ngưỡng điểm tổng hợp cho tín hiệu yếu

    # --- Enhanced Take Profit Logic (Used by Agent's get_trade_signals) ---
    "agent_min_rr_level_for_check": 1,       # Bắt đầu quản lý TP từ RR1
    "agent_max_rr_level_for_check": 10,      # Quản lý TP tối đa đến RR10 (phải <= decision_num_rr_levels)
    "regime_aware_rr_prob_thresholds": {     # Ngưỡng xác suất RR cơ sở để nhắm mục tiêu cao hơn
        'TREND_VERY_STRONG': 0.55, 'TREND_STRONG': 0.58, 'TREND_NORMAL': 0.60,
        'SIDEWAYS_STRONG': 0.65, 'SIDEWAYS_NORMAL': 0.65, 'SIDEWAYS_WEAK': 0.70,
        'UNKNOWN': 0.75 },
    "conservative_rr_prob_threshold": 0.75,
    "prob_threshold": 0.75,  # Ngưỡng RR prob khi có conflict hoặc UNKNOWN
    "low_confidence_threshold": 0.60,        # Ngưỡng confidence để kích hoạt penalty
    "confidence_penalty_factor": 1.15,       # Hệ số tăng ngưỡng RR prob khi confidence thấp
    "hybrid_trend_strong_pass": {'min_steps_confirm': 3, 'max_pullback_ratio': 0.3, 'min_target_cross': 1}, # Config kiểm tra Hybrid cho trend mạnh
    "hybrid_trend_normal_pass": {'min_steps_confirm': 2, 'max_pullback_ratio': 0.5, 'min_target_cross': 1}, # Config kiểm tra Hybrid cho trend thường
    "hybrid_sideways_pass": {'min_target_cross': 1},     # Config kiểm tra Hybrid cho sideways/conservative
    "lock_floor_trend_very_strong_rr_thresh": 0.80, # Ngưỡng RR prob để khóa sàn TP (Trend Very Strong)
    "lock_floor_trend_strong_rr_thresh": 0.85,      # Ngưỡng RR prob để khóa sàn TP (Trend Strong)
    "lock_floor_trend_normal_rr_thresh": 0.90,      # Ngưỡng RR prob để khóa sàn TP (Trend Normal)
    "lock_floor_sideways_rr_thresh": 0.98,          # Ngưỡng RR prob để khóa sàn TP (Sideways)
    "significant_regime_change_def": [ ("TREND_UP", "TREND_DOWN") ], # <<< Chỉ thoát khi đảo chiều >>>
    "enable_dynamic_rr_exit": True,          # Bật/tắt kiểm tra thoát lệnh R:R động
    "exit_threshold_dynamic_RR": 1.0,        # Ngưỡng R:R động để thoát (Reward < X * Risk)
    "min_risk_to_floor_ratio": 0.1,          # Tỷ lệ rủi ro tối thiểu (so với 1R) để kích hoạt kiểm tra R:R động

    # --- Agent SL Limits ---
    "agent_min_sl_atr_multiplier": 0.5,      # Giới hạn dưới cho SL multiplier từ Decision Model
    "agent_max_sl_atr_multiplier": 5.0,      # Giới hạn trên

    # ===========================================================
    # --- CombinedTradingAgent Specific Settings - END          ---
    # ===========================================================

    # --- Monitor Specific Config (Optional - For RealtimeMonitor) ---
    "monitor_per_symbol_config": {
        "BTCUSDT": {"vol_check_interval": 0.8, "vol_threshold_pct": 0.1, "event_throttle_ms": 400, "rr_crossing_buffer_pct": 0.0003},
        "ETHUSDT": {"vol_check_interval": 1.2, "vol_threshold_pct": 0.15, "event_throttle_ms": 600, "rr_crossing_buffer_pct": 0.0006}
    },
    "monitor_max_rr_level": 10.0,             # Ví dụ: Monitor chỉ theo dõi đến RR10 để gửi sự kiện RR_CROSSED
    "monitor_rr_step": 0.5,                  # Ví dụ: Bước nhảy RR cho monitor tính toán các mức giá RR
}
EMBEDDING_DIM = CONFIG.get("decision_d_model", 64) 
MLP_INPUT_DIM = CONFIG.get("mlp_action_input_dim", 16) # Lấy từ config hoặc default
MLP_OUTPUT_DIM = CONFIG.get("mlp_action_output_dim", 1)
MLP_HIDDEN_DIMS_DEFAULT = CONFIG.get("mlp_action_hidden_dims_default", [245, 242, 234]) # Default từ Optuna
MLP_DROPOUT_DEFAULT = CONFIG.get("mlp_action_dropout_p_default", 0.1508988103823329) # Default từ Optuna
MLP_ACTIVATION_DEFAULT_NAME = CONFIG.get("mlp_action_activation_default", "GELU") # Default từ Optuna

# Xác định lớp activation mặc định từ tên
if hasattr(nn, MLP_ACTIVATION_DEFAULT_NAME):
    MLP_ACTIVATION_DEFAULT_CLS = getattr(nn, MLP_ACTIVATION_DEFAULT_NAME)
else:
    logging.warning(f"Default activation '{MLP_ACTIVATION_DEFAULT_NAME}' not found. Using GELU.")
    MLP_ACTIVATION_DEFAULT_CLS = nn.GELU

SCRIPT_DIR_API = Path(__file__).resolve().parent # Lấy đường dẫn của api.py
MODEL_SAVE_DIR_ISS = SCRIPT_DIR_API / "trained_models_iss"
ISS_MODEL_PATH = MODEL_SAVE_DIR_ISS / "iss_transformer_forecaster_final_optuna.pth"
ISS_INPUT_SCALER_PATH = MODEL_SAVE_DIR_ISS / "iss_transformer_input_scaler.pkl"
ISS_TARGET_SCALER_PATH = MODEL_SAVE_DIR_ISS / "iss_transformer_target_scaler.pkl"

class RiskFeatureEncoder:
    def encode(self, risk, market_conditions):
        return np.array([
            risk,                          # 1: Rủi ro giao dịch
            market_conditions["volatility"],  # 2: Độ biến động
            market_conditions["atr"],         # 3: ATR
            market_conditions["rsi"],         # 4: RSI
            market_conditions["adx"],         # 5: ADX
            market_conditions["ema_diff"],    # 6: Chênh lệch EMA
            market_conditions["volume"],      # 7: Khối lượng giao dịch
            market_conditions["sentiment"]    # 8: Sentiment score
        ], dtype=np.float32)

# 1. Real-Time Data Feed Class
class RealTimeDataFeed:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol.lower().replace('/', '')
        self.timeframe = timeframe
        self.websocket_url = CONFIG["websocket_url"]
        self.data_queue = asyncio.Queue()
        self.running = False
        
    async def get_next_data(self):
        data = await self.data_queue.get()
        if 'timestamp' not in data:
            data['timestamp'] = int(time.time() * 1000)
        return data

    async def connect(self):
        self.running = True
        stream_name = f"{self.symbol}@kline_{self.timeframe}"
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.websocket_url) as ws:
                await ws.send_json({
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": 1
                })
                async for msg in ws:
                    if not self.running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if "k" in data:
                            kline = data["k"]
                            await self.data_queue.put({
                                "timestamp": kline["t"],
                                "open": float(kline["o"]),
                                "high": float(kline["h"]),
                                "low": float(kline["l"]),
                                "close": float(kline["c"]),
                                "volume": float(kline["v"])
                            })

    def stop(self):
        self.running = False

# 2. LOBTransformer for Order Book Analysis (Cải tiến Entry Point Optimization)
class LOBTransformer(nn.Module):
    def __init__(self, input_dim=20, d_model=64, nhead=4, num_layers=2, scaler_path: Optional[str] = None, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device
        self.encoder = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, d_model)

        # Load scaler nếu có
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                logging.info(f"LOBTransformer: Loaded input scaler from {scaler_path}")
                if not hasattr(self.scaler, 'n_features_in_') or self.scaler.n_features_in_ != self.input_dim:
                    logging.error(f"LOBTransformer: Loaded scaler dimension mismatch ({getattr(self.scaler, 'n_features_in_', 'N/A')} vs {self.input_dim}). Scaler disabled.")
                    self.scaler = None
            except Exception as e:
                logging.error(f"LOBTransformer: Failed to load scaler from {scaler_path}: {e}. Scaler disabled.")
        elif scaler_path:
            logging.warning(f"LOBTransformer: Scaler file not found at {scaler_path}. Scaler disabled.")
        else:
            logging.info("LOBTransformer: No scaler path provided. Scaler disabled.")

        self.to(self.device)

    def transform(self, order_book: dict) -> np.ndarray:
        default_output = np.zeros(self.d_model, dtype=np.float32)

        # Kiểm tra cấu trúc order book
        if not isinstance(order_book, dict) or \
           "bids" not in order_book or not isinstance(order_book["bids"], list) or \
           "asks" not in order_book or not isinstance(order_book["asks"], list):
            logging.warning("LOBTransformer: Invalid order book structure received.")
            return default_output

        levels_to_use = self.input_dim // 4
        try:
            bids_raw = [item for item in order_book.get("bids", []) if isinstance(item, (list, tuple)) and len(item) == 2][:levels_to_use]
            asks_raw = [item for item in order_book.get("asks", []) if isinstance(item, (list, tuple)) and len(item) == 2][:levels_to_use]
        except Exception as e:
            logging.error(f"LOBTransformer: Error accessing bids/asks: {e}")
            return default_output

        dummy_entry = [0.0, 0.0]
        bids = bids_raw + [dummy_entry] * (levels_to_use - len(bids_raw))
        asks = asks_raw + [dummy_entry] * (levels_to_use - len(asks_raw))

        try:
            bid_prices = [float(p) for p, s in bids]
            bid_volumes = [float(s) for p, s in bids]
            ask_prices = [float(p) for p, s in asks]
            ask_volumes = [float(s) for p, s in asks]
            features_unscaled = np.array(bid_prices + bid_volumes + ask_prices + ask_volumes, dtype=np.float32)

            if features_unscaled.shape[0] != self.input_dim:
                logging.error(f"LOBTransformer: Feature vector shape mismatch: {features_unscaled.shape}, expected ({self.input_dim},)")
                return default_output
        except (TypeError, ValueError) as e:
            logging.error(f"LOBTransformer: Could not convert order book data to float: {e}")
            return default_output
        except Exception as e:
            logging.error(f"LOBTransformer: Error preparing features: {e}")
            return default_output

        # Scaling nếu scaler có sẵn
        if self.scaler:
            try:
                if not np.isfinite(features_unscaled).all():
                    logging.warning("NaN/Inf detected in LOB features before scaling. Replacing with 0.")
                    features_unscaled = np.nan_to_num(features_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
                features_scaled = self.scaler.transform(features_unscaled.reshape(1, -1))
                if not np.isfinite(features_scaled).all():
                    logging.error("NaN/Inf detected after scaling LOB features. Using zeros.")
                    features_scaled = np.zeros((1, self.input_dim), dtype=np.float32)
            except Exception as scale_e:
                logging.error(f"LOBTransformer: Error applying scaler: {scale_e}. Using unscaled features.")
                features_scaled = features_unscaled.reshape(1, -1)
        else:
            features_scaled = features_unscaled.reshape(1, -1)

        # Inference
        try:
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            self.eval()
            with torch.no_grad():
                encoded = self.encoder(features_tensor)
                encoded_for_transformer = encoded.unsqueeze(1)
                transformed = self.transformer(encoded_for_transformer)
                decoded_input = transformed.squeeze(1)
                decoded_features = self.decoder(decoded_input)
                result = decoded_features.squeeze(0).cpu().numpy()
            if result.shape != (self.d_model,):
                logging.error(f"LOBTransformer: Final output shape mismatch: {result.shape}, expected ({self.d_model},)")
                return default_output
            return result
        except Exception as e:
            logging.error(f"LOBTransformer: Error during model inference: {e}", exc_info=True)
            return default_output

class MLPAction(nn.Module):
    def __init__(self, input_dim=MLP_INPUT_DIM,
                 hidden_dims=MLP_HIDDEN_DIMS_DEFAULT, # Sử dụng default từ Optuna
                 output_dim=MLP_OUTPUT_DIM,
                 dropout_p=MLP_DROPOUT_DEFAULT, # Sử dụng default từ Optuna
                 activation_fn=MLP_ACTIVATION_DEFAULT_CLS, # Thêm activation_fn, default từ Optuna
                 device=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.activation_fn = activation_fn # Lưu activation function class

        if device: self.device = torch.device(device)
        else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Log thông tin khởi tạo
        logging.info(f"{self.__class__.__name__} Initializing: input={input_dim}, output={output_dim}, "
                     f"hidden={hidden_dims}, dropout={dropout_p:.6f}, " # Log dropout chi tiết hơn
                     f"activation={activation_fn.__name__}, device={self.device}")

        self._build_network()
        self.is_trained = False
        self.to(self.device)

        # Tối ưu Inference với torch.compile
        try:
            if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
                 self.network = torch.compile(self.network, mode="reduce-overhead")
                 logging.info(f"{self.__class__.__name__}: Model compiled for inference.")
        except Exception as compile_e:
            logging.warning(f"{self.__class__.__name__}: Failed to compile model: {compile_e}.")

    def _build_network(self):
        """Xây dựng mạng nơ-ron dựa trên các tham số hiện tại của instance."""
        layers = []
        in_dim = self.input_dim
        for i, h_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(self.activation_fn()) # <<< Sử dụng activation_fn đã lưu >>>
            layers.append(nn.Dropout(p=self.dropout_p))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.network = nn.Sequential(*layers)
        logging.debug(f"{self.__class__.__name__}: Network built/rebuilt.") # Thêm log debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        # Kiểm tra dimension dựa trên self.input_dim của instance
        elif x.ndim != 2 or x.shape[1] != self.input_dim:
             raise ValueError(f"{self.__class__.__name__}: Invalid input shape/dim. Expected dim {self.input_dim}, got {x.shape[1]}")
        return self.network(x.to(self.device))

    def load_model(self, file_path="mlp_action_model.pth"):
        """Tải trọng số và cấu hình từ file checkpoint."""
        if not os.path.exists(file_path):
            logging.error(f"{self.__class__.__name__}: Model file not found: {file_path}.")
            self.is_trained = False; return False, None
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
                 logging.error(f"{self.__class__.__name__}: Invalid checkpoint format in {file_path}.")
                 self.is_trained = False; return False, None

            # --- Load thông tin cấu hình từ checkpoint ---
            # Sử dụng default từ Optuna nếu key không tồn tại trong checkpoint
            loaded_input_dim = checkpoint.get('input_dim', MLP_INPUT_DIM) # Nên lấy từ checkpoint
            loaded_hidden_dims = checkpoint.get('hidden_dims', MLP_HIDDEN_DIMS_DEFAULT)
            loaded_output_dim = checkpoint.get('output_dim', MLP_OUTPUT_DIM) # Nên lấy từ checkpoint
            loaded_dropout_p = checkpoint.get('dropout_p', MLP_DROPOUT_DEFAULT)
            # <<< Load tên activation, default là GELU >>>
            loaded_activation_name = checkpoint.get('activation_name', MLP_ACTIVATION_DEFAULT_NAME)
            loaded_fallback_value = checkpoint.get('fallback_prediction', 0.0)

            # --- Validate Dimensions quan trọng ---
            if loaded_input_dim != self.input_dim:
                 logging.error(f"{self.__class__.__name__}: Input dim mismatch! Expected {self.input_dim} (from init), loaded {loaded_input_dim}. Model cannot be used.")
                 self.is_trained = False; return False, None
            if loaded_output_dim != self.output_dim:
                 logging.error(f"{self.__class__.__name__}: Output dim mismatch! Expected {self.output_dim} (from init), loaded {loaded_output_dim}. Model cannot be used.")
                 self.is_trained = False; return False, None

            # --- Xác định Activation Class từ tên ---
            if hasattr(nn, loaded_activation_name):
                loaded_activation_cls = getattr(nn, loaded_activation_name)
            else:
                logging.warning(f"Loaded activation '{loaded_activation_name}' not found. Using default {MLP_ACTIVATION_DEFAULT_CLS.__name__}.")
                loaded_activation_cls = MLP_ACTIVATION_DEFAULT_CLS

            # --- Kiểm tra xem có cần Rebuild mạng không ---
            rebuild_needed = False
            if loaded_hidden_dims != self.hidden_dims:
                logging.info(f"{self.__class__.__name__}: Hidden dims changing from {self.hidden_dims} to {loaded_hidden_dims}.")
                self.hidden_dims = loaded_hidden_dims # Cập nhật instance variable
                rebuild_needed = True
            # So sánh dropout chính xác
            if abs(loaded_dropout_p - self.dropout_p) > 1e-9: # Dùng sai số nhỏ để so sánh float
                logging.info(f"{self.__class__.__name__}: Dropout changing from {self.dropout_p:.6f} to {loaded_dropout_p:.6f}.")
                self.dropout_p = loaded_dropout_p # Cập nhật instance variable
                rebuild_needed = True
            # So sánh activation class
            if self.activation_fn != loaded_activation_cls:
                 logging.info(f"{self.__class__.__name__}: Activation changing from {self.activation_fn.__name__} to {loaded_activation_cls.__name__}.")
                 self.activation_fn = loaded_activation_cls # Cập nhật instance variable
                 rebuild_needed = True

            # --- Rebuild nếu cần ---
            if rebuild_needed:
                 logging.info(f"Rebuilding network based on loaded configuration.")
                 self._build_network() # Gọi hàm build lại mạng
                 self.to(self.device) # Đảm bảo model trên đúng device sau khi build lại
                 try: # Re-compile
                      if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
                          self.network = torch.compile(self.network, mode="reduce-overhead")
                          logging.info(f"{self.__class__.__name__}: Re-compiled model.")
                 except Exception as compile_e:
                     logging.warning(f"{self.__class__.__name__}: Re-compile failed after rebuild: {compile_e}")

            # --- Load trọng số ---
            self.load_state_dict(checkpoint['state_dict'])
            self.is_trained = True
            self.eval() # Chuyển sang chế độ đánh giá sau khi load thành công
            logging.info(f"{self.__class__.__name__} model loaded successfully from {file_path} (Fallback: {loaded_fallback_value:.4f})")
            return True, loaded_fallback_value # Trả về trạng thái load và fallback

        except Exception as e:
            logging.error(f"Error loading {self.__class__.__name__} model from {file_path}: {e}", exc_info=True)
            self.is_trained = False; return False, None

    @torch.no_grad()
    def predict(self, x: np.ndarray, fallback_value: float = 0.0) -> np.ndarray:
        """Dự đoán giá trị từ input features NumPy, sử dụng fallback."""
        self.eval() # Đảm bảo đang ở chế độ đánh giá
        default_output_shape = (x.shape[0], self.output_dim) if x.ndim == 2 else (self.output_dim,)

        if not self.is_trained:
            # Giảm logging ở đây để tránh spam nếu model load lỗi
            # logging.warning(f"{self.__class__.__name__}: Predict on untrained model. Returning fallback: {fallback_value}")
            return np.full(default_output_shape, fallback_value, dtype=np.float32)

        try:
            # Chuyển đổi sang tensor
            # Di chuyển đến device được thực hiện trong self.forward()
            x_tensor = torch.tensor(x, dtype=torch.float32)

            # Thực hiện dự đoán (gọi self.forward)
            predictions_tensor = self(x_tensor)

            # Chuyển kết quả về numpy array trên CPU
            result = predictions_tensor.cpu().numpy()

            # Kiểm tra NaN/Inf trong kết quả predict
            if not np.all(np.isfinite(result)):
                 logging.warning(f"{self.__class__.__name__}: Prediction resulted in NaN/Inf. Returning fallback: {fallback_value}")
                 return np.full(default_output_shape, fallback_value, dtype=np.float32)

            # Kiểm tra dimension output cuối cùng (an toàn hơn)
            if result.shape[-1] != self.output_dim:
                logging.error(f"{self.__class__.__name__}: Output dimension mismatch after prediction! Expected {self.output_dim}, got {result.shape[-1]}. Returning fallback.")
                return np.full(default_output_shape, fallback_value, dtype=np.float32)

            return result # Trả về kết quả dự đoán hợp lệ
        except ValueError as ve: # Bắt lỗi shape từ self.forward
            logging.error(f"Error during {self.__class__.__name__} prediction (likely shape): {ve}")
            return np.full(default_output_shape, fallback_value, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} prediction: {e}", exc_info=True)
            logging.warning(f"{self.__class__.__name__}: Prediction failed. Returning fallback value: {fallback_value}")
            return np.full(default_output_shape, fallback_value, dtype=np.float32)

# 4. Market Environment with Real-Time Data (Cập nhật với LOBTransformer và MLPAction)
class MarketEnvironment(Env):
    def __init__(self, bot, data: Dict[str, pd.DataFrame], order_book_data: List[dict], initial_symbol: str):
        super().__init__() # Gọi __init__ của lớp cha Env
        self.bot = bot
        self.data = data
        self.order_book_data = order_book_data
        self.current_symbol = initial_symbol
        self.current_step = 0
        if "15m" in self.data and not self.data["15m"].empty:
             try:
                  self.current_step = bot.get_adaptive_warmup_period(data["15m"])
                  self.max_steps = len(data["15m"]) - 20 
                  self.max_steps = max(0, self.max_steps)
                  self.current_step = min(self.current_step, self.max_steps)
             except Exception as e:
                  logging.error(f"Error getting warmup period: {e}. Setting current_step to default.")
                  # Đặt giá trị mặc định hoặc tối thiểu nếu có lỗi
                  self.current_step = CONFIG.get("hmm_warmup_min", 50)
                  self.max_steps = len(data.get("15m", pd.DataFrame())) - 20
                  self.max_steps = max(0, self.max_steps)
                  self.current_step = min(self.current_step, self.max_steps)

        self.max_steps = len(self.data.get("15m", pd.DataFrame())) - 20 # Xử lý nếu '15m' không có
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32) # Thêm dtype
        dim_base_features = 4          # close, RSI, ATR, volatility
        dim_ob_extra = 2               # bid_ask_spread, liquidity
        dim_lob_transformer = CONFIG.get("lob_d_model", 64) # Lấy từ config
        dim_mlp_action = CONFIG.get("mlp_action_output_dim", 1)          # Output của PriceActionTransformer
        dim_additional = 2             # momentum, order_imbalance
        dim_sentiment = 1  

        # Tính tổng dimension
        self.observation_space_dim = (
            dim_base_features
            + dim_ob_extra
            + dim_lob_transformer # Sử dụng toàn bộ 64 (hoặc giá trị config)
            + dim_mlp_action 
            + dim_additional
            + dim_sentiment 
        )
        logging.info(f"Calculated observation_space_dim: {self.observation_space_dim}")
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32)
        self.data_feed = None # Khởi tạo data_feed là None

        # Khởi tạo các model con và chuyển đến device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(bot, 'mlp_action_analyzer') and bot.mlp_action_analyzer:
            self.mlp_action_analyzer = bot.mlp_action_analyzer
            self.mlp_action_analyzer.to(self.device).eval() # Ensure on correct device and in eval mode
            if not self.mlp_action_analyzer.is_trained:
                 logging.warning("MarketEnvironment: MLPAction model provided is not trained.")
        else:
            # This is critical - the environment cannot function without the MLP model instance
            logging.error("MarketEnvironment CRITICAL ERROR: mlp_action_analyzer not found in bot instance!")
            raise ValueError("MLP Action Analyzer instance not found in bot.")

        if hasattr(bot, 'mlp_action_scaler') and bot.mlp_action_scaler:
            self.mlp_action_scaler = bot.mlp_action_scaler
        else:
            # Scaler is also critical if the model was trained with one
            logging.error("MarketEnvironment CRITICAL ERROR: mlp_action_scaler not found in bot instance!")
            raise ValueError("MLP Action Scaler instance not found in bot.")

        # Get fallback value from bot
        self.mlp_action_fallback = getattr(bot, 'mlp_action_fallback', 0.0)
        logging.info(f"MarketEnvironment: Linked MLP Analyzer, Scaler, and Fallback ({self.mlp_action_fallback:.4f}) from bot.")

        # Đặt trong try-except để xử lý lỗi nếu khởi tạo model thất bại
        try:
            lob_scaler_path = "trained_models_lob_final/lob_input_scaler_optuna.pkl"
            lob_model_path = "trained_models_lob_final/lob_transformer_backbone_optuna_final.pth"
            self.order_book_analyzer = LOBTransformer(
                input_dim=CONFIG["lob_input_dim"],
                d_model=CONFIG["lob_d_model"],
                nhead=CONFIG["lob_nhead"],
                num_layers=CONFIG["lob_num_layers"],
                scaler_path=lob_scaler_path,
                device=self.device
            )
            if os.path.exists(lob_model_path):
                try:
                    state_dict = torch.load(lob_model_path, map_location=self.device)
                    self.order_book_analyzer.load_state_dict(state_dict)
                    logging.info(f"MarketEnvironment: Successfully loaded trained LOBTransformer weights.")
                except Exception as load_e:
                    logging.error(f"MarketEnvironment: Failed to load LOBTransformer weights from {lob_model_path}: {load_e}. Using random weights.")
            else:
                logging.warning(f"MarketEnvironment: Trained LOB weights not found at {lob_model_path}. Using random weights.")
            self.order_book_analyzer.eval() # Đặt chế độ eval
        except Exception as e:
             logging.error(f"Failed to initialize LOB/PriceAction Analyzers: {e}", exc_info=True)
             raise # Raise lỗi để biết môi trường khởi tạo không thành công

        # Khởi tạo Scaler
        self.scaler = None # Sẽ được fit trong _initialize_scaler
        self._initialize_scaler() # Gọi hàm fit scaler

    def _initialize_scaler(self):
        logging.info("Initializing and fitting the observation scaler...")
        try:
            df_15m = self.data.get("15m")
            if df_15m is None or df_15m.empty:
                 raise ValueError("Cannot fit scaler: '15m' data is missing or empty.")

            warmup = self.bot.get_adaptive_warmup_period(df_15m)
            fit_steps = min(len(df_15m) - warmup - 1, 5000)

            if fit_steps < 100:
                 logging.warning(f"Insufficient data ({fit_steps} steps) to fit scaler properly. Using dummy scaler.")
                 self.scaler = StandardScaler()
                 self.scaler.fit(np.zeros((2, self.observation_space_dim))) # Fit với dữ liệu giả
                 return

            sample_observations = []
            original_step = self.current_step
            start_fit_step = warmup
            end_fit_step = warmup + fit_steps

            logging.info(f"Fitting scaler on steps {start_fit_step} to {end_fit_step-1}...")

            for i in range(start_fit_step, end_fit_step):
                self.current_step = i # Tạm thời đặt step để lấy obs
                try:
                    # Kiểm tra index trước khi gọi iloc
                    if i < len(df_15m):
                         obs_unscaled = self._get_observation_unscaled()
                         # Kiểm tra NaN/Inf trong obs_unscaled trước khi thêm
                         if np.isfinite(obs_unscaled).all():
                              sample_observations.append(obs_unscaled)
                         else:
                              logging.warning(f"NaN/Inf detected in unscaled observation at step {i}. Skipping for scaler fitting.")
                    else:
                        logging.warning(f"Index {i} out of bounds during scaler fitting (len={len(df_15m)}).")
                        break # Dừng nếu vượt quá index

                except Exception as e:
                    logging.warning(f"Error getting unscaled observation at step {i} during scaler fitting: {e}. Skipping.")
                    continue

            self.current_step = original_step # Khôi phục step

            if not sample_observations:
                logging.error("Failed to collect any valid sample observations for scaler fitting. Using dummy scaler.")
                self.scaler = StandardScaler()
                self.scaler.fit(np.zeros((2, self.observation_space_dim)))
                return

            sample_observations_np = np.array(sample_observations)

            if sample_observations_np.ndim != 2 or sample_observations_np.shape[1] != self.observation_space_dim:
                 logging.error(f"Sample observations have incorrect shape {sample_observations_np.shape} for scaler fitting. Using dummy scaler.")
                 self.scaler = StandardScaler()
                 self.scaler.fit(np.zeros((2, self.observation_space_dim)))
                 return

            self.scaler = StandardScaler()
            self.scaler.fit(sample_observations_np)
            logging.info(f"Observation scaler fitted successfully on {len(sample_observations)} valid samples.")
            # joblib.dump(self.scaler, "observation_scaler.pkl") # Lưu scaler nếu cần

        except Exception as e:
            logging.error(f"Error during scaler initialization: {e}", exc_info=True)
            logging.warning("Using dummy StandardScaler due to initialization error.")
            self.scaler = StandardScaler()
            self.scaler.fit(np.zeros((2, self.observation_space_dim)))

    async def setup_real_time(self, symbol: str):
        """Thiết lập feed dữ liệu thời gian thực."""
        self.data_feed = RealTimeDataFeed(symbol, "15m")
        asyncio.create_task(self.data_feed.connect())
        logging.info(f"Real-time data feed task created for {symbol} 15m")

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        logging.debug(f"Resetting MarketEnvironment for symbol: {self.current_symbol}")
        # Reset step về đầu giai đoạn (sau warmup) cho symbol hiện tại
        if self.current_symbol in self.data and "15m" in self.data[self.current_symbol]:
             df_15m_current = self.data[self.current_symbol]["15m"]
             try:
                  self.current_step = self.bot.get_adaptive_warmup_period(df_15m_current)
                  self.max_steps = len(df_15m_current) - 20 # Cập nhật max_steps cho symbol này
                  self.max_steps = max(0, self.max_steps)
                  self.current_step = min(self.current_step, self.max_steps)
             except Exception as e:
                  logging.error(f"Error getting warmup period during reset for {self.current_symbol}: {e}. Using default.")
                  self.current_step = CONFIG.get("hmm_warmup_min", 50)
                  self.max_steps = len(df_15m_current) - 20
                  self.max_steps = max(0, self.max_steps)
                  self.current_step = min(self.current_step, self.max_steps)
        else:
             self.current_step = 0 # Fallback
             self.max_steps = 0

        obs = self._get_observation()
        info = {'current_symbol': self.current_symbol} # Thêm thông tin symbol vào info
        logging.debug(f"Environment reset complete. Current step: {self.current_step}")
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Khởi tạo các giá trị mặc định
        pnl = 0.0
        reward = 0.0
        info = {'trade_executed': False, 'reason': ''} # Thêm 'reason' vào info
        terminated = False # Sẽ được cập nhật
        truncated = False  # Mặc định là False, có thể thay đổi nếu có giới hạn thời gian riêng

        # Diễn giải action
        action_value = 0.0
        if isinstance(action, np.ndarray) and action.size == 1:
            action_value = action.item()
        elif isinstance(action, (int, float)):
            action_value = action
        logging.debug(f"MarketEnv Step {self.current_step}: Raw action_value = {action_value:.4f}")

        # --- Bắt đầu khối try-except lớn để đảm bảo an toàn ---
        try:
            # --- 1. Kiểm tra điều kiện dữ liệu và bước hiện tại ---
            if self.current_step >= len(self.data["15m"]):
                logging.warning(f"MarketEnvironment.step: current_step {self.current_step} out of bounds for '15m' data (len={len(self.data['15m'])}). Ending episode.")
                obs = self._get_observation()  # Cố gắng lấy obs cuối cùng
                # Đảm bảo shape obs an toàn
                if obs.shape != self.observation_space.shape: # Sử dụng self.observation_space.shape
                    obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                info['reason'] = 'out_of_data'
                return obs, 0.0, True, False, info # terminated=True, truncated=False

            last_row = self.data["15m"].iloc[self.current_step]
            current_timestamp = pd.to_datetime(last_row.name)
            symbol = CONFIG["symbols"][0] # Giả sử chỉ có 1 symbol chính cho môi trường này

            active_sentiment_score = 0.0
            if hasattr(self.bot, 'sentiment_analyzer') and self.bot.sentiment_analyzer:
                try:
                    sentiment_details = self.bot.sentiment_analyzer.get_detailed_active_sentiment(current_timestamp)
                    active_sentiment_score = sentiment_details.get("total_score_adj_confidence", 0.0)
                except Exception as sent_e:
                    logging.error(f"Error getting sentiment in MarketEnvironment.step: {sent_e}")

            # --- 2. Tạo Signal từ Action ---
            direction = None
            if action_value > 0.1: direction = "LONG"
            elif action_value < -0.1: direction = "SHORT"

            signal = None
            if direction:
                signal_data_points = {
                    "entry_price": last_row.get("close"),
                    "atr": last_row.get("ATR"),
                    "adx": last_row.get("ADX"),
                }
                if None in signal_data_points.values() or pd.isna(list(signal_data_points.values())).any():
                    logging.warning(f"Missing required data (close/ATR/ADX) in last_row at step {self.current_step}. Cannot create valid signal.")
                else:
                    signal = {
                        "direction": direction,
                        "entry_price": signal_data_points["entry_price"],
                        "atr": signal_data_points["atr"],
                        "adx": signal_data_points["adx"],
                        "timestamp": last_row.name, # Giữ pd.Timestamp
                        "win_prob": 0.5,  # Placeholder, nên được tính toán hoặc lấy từ model
                        "symbol": symbol
                    }
                    logging.debug(f"MarketEnvironment.step ({symbol}): Action {action_value:.2f} interpreted as {direction}")
            else:
                logging.debug(f"MarketEnv Step {self.current_step}: Action {action_value:.2f} resulted in NO direction. Skipping trade simulation.")

            # --- 3. Mô phỏng Giao dịch nếu có Tín hiệu Hợp Lệ ---
            if signal:
                try:
                    # 3a. Tính TP/SL
                    tp_levels, sl_price = self.bot.dynamic_stop_management( # Truyền signal dictionary
                        signal # Truyền toàn bộ dict signal
                        # entry_price=signal["entry_price"], atr_value=signal["atr"], adx_value=signal["adx"],
                        # direction=signal["direction"], symbol=signal["symbol"]
                    )

                    # 3b. Tính Size
                    strategy_params = {
                        "symbol": signal["symbol"],
                        "entry_price": signal["entry_price"],
                        "atr": signal["atr"],
                        "risk": CONFIG["risk_per_trade"],
                        "volatility": last_row.get("volatility", 0.0),
                        "rsi": last_row.get("RSI", 50.0),
                        "adx": signal["adx"],
                        "ema_diff": last_row.get("EMA_50", last_row.get("close", 0)) - last_row.get("EMA_200", last_row.get("close", 0)),
                        "volume": last_row.get("volume", 0.0),
                        "sentiment": active_sentiment_score,
                        "order_book": self.order_book_data[min(self.current_step, len(self.order_book_data) - 1)] if self.order_book_data else {"bids": [], "asks": []},
                        "direction": signal["direction"] # Thêm direction vào đây
                    }
                    # Gọi qua self.bot.position_sizer
                    position_size = self.bot.position_sizer.calculate_size(
                        strategy_params,
                        stop_loss_price=sl_price, # Truyền giá SL
                        # risk_factor đã được bao gồm trong strategy_params["risk"]
                    )

                    # 3c. Kiểm tra Exposure
                    if position_size > 0:
                        # Hàm check_position_exposure đã được gọi bên trong calculate_size của sizer mới.
                        # Nếu không, bạn cần gọi lại ở đây:
                        # if not self.bot.check_position_exposure(position_size, signal["entry_price"], signal["symbol"]):
                        #     logging.warning(f"MarketEnvironment.step ({signal['symbol']}): Trade skipped due to exposure limit (Size: {position_size}).")
                        #     signal = None # Hủy signal
                        # else:
                        #     pass # Hợp lệ
                        # Tuy nhiên, logic hiện tại của calculate_size chưa thấy gọi check_position_exposure
                        # Tạm thời giả định sizer không check, ta check ở đây:
                        if not self.bot.check_position_exposure(position_size, signal["entry_price"], signal["symbol"]):
                             logging.warning(f"MarketEnvironment.step ({signal['symbol']}): Trade skipped due to exposure limit after size calculation (Size: {position_size}).")
                             signal = None
                    elif position_size <= 0:
                        logging.warning(f"MarketEnvironment.step ({signal['symbol']}): Calculated position size is zero or negative ({position_size}). Skipping trade.")
                        signal = None

                    # 3d. Mô phỏng Trade nếu Signal vẫn hợp lệ và Size > 0
                    if signal and position_size > 0:
                        # Sử dụng _simulate_trade_sync (phiên bản đồng bộ)
                        exit_price, pnl = self.bot._simulate_trade_sync(
                            {"15m": self.data["15m"]}, # Truyền data đúng format
                            self.current_step,
                            signal,
                            tp_levels,
                            sl_price,
                            position_size
                        )

                        # Lưu lịch sử (nếu cần thiết cho môi trường)
                        self.bot.trade_history.append({
                            "symbol": signal["symbol"], "position_size": position_size,
                            "entry_price": signal["entry_price"], "exit_price": exit_price,
                            "pnl": pnl, "action_value": action_value, # Lưu action_value gốc
                            "timestamp": signal["timestamp"]
                        })
                        info['trade_executed'] = True
                        info['pnl'] = pnl
                        info['exit_price'] = exit_price
                    else:
                        pnl = 0.0 # Không có trade -> PnL = 0
                except Exception as sim_err:
                    logging.error(f"MarketEnvironment.step: Error during SL/Size/Simulation for {signal.get('symbol', 'N/A')}: {sim_err}", exc_info=True)
                    pnl = 0.0
                    info['trade_error'] = str(sim_err)
            else: # signal is None
                pnl = 0.0

            # --- 4. Tính toán Reward ---
            try:
                reward = self._calculate_reward(pnl, action) # Truyền action gốc
                info['reward'] = reward
            except Exception as reward_err:
                logging.error(f"Error calculating reward in MarketEnvironment.step: {reward_err}", exc_info=True)
                reward = 0.0
                info['reward_error'] = str(reward_err)

            # --- 5. Cập nhật trạng thái môi trường ---
            self.current_step += 1
            # `terminated` là khi episode kết thúc tự nhiên (hết data, mục tiêu đạt được, agent "chết")
            terminated = self.current_step >= self.max_steps
            if terminated:
                info['reason'] = 'max_steps_reached'

            # `truncated` là khi episode kết thúc do giới hạn thời gian bên ngoài (ít dùng ở đây)
            truncated = False # Giả sử không có time limit riêng

            # --- 6. Lấy Observation cho bước tiếp theo ---
            obs = self._get_observation() # Hàm này đã có xử lý scaler và shape
            # Đảm bảo obs có shape đúng sau khi lấy
            if obs.shape != self.observation_space.shape:
                logging.error(f"MarketEnvironment.step {self.current_step-1}: Observation shape mismatch after _get_observation: {obs.shape}. Fixing to zeros.")
                obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

            logging.debug(f"MarketEnvironment.step {self.current_step-1}: Returning state, reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
            return obs, reward, terminated, truncated, info

        # --- Xử lý lỗi nghiêm trọng trong toàn bộ step ---
        except Exception as e:
            logging.critical(f"MarketEnvironment.step: Unhandled CRITICAL error at step {self.current_step}: {e}", exc_info=True)
            # Trạng thái an toàn khi có lỗi không mong muốn
            obs = self._get_observation() # Cố gắng lấy obs hiện tại
            if obs.shape != self.observation_space.shape:
                 obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            reward = 0.0
            terminated = True # Kết thúc episode nếu có lỗi nghiêm trọng
            truncated = False
            info = {"critical_error": str(e), "reason": "critical_step_error"}
            return obs, reward, terminated, truncated, info
        
    async def _step_async(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        logging.debug(f"Entering _step_async with action: {action}")
        # Khởi tạo các giá trị mặc định
        pnl = 0.0
        reward = 0.0
        info = {'trade_executed': False, 'reason': ''}
        terminated = False
        truncated = False

        action_value = 0.0
        if isinstance(action, np.ndarray) and action.size == 1:
            action_value = action.item()
        elif isinstance(action, (int, float)):
            action_value = action

        # Biến cục bộ để lưu trữ dữ liệu cần thiết từ bước 1 và 2
        last_row = None
        current_timestamp = None
        symbol_local = None # Đổi tên để tránh trùng với symbol trong signal
        active_sentiment_score_local = 0.0 # Đổi tên
        signal = None # Khởi tạo signal là None

        # --- Bắt đầu khối try-except lớn để đảm bảo an toàn ---
        try:
            # --- 1. Kiểm tra điều kiện dữ liệu và bước hiện tại ---
            if self.current_step >= len(self.data["15m"]):
                logging.warning(f"_step_async: current_step {self.current_step} out of bounds for '15m' data (len={len(self.data['15m'])}). Ending episode.")
                obs = self._get_observation()
                if obs.shape != self.observation_space.shape:
                    obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                info['reason'] = 'out_of_data'
                return obs, 0.0, True, False, info

            # --- Lấy dữ liệu dòng cuối và thông tin cơ bản (có thể gây IndexError) ---
            last_row = self.data["15m"].iloc[self.current_step]
            current_timestamp = pd.to_datetime(last_row.name)
            symbol_local = CONFIG["symbols"][0] # Lấy symbol từ config

            if hasattr(self.bot, 'sentiment_analyzer') and self.bot.sentiment_analyzer:
                try:
                    sentiment_details = self.bot.sentiment_analyzer.get_detailed_active_sentiment(current_timestamp)
                    active_sentiment_score_local = sentiment_details.get("total_score_adj_confidence", 0.0)
                except Exception as sent_e:
                    logging.error(f"Error getting sentiment in _step_async: {sent_e}")
                    # active_sentiment_score_local vẫn là 0.0

            # --- 2. Tạo Signal từ Action (có thể có lỗi khác ở đây) ---
            direction = None
            if action_value > 0.1: direction = "LONG"
            elif action_value < -0.1: direction = "SHORT"

            if direction:
                # Các giá trị này phụ thuộc vào last_row đã lấy ở trên
                signal_data_points = {
                    "entry_price": last_row.get("close"),
                    "atr": last_row.get("ATR"),
                    "adx": last_row.get("ADX"),
                }
                if any(val is None or pd.isna(val) for val in signal_data_points.values()):
                    logging.warning(f"Missing or NaN required data (close/ATR/ADX) in last_row at step {self.current_step} (async). Signal not created.")
                    # signal sẽ vẫn là None
                else:
                    signal = {
                        "direction": direction,
                        "entry_price": signal_data_points["entry_price"],
                        "atr": signal_data_points["atr"],
                        "adx": signal_data_points["adx"],
                        "timestamp": last_row.name, # last_row.name là pd.Timestamp
                        "win_prob": 0.5,
                        "symbol": symbol_local # Sử dụng symbol_local
                    }
                    logging.debug(f"_step_async ({symbol_local}): Action {action_value:.2f} interpreted as {direction}")


            # --- 3. Mô phỏng Giao dịch nếu có Tín hiệu HỢP LỆ ---
            # `signal` có thể vẫn là None nếu không có direction hoặc thiếu dữ liệu ở bước 2
            if signal:
                try:
                    # 3a. Tính TP/SL
                    tp_levels, sl_price = self.bot.dynamic_stop_management(
                        signal["entry_price"], signal["atr"], signal["adx"],
                        signal["direction"], signal["symbol"] # Truyền các giá trị từ `signal` dict
                    )

                    # 3b. Tính Size
                    # `last_row` và `active_sentiment_score_local` đã được lấy ở các bước trước
                    strategy_params = {
                        "symbol": signal["symbol"], "entry_price": signal["entry_price"],
                        "atr": signal["atr"], "risk": CONFIG["risk_per_trade"],
                        "volatility": last_row.get("volatility", 0.0), # Sử dụng last_row
                        "rsi": last_row.get("RSI", 50.0), "adx": signal["adx"],
                        "ema_diff": last_row.get("EMA_50", last_row.get("close", 0)) - last_row.get("EMA_200", last_row.get("close", 0)),
                        "volume": last_row.get("volume", 0.0),  "sentiment": active_sentiment_score_local, # Sử dụng biến cục bộ
                        "order_book": await self.bot.fetch_order_book(signal["symbol"])
                    }
                    if strategy_params["order_book"] is None:
                        logging.warning(f"_step_async ({signal['symbol']}): Failed to fetch order book. Using empty OB for sizing.")
                        strategy_params["order_book"] = {"bids": [], "asks": []}

                    position_size = self.bot.position_sizer.calculate_size(
                        strategy_params,
                        stop_loss_price=sl_price
                    )

                    # 3c. Kiểm tra Exposure
                    if position_size > 0:
                        if not self.bot.check_position_exposure(position_size, signal["entry_price"], signal["symbol"]):
                            logging.warning(f"_step_async ({signal['symbol']}): Trade skipped due to exposure limit (Size: {position_size}).")
                            # Không hủy signal ở đây nữa, để logic `if signal and position_size > 0` ở dưới quyết định
                            position_size = 0 # Đặt size về 0 để không mô phỏng
                    elif position_size <= 0:
                        logging.warning(f"_step_async ({signal['symbol']}): Calculated position size is zero or negative ({position_size}).")
                        position_size = 0 # Đảm bảo size là 0

                    # 3d. Mô phỏng Trade nếu Signal vẫn hợp lệ và Size > 0
                    # `signal` vẫn là dict ban đầu, chỉ `position_size` có thể đã thay đổi
                    if signal and position_size > 0:
                        exit_price, pnl = await self.bot._simulate_trade(
                            {"15m": self.data["15m"]}, self.current_step, signal, tp_levels, sl_price, position_size
                        )
                        self.bot.trade_history.append({
                            "symbol": signal["symbol"], "position_size": position_size,
                            "entry_price": signal["entry_price"], "exit_price": exit_price,
                            "pnl": pnl, "action_sac": action_value,
                            "timestamp": signal["timestamp"]
                        })
                        info['trade_executed'] = True
                        info['pnl'] = pnl
                        info['exit_price'] = exit_price
                    else: # signal hợp lệ nhưng size <= 0 hoặc bị exposure chặn
                        pnl = 0.0
                except Exception as sim_err_async:
                    logging.error(f"MarketEnvironment._step_async: Error during SL/Size/Simulation for {signal.get('symbol', 'N/A')}: {sim_err_async}", exc_info=True)
                    pnl = 0.0
                    info['trade_error'] = str(sim_err_async)
            else: # signal is None (từ bước 2)
                pnl = 0.0

            # --- 4. Tính toán Reward ---
            try:
                reward = self._calculate_reward(pnl, action)
                info['reward_async'] = reward
            except Exception as reward_err_async:
                logging.error(f"Error calculating reward in _step_async: {reward_err_async}", exc_info=True)
                reward = 0.0
                info['reward_error_async'] = str(reward_err_async)

            # --- 5. Cập nhật trạng thái môi trường ---
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            if terminated:
                info['reason'] = 'max_steps_reached'
            truncated = False

            # --- 6. Lấy Observation cho bước tiếp theo ---
            obs = self._get_observation()
            if obs.shape != self.observation_space.shape:
                logging.error(f"_step_async {self.current_step-1}: Observation shape mismatch after _get_observation: {obs.shape}. Fixing to zeros.")
                obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

            logging.debug(f"Exiting _step_async (Step: {self.current_step-1}). Reward={reward:.4f}, Terminated={terminated}, Truncated={truncated}")
            return obs, reward, terminated, truncated, info

        # --- Xử lý lỗi nghiêm trọng trong toàn bộ _step_async ---
        except IndexError as e_idx:
            logging.error(f"MarketEnvironment._step_async: IndexError accessing data at step {self.current_step}: {e_idx}", exc_info=True)
            obs = self._get_observation() # Cố gắng lấy obs
            if obs.shape != self.observation_space.shape:
                obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            info['reason'] = 'data_access_error_outer_index'
            # Đặt reward về 0 và terminated=True
            return obs, 0.0, True, False, info

        except Exception as e_outer: # Bắt tất cả các lỗi khác không được xử lý cụ thể
            logging.critical(f"MarketEnvironment._step_async: Unhandled CRITICAL error at step {self.current_step}: {e_outer}", exc_info=True)
            obs = self._get_observation() # Cố gắng lấy obs
            if obs.shape != self.observation_space.shape:
                obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            # Đặt reward về 0 và terminated=True
            return obs, 0.0, True, False, {"critical_error": str(e_outer), "reason": "critical_async_step_error_outer"}

    # --- HÀM _get_observation ĐÃ SỬA ĐỂ DÙNG SCALER ---
    def _get_observation(self) -> np.ndarray:
        obs_unscaled = self._get_observation_unscaled()

        if self.scaler is None:
            logging.warning("MarketEnvironment: Scaler is not initialized. Returning unscaled observation.")
            if obs_unscaled.shape != (self.observation_space_dim,):
                obs_unscaled = np.zeros(self.observation_space_dim, dtype=np.float32)
            # <<< THÊM XỬ LÝ NAN/INF CHO UNCALED TRƯỚC KHI TRẢ VỀ >>>
            if not np.isfinite(obs_unscaled).all():
                logging.warning(f"MarketEnvironment: NaN/Inf in unscaled obs (scaler not ready) at step {self.current_step}. Replacing with 0.")
                obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
            return obs_unscaled.astype(np.float32)

        try:
            if obs_unscaled.shape != (self.observation_space_dim,):
                logging.error(f"MarketEnvironment _get_observation: Unscaled obs shape mismatch {obs_unscaled.shape}. Returning zeros.")
                return np.zeros(self.observation_space_dim, dtype=np.float32)

            # <<< THÊM XỬ LÝ NAN/INF TRƯỚC KHI SCALE >>>
            if not np.isfinite(obs_unscaled).all():
                logging.warning(f"MarketEnvironment: NaN/Inf found in unscaled observation before scaling at step {self.current_step}. Replacing with 0.")
                obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0)

            obs_reshaped = obs_unscaled.reshape(1, -1)
            obs_scaled = self.scaler.transform(obs_reshaped)

            # <<< THÊM XỬ LÝ NAN/INF SAU KHI SCALE >>>
            if not np.isfinite(obs_scaled).all():
                logging.error(f"MarketEnvironment _get_observation: NaN/Inf detected after scaling at step {self.current_step}. Replacing with zeros.")
                obs_scaled = np.nan_to_num(obs_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            return obs_scaled.flatten().astype(np.float32)
        except Exception as e:
            logging.error(f"MarketEnvironment: Error scaling observation at step {self.current_step}: {e}", exc_info=True)
            # <<< XỬ LÝ NAN/INF CHO FALLBACK UNSCALED >>>
            if obs_unscaled.shape != (self.observation_space_dim,):
                obs_unscaled = np.zeros(self.observation_space_dim, dtype=np.float32)
            if not np.isfinite(obs_unscaled).all():
                obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
            return obs_unscaled.astype(np.float32)


    # --- HÀM _get_observation_unscaled ---
    def _get_observation_unscaled(self) -> np.ndarray:
        """Lấy observation chưa chuẩn hóa."""
        # Đảm bảo current_step hợp lệ
        symbol = self.current_symbol
        current_step_safe = min(self.current_step, len(self.data[symbol]["15m"]) - 1)
        if current_step_safe < 0 : current_step_safe = 0
        if symbol not in self.data or "15m" not in self.data[symbol] or self.data[symbol]["15m"].empty or current_step_safe >= len(self.data[symbol]["15m"]):
            logging.error(f"MarketEnv ({symbol}) - Cannot get observation at step {current_step_safe}.")
            return np.zeros(self.observation_space_dim, dtype=np.float32)

        last_row = self.data[symbol]["15m"].iloc[current_step_safe]
        timestamp_for_obs = pd.to_datetime(last_row.name)
        # --- Lấy features cơ bản và từ order book ---
        # Danh sách các cột thực sự cần thiết cho observation
        required_base_cols = ["close", "RSI", "ATR", "volatility", "momentum", "order_imbalance"]
        obs_values = {}
        all_cols_present = True
        for col in required_base_cols:
            if col not in last_row or pd.isna(last_row[col]):
                logging.warning(f"Missing or NaN value for required column '{col}' at step {current_step_safe}. Using 0.0.")
                obs_values[col] = 0.0
                all_cols_present = False # Đánh dấu là thiếu dữ liệu
            else:
                obs_values[col] = last_row[col]

        # Lấy order book
        order_book_idx_safe = min(current_step_safe, len(self.order_book_data) - 1) if self.order_book_data else -1
        order_book = self.order_book_data[order_book_idx_safe] if order_book_idx_safe >= 0 else {"bids": [], "asks": []}

        # Tính spread và liquidity
        bid_ask_spread = 0.0
        liquidity = 0.0
        try:
            if order_book.get("asks") and order_book.get("bids") and \
               len(order_book["asks"]) > 0 and isinstance(order_book["asks"][0], (list, tuple)) and len(order_book["asks"][0]) > 0 and \
               len(order_book["bids"]) > 0 and isinstance(order_book["bids"][0], (list, tuple)) and len(order_book["bids"][0]) > 0 and \
               float(order_book["bids"][0][0]) > 0:
                bid_ask_spread = (float(order_book["asks"][0][0]) - float(order_book["bids"][0][0])) / float(order_book["bids"][0][0])

            bids_vol = sum(float(bid[1]) for bid in order_book.get("bids", [])[:5] if isinstance(bid, (list, tuple)) and len(bid) == 2)
            asks_vol = sum(float(ask[1]) for ask in order_book.get("asks", [])[:5] if isinstance(ask, (list, tuple)) and len(ask) == 2)
            liquidity = bids_vol + asks_vol
        except (TypeError, ValueError, IndexError, ZeroDivisionError) as e:
             logging.warning(f"Error calculating spread/liquidity at step {current_step_safe}: {e}. Using 0.0.")
             bid_ask_spread = 0.0
             liquidity = 0.0
        obs_values["bid_ask_spread"] = bid_ask_spread
        obs_values["liquidity"] = liquidity


        # --- Lấy features từ LOBTransformer ---
        lob_features = np.zeros(CONFIG.get("lob_d_model", 64), dtype=np.float32) # Sử dụng config
        try:
            order_book_current = self.data.get(symbol, {}).get("order_book", {"bids": [], "asks": []})
            if order_book_current and hasattr(self, 'order_book_analyzer') and self.order_book_analyzer:
                 lob_features = self.order_book_analyzer.transform(order_book_current)
        except Exception as e:
            logging.error(f"Error getting LOB features via transform at step {current_step_safe} for {symbol}: {e}", exc_info=True)

        # --- Lấy features từ MLPAction ---
        mlp_input_feature_names = [ 
            "RSI", "ATR", "EMA_diff", "MACD", "MACD_signal", "ADX", "momentum",
            "log_volume", "VWAP", "BB_width", "VAH", "regime", "hour",
            "VWAP_ADX_interaction", "BB_EMA_sync", "volume_anomaly"
        ]
        mlp_input_list_raw = []
        missing_mlp_input = False
        for feat_name in mlp_input_feature_names:
            val = last_row.get(feat_name)
            if pd.isna(val):
                logging.warning(f"NaN found for MLP input feature '{feat_name}'. Using 0.")
                mlp_input_list_raw.append(0.0)
                missing_mlp_input = True
            else:
                try:
                    mlp_input_list_raw.append(float(val))
                except (ValueError, TypeError):
                    logging.warning(f"Could not convert MLP input '{feat_name}' value '{val}' to float. Using 0.")
                    mlp_input_list_raw.append(0.0)
                    missing_mlp_input = True
        mlp_input_numpy_raw = np.array(mlp_input_list_raw, dtype=np.float32)
        mlp_action_feature_val = self.mlp_action_fallback # Default fallback

        if self.mlp_action_scaler and self.mlp_action_analyzer:
            try:
                # <<< Scale input 16 features >>>
                mlp_input_numpy_scaled = self.mlp_action_scaler.transform(mlp_input_numpy_raw.reshape(1, -1)).flatten()
                if np.isfinite(mlp_input_numpy_scaled).all():
                    # <<< Dự đoán bằng MLP >>>
                    pa_output_numpy = self.mlp_action_analyzer.predict(mlp_input_numpy_scaled, fallback_value=self.mlp_action_fallback)
                    mlp_action_feature_val = pa_output_numpy.item() if pa_output_numpy is not None and pa_output_numpy.size > 0 else self.mlp_action_fallback
                else: logging.warning("NaN after scaling MLP input.")
            except Exception as pa_e: logging.error(f"Error scaling/predicting MLP: {pa_e}")
        else: logging.warning("MLP scaler or analyzer not available.")

        # <<< Feature từ MLP (giờ là vol_normalized_return) >>>
        mlp_output_feature = np.array([np.nan_to_num(mlp_action_feature_val)], dtype=np.float32) 

        # --- Lấy các features bổ sung ---
        additional_features = np.array([
            last_row.get("momentum", 0.0),
            last_row.get("order_imbalance", 0.0)
        ], dtype=np.float32) # Shape (2,)

        # --- *** LẤY SENTIMENT FEATURE *** ---
        active_sentiment_score = 0.0
        if hasattr(self.bot, 'sentiment_analyzer') and self.bot.sentiment_analyzer:
            try:
                # Sử dụng timestamp của dòng dữ liệu hiện tại
                sentiment_details = self.bot.sentiment_analyzer.get_detailed_active_sentiment(timestamp_for_obs)
                # Lấy điểm đã điều chỉnh theo confidence
                active_sentiment_score = sentiment_details.get("total_score_adj_confidence", 0.0)
                logging.debug(f"Sentiment score for obs at {timestamp_for_obs}: {active_sentiment_score:.4f}")
            except Exception as sent_e:
                logging.error(f"Error getting sentiment for observation at step {current_step_safe}: {sent_e}")
        sentiment_feature = np.array([active_sentiment_score], dtype=np.float32) # Shape (1,)
        
        try:
            # Tạo base_features từ obs_values (hoặc trực tiếp từ last_row)
            # Đảm bảo các key này tồn tại trong obs_values và đã được gán giá trị số
            base_features_list = [
                obs_values.get("close", 0.0),       # Lấy từ obs_values, fallback về 0.0 nếu thiếu
                obs_values.get("RSI", 50.0),        # Fallback về 50.0
                obs_values.get("ATR", 0.0),
                obs_values.get("volatility", 0.0)
            ]
            base_features = np.array(base_features_list, dtype=np.float32) # Shape (4,)

            # Tạo ob_extra_features từ obs_values
            ob_extra_features_list = [
                obs_values.get("bid_ask_spread", 0.0),
                obs_values.get("liquidity", 0.0)
            ]
            ob_extra_features = np.array(ob_extra_features_list, dtype=np.float32) # Shape (2,)
        except KeyError as ke:
            logging.error(f"MarketEnv ({symbol}) - KeyError preparing base/ob_extra features: {ke}. Returning zeros.")
            return np.zeros(self.observation_space_dim, dtype=np.float32)
        except Exception as prep_e:
            logging.error(f"MarketEnv ({symbol}) - Error preparing feature arrays: {prep_e}. Returning zeros.", exc_info=True)
            return np.zeros(self.observation_space_dim, dtype=np.float32)

        # --- Ghép nối các features ---
        try:
            logging.debug(f"Shapes before concat: base={base_features.shape}, ob_extra={ob_extra_features.shape}, lob={lob_features.shape}, mlp={mlp_output_feature.shape}, additional={additional_features.shape}, sentiment={sentiment_feature.shape}")
            # Kiểm tra kiểu dữ liệu và chuyển đổi 
            lob_features = np.asarray(lob_features, dtype=np.float32) # Đã là float32 từ LOBTransformer
            additional_features = np.asarray(additional_features, dtype=np.float32)
            sentiment_feature = np.array([active_sentiment_score], dtype=np.float32) # Nếu dùng sentiment

            # --- Ghép nối các mảng lại với nhau ---
            feature_parts = [
                base_features,          # Shape (4,)
                ob_extra_features,      # Shape (2,)
                lob_features,           # Shape (64,)
                mlp_output_feature, # Shape (1,)
                additional_features,     # Shape (2,)
                sentiment_feature    # Shape (1,) - Thêm vào nếu observation_space_dim=74
            ]

            # Kiểm tra xem tất cả các phần có phải là mảng NumPy không
            if not all(isinstance(part, np.ndarray) for part in feature_parts):
                logging.error(f"Not all feature parts are numpy arrays at step {current_step_safe}. Cannot concatenate.")
                return np.zeros(self.observation_space_dim, dtype=np.float32)

            # Thực hiện ghép nối
            obs_unscaled = np.concatenate(feature_parts)

            # --- KIỂM TRA CUỐI CÙNG ---
            # 1. Kiểm tra Dimension cuối cùng
            if obs_unscaled.shape[0] != self.observation_space_dim:
                logging.error(f"Final unscaled observation shape mismatch at step {current_step_safe}: "
                              f"Got {obs_unscaled.shape}, expected ({self.observation_space_dim},). This indicates an issue in feature calculation or concatenation order.")
                logging.warning(f"Returning zeros due to shape mismatch.")
                return np.zeros(self.observation_space_dim, dtype=np.float32)

            # 2. Kiểm tra NaN/Infinity
            if not np.isfinite(obs_unscaled).all():
                nan_inf_count = np.sum(~np.isfinite(obs_unscaled))
                logging.warning(f"NaN/Inf detected in final unscaled observation at step {current_step_safe} ({nan_inf_count} values). Replacing with zeros.")
                # Ghi lại các feature gây ra NaN/Inf 
                nan_inf_indices = np.where(~np.isfinite(obs_unscaled))[0]
                logging.debug(f"NaN/Inf indices: {nan_inf_indices}")
                obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0) # Thay thế bằng 0

            # 3. Đảm bảo đúng dtype
            return obs_unscaled.astype(np.float32)

        except ValueError as e:
             # Lỗi này thường xảy ra nếu các mảng trong list `feature_parts` có số chiều không khớp (ví dụ: 1D vs 2D)
             logging.error(f"ValueError during concatenation at step {current_step_safe}: {e}. Check shapes of feature parts.", exc_info=True)
             # Log shapes của các phần tử để debug
             shapes_str = ", ".join([str(getattr(p, 'shape', 'Not an array')) for p in feature_parts])
             logging.error(f"Shapes of parts: {shapes_str}")
             return np.zeros(self.observation_space_dim, dtype=np.float32)
        except Exception as e:
             # Bắt các lỗi không mong muốn khác
             logging.error(f"Unexpected error creating unscaled observation at step {current_step_safe}: {e}", exc_info=True)
             return np.zeros(self.observation_space_dim, dtype=np.float32)

    #  --- HÀM _calculate_reward ---
    def _calculate_reward(self, pnl: float, action: np.ndarray) -> float:
        W_PNL = 0.35; W_LIQUIDITY = 0.05; W_VOLATILITY = 0.35; W_TIMING = 0.25
        assert abs(W_PNL + W_LIQUIDITY + W_VOLATILITY + W_TIMING - 1.0) < 1e-6
        try:
            current_step_safe = min(self.current_step, len(self.data["15m"]) - 1)
            if current_step_safe < 0: return 0.0
            last_row = self.data["15m"].iloc[current_step_safe]

            reward_pnl = pnl * W_PNL; reward_liquidity = 0.0; reward_volatility = 0.0; reward_timing = 0.0 # Init

            # Liquidity
            order_book_idx_safe = min(current_step_safe, len(self.order_book_data) - 1) if self.order_book_data else -1
            order_book = self.order_book_data[order_book_idx_safe] if order_book_idx_safe >= 0 else {"bids": [], "asks": []}
            liquidity = sum(float(bid[1]) for bid in order_book.get("bids", [])[:5] if isinstance(bid, (list, tuple)) and len(bid) == 2) + \
                        sum(float(ask[1]) for ask in order_book.get("asks", [])[:5] if isinstance(ask, (list, tuple)) and len(ask) == 2)
            reward_liquidity = np.tanh(liquidity * 0.001) * W_LIQUIDITY

            # Volatility
            volatility = last_row.get("volatility", 0.0)
            volatility_reward_value_norm = np.tanh(np.sqrt(max(volatility, 0)) * 0.5)
            reward_volatility = volatility_reward_value_norm * W_VOLATILITY

            # Timing
            momentum = last_row.get("momentum", 0.0); order_imbalance = last_row.get("order_imbalance", 0.0)
            consensus_sign = np.sign(momentum) if np.sign(momentum) != 0 and np.sign(momentum) == np.sign(order_imbalance) else 0
            if consensus_sign != 0:
                action_value = action.item() if isinstance(action, np.ndarray) and action.size == 1 else action if isinstance(action, (int, float)) else 0.0
                intended_direction_sign = 1 if action_value > 0 else -1 if action_value < 0 else 0
                timing_reward_value = 0.0
                if intended_direction_sign == consensus_sign:
                    magnitude = min((abs(momentum) * 10) + (abs(order_imbalance) * 0.5), 0.6)
                    timing_reward_value = magnitude * 0.8
                elif intended_direction_sign == -consensus_sign:
                    magnitude = min((abs(momentum) * 5) + (abs(order_imbalance) * 0.25), 0.3)
                    timing_reward_value = -magnitude * 0.8
                reward_timing = timing_reward_value * W_TIMING

            total_reward = reward_pnl + reward_liquidity + reward_volatility + reward_timing
            total_reward = np.clip(total_reward, -2.0, 2.0)
        
            # Xử lý NaN/Inf cuối cùng và log kết quả
            final_reward = np.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)
            logging.debug(f"Calculated Reward (Step {self.current_step}): {final_reward:.4f}")
            return final_reward
        except Exception as e:
             logging.error(f"Error in _calculate_reward: {e}", exc_info=True)
             return 0.0

class EntryPointOptimizer:
    def __init__(self, bot, data: Dict[str, pd.DataFrame], order_book_data: List[dict], model_path: str = "sac_entry_model"):
        self.bot = bot
        self.env = DummyVecEnv([lambda: Monitor(MarketEnvironment(bot, data, order_book_data))])
        self.model_path = model_path
        self.actor_critic = None
        self.order_book_analyzer = (LOBTransformer().cuda() if torch.cuda.is_available() else LOBTransformer())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(f"{self.model_path}.zip"):
            logging.info(f"Loading pre-trained SAC model from {self.model_path}")
            try:
                self.actor_critic = SAC.load(self.model_path, env=self.env, device=self.device)
            except Exception as e:
                logging.error(f"Failed to load SAC model: {str(e)}")
                self.actor_critic = None
        
    async def train_entry_model(self, n_trials: int = 20, total_timesteps: int = 15000):
        logging.info("Re-creating SAC environment with latest data...")
        try:
            first_symbol = CONFIG["symbols"][0]
            latest_data = self.bot.data.get(first_symbol)
            latest_ob_data = [self.bot.data[first_symbol].get("order_book", {"bids":[], "asks":[]})] # Lấy OB đã lưu gần nhất
            if not latest_data or "15m" not in latest_data or latest_data["15m"].empty:
                logging.error(f"Cannot recreate SAC env: Missing latest data for {first_symbol}")
                return
            def make_new_env():
                return Monitor(MarketEnvironment(self.bot, latest_data, latest_ob_data))
            if hasattr(self.env, 'close'):
                try: self.env.close()
                except Exception as close_e: logging.warning(f"Error closing old SAC env: {close_e}")
            # Tạo VecEnv mới
            self.env = DummyVecEnv([make_new_env])
            logging.info("SAC environment recreated.")
            # Nếu đang load model, cần cập nhật env cho model đó
            if self.actor_critic:
                self.actor_critic.set_env(self.env)

        except Exception as env_recreate_e:
            logging.error(f"Failed to recreate SAC environment: {env_recreate_e}", exc_info=True)
            pass


        if not hasattr(self.env.envs[0].env, 'setup_real_time'):
            logging.error("MarketEnvironment does not have setup_real_time method")
            return
        try:
            await self.env.envs[0].env.setup_real_time(CONFIG["symbols"][0])
            logging.info("Real-time setup completed")
        except Exception as e:
            logging.error(f"Failed to setup real-time data: {str(e)}")
            return
        logging.info("Optimizing SAC hyperparameters with Optuna...")

        def objective(trial):
            try:
                # --- Gợi ý siêu tham số (giữ nguyên) ---
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                buffer_size = trial.suggest_int("buffer_size", 10000, 100000)
                batch_size = trial.suggest_categorical("batch_size", [64, 128, 256]) # Dùng categorical và giá trị lớn hơn
                gamma = trial.suggest_float("gamma", 0.95, 0.999)
                tau = trial.suggest_float("tau", 0.001, 0.01)
                model = SAC(
                    "MlpPolicy",
                    self.env, 
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    gamma=gamma,
                    tau=tau,
                    # ent_coef=ent_coef, # Thêm nếu tối ưu
                    # target_update_interval=target_update_interval, # Thêm nếu tối ưu
                    verbose=0, # Giữ im lặng trong quá trình tối ưu
                    device=self.device,
                    seed=42 # Thêm seed để kết quả trial ổn định hơn
                )

                # --- Huấn luyện Model cho Trial ---
                train_steps_trial = 10000 
                self.env.reset()
                self.bot.trade_history = []
                self.bot.balance = CONFIG.get("initial_balance", 10000)
                self.bot.equity_peak = self.bot.balance
                # Reset exposure nếu cần
                for symbol in self.bot.exposure_per_symbol:
                     self.bot.exposure_per_symbol[symbol] = 0.0
                self.bot.open_positions = {}
                model.learn(total_timesteps=train_steps_trial, reset_num_timesteps=True, callback=self._sac_callback)

                # --- *** Đánh giá Trial bằng Reward từ Monitor *** ---
                trial_value = -float('inf') 

                if hasattr(self.env, 'envs') and self.env.envs:
                    monitor_env = self.env.envs[0] # Lấy môi trường Monitor gốc
                    ep_info_buffer = getattr(monitor_env, 'ep_info_buffer', None)

                    if ep_info_buffer and isinstance(ep_info_buffer, list) and len(ep_info_buffer) > 0:
                        # Lấy reward trung bình của các episode gần nhất (ví dụ: 5 episode cuối)
                        # Hoặc có thể lấy trung bình của tất cả episode trong trial nếu muốn
                        num_episodes_to_average = min(len(ep_info_buffer), 5) # Lấy tối đa 5 ep cuối
                        recent_rewards = [info['r'] for info in ep_info_buffer[-num_episodes_to_average:]]

                        if recent_rewards:
                            avg_reward = np.mean(recent_rewards)
                            trial_value = avg_reward # Dùng reward trung bình làm mục tiêu
                            logging.info(f"Trial {trial.number} completed. Avg Reward (last {num_episodes_to_average} eps): {avg_reward:.4f}")
                        else:
                            logging.warning(f"Trial {trial.number}: Found ep_info_buffer but no rewards in recent episodes.")
                    elif ep_info_buffer is not None: # Buffer tồn tại nhưng rỗng
                         logging.info(f"Trial {trial.number}: No completed episodes recorded by Monitor during training.")
                         trial_value = 0.0
                    else: # Không tìm thấy ep_info_buffer
                         logging.error("Could not access 'ep_info_buffer' from Monitor environment. Cannot evaluate trial using rewards.")
                         trial_value = -float('inf') # Hoặc coi như lỗi

                else: # Không truy cập được envs
                     logging.error("Could not access monitor environment within DummyVecEnv.")
                     trial_value = -float('inf')
                return trial_value # Trả về trực tiếp reward trung bình

            except Exception as e:
                logging.error(f"Error encountered in SAC Optuna trial {trial.number}: {e}", exc_info=True)
                return -float('inf')

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
            if study.best_value <= -float('inf'):
                logging.warning("Optuna failed to find valid parameters. Using default SAC parameters.")
                best_params = {
                    "learning_rate": 0.0003,
                    "buffer_size": 10000,
                    "batch_size": 64,
                    "gamma": 0.99,
                    "tau": 0.005
                }
            else:
                best_params = study.best_params
                logging.info(f"Best SAC parameters from Optuna: {best_params}")
            
            self.actor_critic = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=best_params["learning_rate"],
                buffer_size=best_params["buffer_size"],
                batch_size=best_params["batch_size"],
                gamma=best_params["gamma"],
                tau=best_params["tau"],
                verbose=1,
                device=self.device
            )
            logging.info("Training SAC model with selected parameters...")
            self.actor_critic.learn(total_timesteps=total_timesteps, callback=self._sac_callback)
            self.actor_critic.save(self.model_path)
            logging.info(f"SAC model training completed and saved to {self.model_path}")
        except Exception as e:
            logging.error(f"Error during final SAC training: {str(e)}")
            self.actor_critic = None
            raise
    def _sac_callback(self, _locals, _globals):
         time.sleep(0.01)
         return True
            
    def get_entry_signal(self, state: np.ndarray) -> Optional[str]:
        if self.actor_critic is None:
            logging.error("SAC model not trained or loaded!")
            return None
        if state.shape != (29,):
            logging.error(f"Invalid state shape: got {state.shape}, expected (29,)")
            return None
        try:
            action, _ = self.actor_critic.predict(state, deterministic=True)
            return "LONG" if action > 0 else "SHORT" if action < 0 else None
        except Exception as e:
            logging.error(f"Error in SAC prediction: {str(e)}")
            return None
    def is_trained(self) -> bool:
        return self.actor_critic is not None

# 5. PNL Optimization with Bayesian Neural Networks (Cải tiến với Correlation)
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

    def model(self, x, y):
        w1_prior = dist.Normal(torch.zeros(self.layer1.weight.shape), torch.ones(self.layer1.weight.shape)).to_event(2)
        b1_prior = dist.Normal(torch.zeros(self.layer1.bias.shape), torch.ones(self.layer1.bias.shape)).to_event(1)
        w2_prior = dist.Normal(torch.zeros(self.layer2.weight.shape), torch.ones(self.layer2.weight.shape)).to_event(2)
        b2_prior = dist.Normal(torch.zeros(self.layer2.bias.shape), torch.ones(self.layer2.bias.shape)).to_event(1)

        priors = {
            "layer1.weight": w1_prior, "layer1.bias": b1_prior,
            "layer2.weight": w2_prior, "layer2.bias": b2_prior
        }
        lifted_module = pyro.random_module("module", self, priors)
        lifted_model = lifted_module()
        with pyro.plate("data", x.shape[0]):
            out = lifted_model(x)
            pyro.sample("obs", dist.Normal(out, 1.0), obs=y)

    def guide(self, x, y):
        w1_mu = torch.randn(self.layer1.weight.shape)
        w1_sigma = torch.randn(self.layer1.weight.shape)
        b1_mu = torch.randn(self.layer1.bias.shape)
        b1_sigma = torch.randn(self.layer1.bias.shape)
        w2_mu = torch.randn(self.layer2.weight.shape)
        w2_sigma = torch.randn(self.layer2.weight.shape)
        b2_mu = torch.randn(self.layer2.bias.shape)
        b2_sigma = torch.randn(self.layer2.bias.shape)

        w1_dist = dist.Normal(w1_mu, torch.exp(w1_sigma)).to_event(2)
        b1_dist = dist.Normal(b1_mu, torch.exp(b1_sigma)).to_event(1)
        w2_dist = dist.Normal(w2_mu, torch.exp(w2_sigma)).to_event(2)
        b2_dist = dist.Normal(b2_mu, torch.exp(b2_sigma)).to_event(1)

        dists = {
            "layer1.weight": w1_dist, "layer1.bias": b1_dist,
            "layer2.weight": w2_dist, "layer2.bias": b2_dist
        }
        lifted_module = pyro.random_module("module", self, dists)
        return lifted_module()
    
class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Đổi thành buffer để không được coi là tham số model
        self.register_buffer('pe', pe.transpose(0, 1), persistent=False) # batch_first=True requires (batch, seq, feature) -> (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) là seq_len
        # self.pe shape là (1, max_len, d_model)
        # Cần lấy phần pe tương ứng với seq_len của input: self.pe[:, :x.size(1)]
        # Shape của phần pe được lấy: (1, seq_len, d_model)
        # Broadcasting sẽ tự động áp dụng cho batch_size
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 6. Dynamic Stop-Loss System (Cải tiến với GNN và TransformerForecaster)
class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1): # Giữ nguyên tham số dropout
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        logging.info(f"Initializing TransformerForecaster with input_dim={input_dim}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dropout={dropout}") # Có thể bỏ log này nếu ISS load nhiều lần
        self.encoder = nn.Linear(input_dim, d_model)
        # <<< THÊM PositionalEncoding >>>
        self.pos_encoder = PositionalEncoding(d_model, dropout) # Sử dụng dropout rate của model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout) # Truyền dropout vào layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Xử lý dữ liệu đầu vào và trả về dự đoán."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Kích thước chiều cuối của tensor đầu vào ({x.shape[-1]}) không khớp với input_dim của mô hình ({self.input_dim})")
        device = next(self.parameters()).device
        x = x.to(device)
        x = F.relu(self.encoder(x))
        # <<< ÁP DỤNG Positional Encoding >>>
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.decoder(x)

    @torch.no_grad()
    def predict_volatility(self, lookback_data_tensor: torch.Tensor, forward_periods: int) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        if lookback_data_tensor.device != device:
             lookback_data_tensor = lookback_data_tensor.to(device)

        if not (lookback_data_tensor.ndim == 3 and lookback_data_tensor.shape[0] == 1 and lookback_data_tensor.shape[2] == self.input_dim):
             logging.error(f"TransformerForecaster predict_volatility: Invalid input tensor shape: {lookback_data_tensor.shape}. Expected (1, seq_len, {self.input_dim})")
             return np.full(forward_periods, np.nan)
        if lookback_data_tensor.shape[1] == 0:
             logging.error(f"TransformerForecaster predict_volatility: Input tensor has zero sequence length.")
             return np.full(forward_periods, np.nan)

        try:
            pred = self.forward(lookback_data_tensor) # Shape: (1, seq_len, 1)
            if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[2] != 1 or pred.shape[1] == 0:
                logging.error(f"Shape đầu ra không mong đợi từ forward: {pred.shape}. Kỳ vọng (1, seq_len, 1)")
                return np.full(forward_periods, np.nan)

            last_pred_value = pred[0, -1, 0].item()

            # <<< SỬA LOGIC TRẢ VỀ >>>
            if forward_periods == 1:
                # Trả về dự đoán cho bước tiếp theo
                return np.array([last_pred_value])
            else:
                # Mô hình chỉ được huấn luyện cho 1 bước.
                # Trả về dự báo naive persistence cho các bước tiếp theo.
                logging.warning(f"TransformerForecaster predicting {forward_periods} steps using naive persistence (repeating last value). Model was trained for 1 step.")
                return np.array([last_pred_value] * forward_periods)

        except Exception as e:
            logging.error(f"Lỗi trong quá trình dự đoán volatility: {e}", exc_info=True)
            return np.full(forward_periods, np.nan)

class GraphNeuralNetPyG_Advanced(nn.Module):
    def __init__(self, node_dim, num_layers=3, edge_dim=2, hidden_dim=64, out_dim=1, dropout=0.3, heads=4):
        super().__init__()
        if not _torch_geometric_available: raise ImportError("torch_geometric required")
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim if isinstance(edge_dim, int) and edge_dim > 0 else None
        intermediate_dim = hidden_dim // heads
        self.num_layers = num_layers # Lưu số lớp
        self.input_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList(); self.skips = nn.ModuleList()
        in_channels = node_dim
        for i in range(num_layers):
            out_channels_conv = hidden_dim if i < num_layers - 1 else hidden_dim
            current_heads = heads if i < num_layers - 1 else 1
            concat_final = True if i < num_layers - 1 else False
            out_channels_actual = out_channels_conv
            if i == 0: self.skips.append(nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity())
            elif in_channels != hidden_dim: self.skips.append(nn.Linear(in_channels, hidden_dim))
            else: self.skips.append(nn.Identity())
            self.convs.append(GraphConv(in_channels, hidden_dim // current_heads if current_heads > 0 else hidden_dim, heads=current_heads, dropout=dropout, edge_dim=self.edge_dim, concat=concat_final))
            self.bns.append(nn.BatchNorm1d(out_channels_actual))
            in_channels = out_channels_actual
        self.dropout_layer = nn.Dropout(p=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout/2), nn.Linear(hidden_dim, out_dim))
        logging.info(f"Initialized FINAL GNN (GAT) Layers={num_layers}, Hidden={hidden_dim}, Heads={heads}, Dropout={dropout}")
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]=None, batch: Optional[torch.Tensor]=None) -> torch.Tensor:
        target_device = x.device if x is not None else next(self.parameters()).device
        if x is None or edge_index is None or x.nelement()==0: batch_size=batch.max().item()+1 if batch is not None and batch.numel()>0 else 1; return torch.zeros((batch_size,1),device=target_device)
        try:
            use_edge_attr = self.edge_dim is not None and edge_attr is not None
            if use_edge_attr and edge_attr.shape[1] != self.edge_dim:
                use_edge_attr = False

            identity = x
            for i in range(self.num_layers):
                layer_identity=self.skips[i](identity); x_in=x
                x=self.convs[i](x_in,edge_index,edge_attr=edge_attr) if use_edge_attr else self.convs[i](x_in,edge_index)
                x=self.bns[i](x); x=F.elu(x); x=self.dropout_layer(x)
                try:
                    if x.shape==layer_identity.shape: x=x+layer_identity
                except RuntimeError as res_e: logging.error(f"RuntimeError Residual {i+1}: {res_e}")
                identity=x
            if batch is None: graph_embedding=torch.mean(x,dim=0,keepdim=True); graph_embedding=torch.cat([graph_embedding,graph_embedding],dim=1)
            else: mean_pool=global_mean_pool(x,batch); max_pool=global_max_pool(x,batch); graph_embedding=torch.cat([mean_pool,max_pool],dim=1)
            out_logit=self.predictor(graph_embedding); return out_logit
        except Exception as e: logging.error(f"GNN forward error: {e}"); batch_size=batch.max().item()+1 if batch is not None and batch.numel()>0 else 1; return torch.zeros((batch_size,1),device=target_device)

class IntelligentStopSystem:
    def __init__(self): 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transformer_features = CONFIG.get("iss_transformer_features", ["close", "RSI", "ATR", "volatility", "volume"])
        self.gnn_features = CONFIG.get("iss_gnn_features", ["close", "RSI", "ATR", "volatility", "volume"])
        self.node_dim = len(self.gnn_features)
        self.gnn_hparams = {}
        default_gnn_hparams = {'hidden_dim': 64, 'dropout': 0.3, 'heads': 4, 'gnn_layers': 3, 'edge_dim': 2}

        # --- Load GNN Hparams ---
        if GNN_HPARAMS_PATH.exists():
            try:
                with open(GNN_HPARAMS_PATH, 'r') as f: self.gnn_hparams = json.load(f)
                logging.info(f"ISS: Loaded GNN hyperparameters from {GNN_HPARAMS_PATH}")
                for key, default_val in default_gnn_hparams.items():
                     if key not in self.gnn_hparams: self.gnn_hparams[key] = default_val
            except Exception as e:
                logging.error(f"ISS: Error loading GNN hyperparameters: {e}. Using defaults.")
                self.gnn_hparams = default_gnn_hparams
        else:
            logging.warning(f"ISS: GNN hyperparameters file {GNN_HPARAMS_PATH} not found. Using defaults.")
            self.gnn_hparams = default_gnn_hparams
        
        self.best_gnn_threshold = 0.5 # Giá trị mặc định nếu không tải được
        if THRESHOLD_SAVE_PATH.exists():
            try:
                with open(THRESHOLD_SAVE_PATH, 'r') as f:
                    threshold_data = json.load(f)
                    # <<< Lấy ngưỡng từ key "best_threshold" >>>
                    loaded_threshold = threshold_data.get("best_threshold")
                    if isinstance(loaded_threshold, (float, int)) and 0 < loaded_threshold < 1:
                        self.best_gnn_threshold = loaded_threshold
                        logging.info(f"ISS: Loaded best GNN threshold: {self.best_gnn_threshold:.2f}")
                    else:
                         logging.warning(f"Invalid threshold value '{loaded_threshold}' in {THRESHOLD_SAVE_PATH}. Using default 0.5.")
            except Exception as e:
                logging.error(f"ISS: Error loading GNN threshold from {THRESHOLD_SAVE_PATH}: {e}. Using default 0.5.")
        else:
            logging.warning(f"ISS: Best GNN threshold file {THRESHOLD_SAVE_PATH} not found. Using default 0.5.")

        # --- Khởi tạo Models ---
        self.volatility_forecaster = None # Khởi tạo là None
        self.anomaly_detector = None   # Khởi tạo là None
        try:
            best_d_model = CONFIG.get("iss_transformer_d_model", 256)
            best_nhead = CONFIG.get("iss_transformer_nhead", 16)
            best_num_layers = CONFIG.get("iss_transformer_num_layers", 2)
            best_dropout = CONFIG.get("iss_transformer_dropout", 0.11418798847562715)

            self.volatility_forecaster = TransformerForecaster(
                input_dim=len(self.transformer_features), # Sử dụng len() trực tiếp
                d_model=best_d_model, nhead=best_nhead,
                num_layers=best_num_layers, dropout=best_dropout
            ).to(self.device)
            logging.info(f"ISS: Initialized TransformerForecaster with PE. input_dim={len(self.transformer_features)}, d_model={best_d_model}, nhead={best_nhead}, layers={best_num_layers}")

            if _torch_geometric_available:
                self.anomaly_detector = GraphNeuralNetPyG_Advanced(
                    node_dim=self.node_dim, num_layers=self.gnn_hparams.get('gnn_layers', 3),
                    edge_dim=self.gnn_hparams.get('edge_dim', 2), hidden_dim=self.gnn_hparams.get('hidden_dim', 64),
                    dropout=self.gnn_hparams.get('dropout', 0.3), heads=self.gnn_hparams.get('heads', 4)
                ).to(self.device)
                logging.info(f"ISS: Initialized GNN with loaded hparams.")
            else:
                 logging.error("ISS: torch_geometric not available, GNN anomaly detector disabled.")

        except Exception as e:
            logging.error(f"Failed to initialize ISS base models: {e}", exc_info=True)
            # volatility_forecaster và anomaly_detector sẽ giữ giá trị None

        # --- Load Weights (sau khi khởi tạo) ---
        if self.volatility_forecaster is not None:
            try:
                if ISS_MODEL_PATH.exists():
                    state_dict = torch.load(ISS_MODEL_PATH, map_location='cpu')
                    self.volatility_forecaster.load_state_dict(state_dict)
                    self.volatility_forecaster.to(self.device)
                    self.volatility_forecaster.eval()
                    logging.info(f"ISS: Successfully loaded TransformerForecaster weights from {ISS_MODEL_PATH}")
                else:
                    logging.warning(f"ISS: Trained Transformer model weights not found: {ISS_MODEL_PATH}. Using random weights.")
                    self.volatility_forecaster.eval()
            except Exception as e:
                logging.error(f"ISS: Failed to load TransformerForecaster weights: {e}. Using random weights.")
                if self.volatility_forecaster: self.volatility_forecaster.eval()
        else: logging.warning("ISS: Volatility forecaster model was not initialized, skipping weight loading.")

        if self.anomaly_detector is not None: # Chỉ load nếu GNN được khởi tạo
             if GNN_MODEL_PATH.exists():
                 try:
                     state_dict_gnn = torch.load(GNN_MODEL_PATH, map_location=self.device)
                     self.anomaly_detector.load_state_dict(state_dict_gnn)
                     logging.info(f"ISS: Successfully loaded trained GNN weights from {GNN_MODEL_PATH}")
                 except Exception as load_e:
                     logging.error(f"ISS: Failed to load GNN weights: {load_e}. Using random weights.")
             else:
                 logging.warning(f"ISS: Trained GNN model not found at {GNN_MODEL_PATH}. Using random weights.")
             self.anomaly_detector.eval() # Đặt eval mode ngay cả khi dùng random weights

        # --- Load Scalers ---
        self.input_scaler = None; self.target_scaler = None; self.gnn_scaler = None
        try:
            if ISS_INPUT_SCALER_PATH.exists():
                self.input_scaler = joblib.load(ISS_INPUT_SCALER_PATH)
                logging.info(f"ISS: Loaded input scaler from {ISS_INPUT_SCALER_PATH}")
                if hasattr(self.input_scaler, 'n_features_in_') and self.input_scaler.n_features_in_ != len(self.transformer_features):
                    logging.error(f"ISS: Input scaler dim mismatch! Expected {len(self.transformer_features)}, got {self.input_scaler.n_features_in_}. Disabling.")
                    self.input_scaler = None
            else: logging.warning(f"ISS: Input scaler not found: {ISS_INPUT_SCALER_PATH}.")

            if ISS_TARGET_SCALER_PATH.exists():
                self.target_scaler = joblib.load(ISS_TARGET_SCALER_PATH)
                logging.info(f"ISS: Loaded target scaler from {ISS_TARGET_SCALER_PATH}")
                if hasattr(self.target_scaler, 'n_features_in_') and self.target_scaler.n_features_in_ != 1:
                     logging.error(f"ISS: Target scaler dim mismatch! Expected 1, got {self.target_scaler.n_features_in_}. Disabling.")
                     self.target_scaler = None
            else: logging.warning(f"ISS: Target scaler not found: {ISS_TARGET_SCALER_PATH}.")

            if GNN_SCALER_PATH.exists() and _torch_geometric_available and self.anomaly_detector is not None: # Chỉ load nếu GNN tồn tại
                self.gnn_scaler = joblib.load(GNN_SCALER_PATH)
                logging.info(f"ISS: Loaded GNN input scaler from {GNN_SCALER_PATH}")
                if hasattr(self.gnn_scaler, 'n_features_in_') and self.gnn_scaler.n_features_in_ != self.node_dim:
                     logging.error(f"ISS: GNN scaler dim mismatch! Expected {self.node_dim}, got {self.gnn_scaler.n_features_in_}. Disabling.")
                     self.gnn_scaler = None
            elif _torch_geometric_available: logging.warning(f"ISS: GNN scaler not found or GNN model not available: {GNN_SCALER_PATH}.")

        except Exception as e:
            logging.error(f"ISS: Error loading scalers: {e}")
            self.input_scaler = None; self.target_scaler = None; self.gnn_scaler = None

        # <<< Bỏ đoạn kiểm tra dimension dư thừa ở đây >>>

    def _predict_volatility(self, df: pd.DataFrame, lookback_period: int, symbol: str) -> Optional[np.ndarray]:
        """Helper để dự báo biến động (Sử dụng transformer_features)."""
        if self.volatility_forecaster is None or self.input_scaler is None or self.target_scaler is None:
             # logging.debug(f"ISS ({symbol}): Volatility prediction skipped (model/scaler missing).") # Giảm log
             return None
        if len(df) < lookback_period: return None

        try:
            if not all(col in df.columns for col in self.transformer_features):
                 missing_tf_feat = [c for c in self.transformer_features if c not in df.columns]
                 logging.error(f"ISS ({symbol}): Missing required features for Transformer prediction: {missing_tf_feat}")
                 return None
            lookback_df_unscaled = df[self.transformer_features].iloc[-lookback_period:].copy()

            if lookback_df_unscaled.isnull().values.any():
                lookback_df_unscaled.fillna(method='ffill', inplace=True); lookback_df_unscaled.fillna(method='bfill', inplace=True); lookback_df_unscaled.fillna(0, inplace=True)
                if lookback_df_unscaled.isnull().values.any(): return None
            if lookback_df_unscaled.shape != (lookback_period, len(self.transformer_features)):
                 logging.error(f"ISS ({symbol}): Lookback data shape mismatch {lookback_df_unscaled.shape}, expected ({lookback_period}, {len(self.transformer_features)}).")
                 return None

            input_features_scaled = self.input_scaler.transform(lookback_df_unscaled.values)
            lookback_tensor = torch.tensor(input_features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        except Exception as data_prep_e:
             logging.error(f"ISS ({symbol}): Error preparing/scaling input data for prediction: {data_prep_e}", exc_info=False) # Giảm log traceback
             return None

        try:
            predicted_value_array = self.volatility_forecaster.predict_volatility(lookback_tensor, forward_periods=1)

            if predicted_value_array is None or np.isnan(predicted_value_array).any() or predicted_value_array.size != 1:
                 # logging.debug(f"ISS ({symbol}): Invalid prediction received from predict_volatility.") # Giảm log
                 return None

            prediction_value = predicted_value_array.item()
            scaled_pred_reshape = np.array([[prediction_value]])
            unscaled_prediction = self.target_scaler.inverse_transform(scaled_pred_reshape)
            final_prediction_value = unscaled_prediction.item()

            if not np.isfinite(final_prediction_value):
                 logging.error(f"ISS ({symbol}): Prediction became NaN/Inf after inverse transform.")
                 return None

            return np.array([final_prediction_value])

        except Exception as model_pred_e:
            logging.error(f"ISS ({symbol}): Error during model prediction or unscaling: {model_pred_e}", exc_info=False) # Giảm log traceback
            return None

    def calculate_sl(self, position: dict, data: dict, lookback_period: int = 30) -> float:
        # --- 1. Trích xuất thông tin và Kiểm tra Đầu vào Cơ bản ---
        symbol = position.get("symbol", "N/A")
        entry_price = position.get("entry", 0.0)
        direction = position.get("direction", "NONE")
        initial_sl_adjusted = position.get("initial_sl_adjusted", 0.0) # SL đã điều chỉnh từ bên ngoài (ví dụ: dựa trên ATR ban đầu)

        # Kiểm tra entry_price và direction
        if entry_price <= 0 or direction not in ["LONG", "SHORT"]:
            logging.error(f"ISS ({symbol}): Invalid entry price ({entry_price}) or direction ('{direction}'). Returning simple fallback SL.")
            # Fallback rất đơn giản nếu không có entry/direction
            return entry_price * (1.02 if direction == "SHORT" else 0.98)

        # Kiểm tra initial_sl_adjusted (phải hợp lệ so với entry và direction)
        is_initial_sl_invalid = (
            initial_sl_adjusted == 0.0 or
            (direction == "LONG" and initial_sl_adjusted >= entry_price) or
            (direction == "SHORT" and initial_sl_adjusted <= entry_price)
        )

        if is_initial_sl_invalid:
            logging.warning(f"ISS ({symbol}): Invalid initial_sl_adjusted ({initial_sl_adjusted:.4f}) received. Attempting ATR-based fallback.")
            try:
                df_15m_fallback = data.get("15m")
                # <<< KIỂM TRA THÊM df_15m_fallback và cột ATR >>>
                if df_15m_fallback is not None and not df_15m_fallback.empty and 'ATR' in df_15m_fallback.columns:
                     # Lấy ATR cuối cùng, kiểm tra NaN/Zero
                     atr_fallback = df_15m_fallback['ATR'].iloc[-1]
                     if pd.notna(atr_fallback) and atr_fallback > 1e-9: # Thêm kiểm tra > 0 (an toàn hơn)
                          fallback_sl = entry_price - 2 * atr_fallback if direction == "LONG" else entry_price + 2 * atr_fallback
                          logging.info(f"ISS ({symbol}): Using ATR-based fallback SL: {fallback_sl:.4f}")
                          return fallback_sl
                     else:
                          logging.warning(f"ISS ({symbol}): Invalid ATR value ({atr_fallback}) for fallback. Using simple % fallback.")
                else:
                     logging.warning(f"ISS ({symbol}): Missing data for ATR-based fallback. Using simple % fallback.")
            except Exception as fallback_e:
                 logging.error(f"ISS ({symbol}): Error during ATR fallback calculation: {fallback_e}. Using simple % fallback.")
            # Fallback cuối cùng nếu ATR lỗi hoặc không có dữ liệu
            return entry_price * (1.02 if direction == "SHORT" else 0.98)

        # Nếu initial_sl_adjusted hợp lệ, nó sẽ là SL mặc định nếu có lỗi sau này
        default_sl = initial_sl_adjusted

        # --- 2. Kiểm tra Dữ liệu Timeframe Chính (15m) ---
        if "15m" not in data or not isinstance(data["15m"], pd.DataFrame) or data["15m"].empty:
            logging.warning(f"ISS ({symbol}): Missing or empty '15m' data. Returning initial adjusted SL: {default_sl:.4f}")
            return default_sl
        df_15m = data["15m"]

        # Kiểm tra độ dài dữ liệu tối thiểu
        # <<< SỬA: Cần max của lookback_period (cho Transformer) và graph_lookback (cho GNN) >>>
        min_data_len = max(lookback_period, 20) # Giả sử graph_lookback cố định là 20 trong _detect_anomaly
        if len(df_15m) < min_data_len:
            logging.warning(f"ISS ({symbol}): Not enough '15m' data ({len(df_15m)} < {min_data_len}). Returning initial adjusted SL: {default_sl:.4f}")
            return default_sl

        # Kiểm tra sự tồn tại và NaN của các cột cần thiết
        # <<< Lấy feature lists từ self >>>
        tf_features = self.transformer_features
        gnn_features = self.gnn_features
        required_cols = set(tf_features + gnn_features + ["ATR", "close"]) 
        missing_cols = [col for col in required_cols if col not in df_15m.columns]
        if missing_cols:
            logging.warning(f"ISS ({symbol}): Missing required columns in 15m data: {missing_cols}. Returning initial adjusted SL.")
            return default_sl
        try:
            last_row = df_15m.iloc[-1]
            # Kiểm tra NaN kỹ hơn ở dòng cuối
            if last_row[list(required_cols)].isnull().values.any():
                nan_cols = last_row[list(required_cols)].isnull()
                logging.warning(f"ISS ({symbol}): NaN found in last row for columns: {nan_cols[nan_cols].index.tolist()}. Returning initial adjusted SL.")
                return default_sl
        except IndexError:
             logging.error(f"ISS ({symbol}): IndexError accessing last row. Returning initial adjusted SL.")
             return default_sl


        # --- 3. Dự báo Biến động (Transformer) ---
        current_atr = last_row['ATR'] # Lấy ATR hiện tại (đã kiểm tra NaN)
        mean_vol_forecast = current_atr # Mặc định fallback là ATR hiện tại

        # Chỉ dự báo nếu có đủ thành phần
        if self.volatility_forecaster and self.input_scaler and self.target_scaler:
             try:
                 # Gọi hàm dự báo (hàm này nên trả về giá trị đã inverse_transform)
                 vol_forecast_result = self._predict_volatility(df_15m.tail(lookback_period), lookback_period, symbol) # Truyền đủ dữ liệu lookback
                 # Kiểm tra kết quả dự báo
                 if vol_forecast_result is not None and not np.isnan(vol_forecast_result).all() and np.mean(vol_forecast_result) > 1e-9: # Thêm kiểm tra > 0
                      mean_vol_forecast = np.nanmean(vol_forecast_result)
                      logging.info(f"ISS ({symbol}): Volatility Forecast (mean): {mean_vol_forecast:.6f} (Current ATR: {current_atr:.6f})")
                 else:
                      logging.warning(f"ISS ({symbol}): Volatility prediction invalid ({vol_forecast_result}). Using current ATR.")
             except Exception as vol_pred_e:
                  logging.error(f"ISS ({symbol}): Error during volatility prediction: {vol_pred_e}. Using current ATR.")
        else:
             logging.debug(f"ISS ({symbol}): Volatility prediction skipped (model/scalers missing). Using current ATR.")

        # --- 4. Phát hiện Bất thường (GNN) ---
        anomaly_prob = 0.0 # Xác suất bất thường (0-1)
        # Chỉ dự báo nếu có đủ thành phần
        if self.anomaly_detector and self.gnn_scaler:
            try:
                 # Gọi hàm dự báo (trả về xác suất)
                 # <<< Giả sử graph_lookback cố định là 20 như trong hàm gốc >>>
                 anomaly_prob = self._detect_anomaly(df_15m, symbol, graph_lookback=20)
                 logging.info(f"ISS ({symbol}): Anomaly Probability: {anomaly_prob:.4f} (Threshold: {self.best_gnn_threshold:.2f})")
            except Exception as anomaly_e:
                 logging.error(f"ISS ({symbol}): Error during anomaly detection: {anomaly_e}. Anomaly score set to 0.")
        else:
            logging.debug(f"ISS ({symbol}): Anomaly detection skipped (model/scaler missing).")


        # --- 5. Tinh chỉnh SL dựa trên Dự báo và Bất thường ---
        refined_sl = initial_sl_adjusted # Bắt đầu từ SL đã điều chỉnh ban đầu
        anomaly_adj_amount = 0.0
        vol_adj_amount = 0.0

        # Chỉ điều chỉnh nếu ATR hiện tại hợp lệ
        if pd.notna(current_atr) and current_atr > 1e-9:
             # a) Điều chỉnh dựa trên Bất thường GNN (Sử dụng ngưỡng)
             # <<< ÁP DỤNG NGƯỠNG TỐT NHẤT >>>
             if anomaly_prob >= self.best_gnn_threshold:
                 logging.info(f"ISS ({symbol}): Anomaly detected. Applying SL adjustment.")
                 # Logic SIẾT SL (mặc định)
                 confidence_factor = (anomaly_prob - self.best_gnn_threshold) / (1.0 - self.best_gnn_threshold + 1e-9) # Thêm epsilon tránh chia 0
                 anomaly_tighten_factor = np.clip(confidence_factor, 0, 1) * 0.1 # Hệ số ảnh hưởng tối đa 0.1
                 anomaly_adj_amount = anomaly_tighten_factor * current_atr
                 # # HOẶC: Logic NỚI SL (nếu muốn thử)
                 # anomaly_widening_factor = np.clip(confidence_factor, 0, 1) * 0.20
                 # anomaly_adj_amount = - (anomaly_widening_factor * current_atr) # Dấu trừ để nới

             # b) Điều chỉnh dựa trên Dự báo Biến động (Transformer - Logic siết)
             vol_tighten_factor = 0.0
             if mean_vol_forecast > current_atr: # Chỉ siết nếu dự báo cao hơn hiện tại
                  # Tính mức độ khác biệt tương đối
                  vol_diff_ratio = (mean_vol_forecast / current_atr) - 1
                  # Áp dụng hệ số ảnh hưởng (ví dụ: 0.1) và giới hạn
                  vol_tighten_factor = np.clip(vol_diff_ratio * 0.1, 0, 0.1)
             vol_adj_amount = vol_tighten_factor * current_atr # Luôn là giá trị >= 0 (chỉ siết hoặc không làm gì)

             # c) Áp dụng tổng hợp các điều chỉnh
             logging.debug(f"ISS SL Adjustments ({symbol}): AnomalyAdj={anomaly_adj_amount:.4f}, VolAdj={vol_adj_amount:.4f}")
             if direction == "LONG":
                 refined_sl += vol_adj_amount # Siết do Transformer
                 refined_sl += anomaly_adj_amount # Siết (hoặc nới) do GNN
             elif direction == "SHORT":
                 refined_sl -= vol_adj_amount # Siết do Transformer
                 refined_sl -= anomaly_adj_amount # Siết (hoặc nới) do GNN

             # d) Đảm bảo SL không vượt qua giá vào lệnh sau khi tinh chỉnh
             # Dùng % nhỏ để đảm bảo khoảng cách tối thiểu
             min_gap_percent = 1e-5
             if direction == "LONG":
                 refined_sl = min(refined_sl, entry_price * (1 - min_gap_percent))
             elif direction == "SHORT":
                 refined_sl = max(refined_sl, entry_price * (1 + min_gap_percent))
        else:
             logging.warning(f"ISS ({symbol}): Invalid current ATR ({current_atr}). Skipping SL refinement steps.")

        logging.debug(f"ISS Refined SL ({symbol}): {refined_sl:.4f} (Initial Adjusted: {initial_sl_adjusted:.4f})")

        # --- 6. Áp dụng các Biện pháp Bảo vệ Cuối cùng ---
        # Hàm _apply_safeguards sẽ kiểm tra thêm các giới hạn (như max distance)
        # <<< Đảm bảo truyền đúng các tham số cần thiết cho position trong _apply_safeguards >>>
        final_sl = self._apply_safeguards(refined_sl, initial_sl_adjusted, position, df_15m, symbol)

        logging.info(f"ISS Final SL Calculated ({symbol}): {final_sl:.4f}")

        # --- 7. Kiểm tra Hợp lệ Cuối cùng ---
        if (direction == "LONG" and final_sl >= entry_price) or \
           (direction == "SHORT" and final_sl <= entry_price):
            logging.error(f"ISS ({symbol}): Invalid final SL ({final_sl:.4f}) relative to Entry ({entry_price:.4f}). Critical error or safeguard issue. Returning initial adjusted SL.")
            # Trả về initial_sl_adjusted an toàn hơn là fallback %
            return initial_sl_adjusted

        return final_sl

    def _detect_anomaly(self, df: pd.DataFrame, symbol: str, graph_lookback: int = 20) -> float:
        """Helper để phát hiện bất thường bằng GNN."""
        if self.anomaly_detector is None: return 0.0
        if len(df) < graph_lookback: return 0.0

        try:
            # <<< SỬA LỖI: Nhận đủ 3 giá trị >>>
            node_features, edge_index, edge_attr = self.build_market_graph(df, graph_lookback, symbol)

            if node_features is None or edge_index is None: return 0.0

            expected_node_dim = len(self.gnn_features)
            if node_features.shape != (graph_lookback, expected_node_dim):
                 logging.error(f"ISS ({symbol}): Shape node_features không đúng ({node_features.shape}) cho GNN.")
                 return 0.0

            self.anomaly_detector.eval()
            with torch.no_grad():
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                # <<< SỬA LỖI: Chuyển edge_attr đến device NẾU nó tồn tại >>>
                if edge_attr is not None:
                    edge_attr = edge_attr.to(self.device)

                # <<< SỬA LỖI: Truyền edge_attr vào model >>>
                anomaly_logits = self.anomaly_detector(node_features, edge_index, edge_attr=edge_attr, batch=None)
                anomaly_score = torch.sigmoid(anomaly_logits).squeeze().item()
                if not np.isfinite(anomaly_score): anomaly_score = 0.0
            return anomaly_score
        except Exception as e:
            logging.error(f"ISS ({symbol}): Lỗi trong quá trình phát hiện bất thường GNN: {e}", exc_info=False) # Giảm log traceback
            return 0.0

    def build_market_graph(self, df: pd.DataFrame, lookback: int, symbol: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Helper để xây dựng input cho GNN."""
        if len(df) < lookback: return None, None, None
        required_features = self.gnn_features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features: logging.error(f"ISS ({symbol}): Missing GNN features {missing_features}"); return None, None, None

        graph_df_unscaled = df[required_features].tail(lookback).copy()
        if graph_df_unscaled.isnull().values.any():
            graph_df_unscaled.fillna(method='ffill', inplace=True)
            graph_df_unscaled.fillna(method='bfill', inplace=True)
            # <<< Thêm comment >>>
            graph_df_unscaled.fillna(0, inplace=True) # Lưu ý: fillna(0) có thể ảnh hưởng nếu 0 không hợp lý
            if graph_df_unscaled.isnull().values.any(): return None, None, None

        node_features_np = graph_df_unscaled.values
        if self.gnn_scaler:
             try:
                 if node_features_np.shape == (lookback, self.node_dim):
                      node_features_scaled = self.gnn_scaler.transform(node_features_np)
                      if not np.all(np.isfinite(node_features_scaled)):
                           node_features_scaled = np.nan_to_num(node_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                      node_features_np = node_features_scaled
                 else: logging.error(f"ISS ({symbol}): Shape mismatch before GNN scaling.")
             except Exception as scale_e: logging.error(f"ISS ({symbol}): Error applying GNN scaler: {scale_e}. Using unscaled.")

        if node_features_np.shape != (lookback, self.node_dim): return None, None, None

        try:
            node_features = torch.tensor(node_features_np, dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long); edge_attrs_list = []; full_edge_list = []; has_close_vol_indices = False; close_idx=-1; vol_idx=-1
            try: close_idx=self.gnn_features.index('close'); vol_idx=self.gnn_features.index('volume'); has_close_vol_indices=True
            except ValueError: pass
            if lookback >= 2:
                for i in range(lookback - 1):
                    src, dst = i, i + 1; full_edge_list.extend([(src, dst), (dst, src)])
                    if has_close_vol_indices:
                        try:
                            price_change = abs(node_features[dst, close_idx].item() - node_features[src, close_idx].item())
                            vol_change = abs(node_features[dst, vol_idx].item() - node_features[src, vol_idx].item())
                            attr = [price_change, vol_change]; edge_attrs_list.extend([attr, attr])
                        except Exception: has_close_vol_indices=False; edge_attrs_list=[]
                if full_edge_list: edge_index = torch.tensor(full_edge_list, dtype=torch.long).t().contiguous()
            edge_attr = None; expected_edge_dim = self.gnn_hparams.get('edge_dim', 2)
            if has_close_vol_indices and edge_attrs_list and edge_index.shape[1] > 0:
                edge_attr_tensor_temp = torch.tensor(edge_attrs_list, dtype=torch.float32)
                if edge_attr_tensor_temp.shape[0] == edge_index.shape[1] and edge_attr_tensor_temp.shape[1] == expected_edge_dim:
                     edge_attr = edge_attr_tensor_temp
            return node_features, edge_index, edge_attr
        except Exception as e: logging.error(f"ISS ({symbol}): Error building market graph: {e}"); return None, None, None

    def _apply_safeguards(self, calculated_sl: float, initial_sl_adjusted: float, position: dict, df: pd.DataFrame, symbol: str) -> float:
        entry_price = position.get("entry", 0.0); direction = position.get("direction", "NONE"); baseline_sl = initial_sl_adjusted
        if entry_price <= 0 or direction == "NONE": return baseline_sl
        if direction == "LONG": safe_sl_1 = max(calculated_sl, baseline_sl)
        elif direction == "SHORT": safe_sl_1 = min(calculated_sl, baseline_sl)
        else: safe_sl_1 = baseline_sl
        try:
            recent_df = df.tail(20); avg_atr = recent_df['ATR'].mean()
            if pd.isna(avg_atr) or avg_atr <= 0: max_sl_distance = abs(entry_price - baseline_sl)
            else: max_sl_distance = avg_atr * 3.0
        except Exception: max_sl_distance = abs(entry_price - baseline_sl)
        if direction == "LONG": floor_sl = entry_price - max_sl_distance; safe_sl_2 = max(safe_sl_1, floor_sl)
        elif direction == "SHORT": ceiling_sl = entry_price + max_sl_distance; safe_sl_2 = min(safe_sl_1, ceiling_sl)
        else: safe_sl_2 = safe_sl_1
        if (direction == "LONG" and safe_sl_2 >= entry_price) or (direction == "SHORT" and safe_sl_2 <= entry_price):
             final_sl = baseline_sl
        else: final_sl = safe_sl_2
        logging.debug(f"ISS Safeguards ({symbol}): Calc={calculated_sl:.4f}, Baseline={baseline_sl:.4f} -> Safe1={safe_sl_1:.4f}, Safe2={safe_sl_2:.4f} -> Final={final_sl:.4f}")
        return final_sl

# 7. Adaptive Position Sizing with Federated Learning (Cải tiến với Aggregation)
class FederatedAveraging:
    def __init__(self, input_dim=8): 

        self.input_dim = input_dim # Lưu lại input_dim
        # Sửa nn.Linear để sử dụng self.input_dim
        self.model = nn.Linear(self.input_dim, 1).cuda() if torch.cuda.is_available() else nn.Linear(self.input_dim, 1)
        logging.info(f"FederatedAveraging initialized with input_dim={self.input_dim}")
        self.device = next(self.model.parameters()).device # Lấy device từ model
        logging.info(f"FederatedAveraging model device: {self.device}")

    def train(self, federated_data: List[Dict[str, np.ndarray]], epochs: int, batch_size: int):
        if not federated_data:
            logging.error("FederatedAveraging train: No federated data provided.")
            return self.model

        # Chuyển model sang chế độ huấn luyện và đúng device
        self.model.train()
        self.model.to(self.device)

        global_weights = self.model.state_dict() # Lấy state_dict ban đầu

        for round_num in range(epochs):
            local_weights_list = [] # Đổi tên để rõ ràng hơn
            logging.info(f"--- Federated Training Round {round_num + 1}/{epochs} ---")

            # Lặp qua dữ liệu của từng "node"
            for node_idx, node_data in enumerate(federated_data):
                try:
                    # --- Kiểm tra dữ liệu node ---
                    if "features" not in node_data or "labels" not in node_data:
                        logging.warning(f"Node {node_idx}: Missing 'features' or 'labels'. Skipping.")
                        continue
                    if not isinstance(node_data["features"], np.ndarray) or not isinstance(node_data["labels"], np.ndarray):
                        logging.warning(f"Node {node_idx}: 'features' or 'labels' are not numpy arrays. Skipping.")
                        continue
                    if node_data["features"].shape[0] == 0 or node_data["labels"].shape[0] == 0:
                         logging.warning(f"Node {node_idx}: Empty features or labels. Skipping.")
                         continue
                    # <<< KIỂM TRA DIMENSION QUAN TRỌNG >>>
                    if node_data["features"].shape[-1] != self.input_dim:
                        logging.error(f"Node {node_idx}: Features dimension mismatch! Expected {self.input_dim}, got {node_data['features'].shape[-1]}. Skipping node.")
                        continue

                    # Tạo bản sao model và optimizer cho node hiện tại
                    local_model = copy.deepcopy(self.model) # Deep copy để không ảnh hưởng model gốc
                    local_model.to(self.device) # Đảm bảo model local trên đúng device
                    local_model.train() # Đặt local model ở chế độ train
                    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001) # Có thể dùng lr nhỏ hơn

                    # Chuẩn bị dữ liệu tensor cho node
                    X = torch.tensor(node_data["features"], dtype=torch.float32).to(self.device)
                    y = torch.tensor(node_data["labels"], dtype=torch.float32).to(self.device)

                    # Đảm bảo y là vector cột (N, 1)
                    if y.ndim == 1:
                        y = y.unsqueeze(1)
                    elif y.shape[-1] != 1:
                        logging.error(f"Node {node_idx}: Labels shape mismatch! Expected (*, 1), got {y.shape}. Skipping node.")
                        continue

                    # Huấn luyện local trên dữ liệu của node
                    node_total_loss = 0.0
                    num_batches = 0
                    for i in range(0, len(X), batch_size):
                        batch_X = X[i:i+batch_size]
                        batch_y = y[i:i+batch_size]

                        # Kiểm tra batch_y shape
                        if batch_y.ndim == 1:
                             batch_y = batch_y.unsqueeze(1)

                        optimizer.zero_grad()
                        outputs = local_model(batch_X) # Dự đoán
                        loss = F.mse_loss(outputs, batch_y) # Tính loss
                        loss.backward() # Backpropagation
                        optimizer.step() # Cập nhật trọng số local
                        node_total_loss += loss.item()
                        num_batches += 1

                    # Lưu trọng số đã huấn luyện của node này
                    if num_batches > 0:
                        avg_node_loss = node_total_loss / num_batches
                        logging.debug(f"  Node {node_idx}: Trained {num_batches} batches, Avg Loss: {avg_node_loss:.4f}")
                        local_weights_list.append(local_model.state_dict())
                    else:
                         logging.warning(f"  Node {node_idx}: No batches processed (data len: {len(X)}).")

                except Exception as node_e:
                    logging.error(f"Error processing Node {node_idx}: {node_e}", exc_info=True)
                    continue # Bỏ qua node này nếu có lỗi

            # --- Tổng hợp trọng số (Federated Averaging) ---
            if not local_weights_list:
                logging.warning(f"Round {round_num + 1}: No valid local models were trained in this round. Skipping weight aggregation.")
                continue # Bỏ qua vòng này nếu không có model nào được huấn luyện

            try:
                # Lấy trung bình các state_dict
                avg_weights = {}
                all_keys = global_weights.keys() # Các key trong state_dict
                for k in all_keys:
                    # Lấy tensor của key 'k' từ tất cả các model local, chuyển lên device chung và tính trung bình
                    key_tensors = [w[k].to(self.device) for w in local_weights_list]
                    avg_weights[k] = torch.stack(key_tensors).mean(dim=0)

                # Cập nhật trọng số global
                self.model.load_state_dict(avg_weights)
                logging.info(f"Round {round_num + 1}: Aggregated weights from {len(local_weights_list)} nodes.")

            except Exception as agg_e:
                 logging.error(f"Error during weight aggregation in round {round_num + 1}: {agg_e}", exc_info=True)
                 # Cân nhắc nên dừng hay tiếp tục với trọng số cũ
                 continue

        # Sau khi hoàn thành tất cả các vòng, chuyển model sang eval mode
        self.model.eval()
        logging.info("Federated training complete. Model set to eval mode.")
        return self.model

    @torch.no_grad() # Không cần tính gradient khi dự đoán
    def predict(self, features: np.ndarray) -> torch.Tensor:
        self.model.eval() # Đảm bảo đang ở chế độ eval
        try:
            # Chuyển numpy array thành tensor và đảm bảo đúng dtype
            features_tensor = torch.tensor(features, dtype=torch.float32)

            # Xử lý shape: đảm bảo có batch dimension
            if features_tensor.ndim == 1:
                # Nếu là vector 1D (1 mẫu), thêm batch dimension
                if features_tensor.shape[0] != self.input_dim:
                     logging.error(f"FederatedAveraging predict: Single sample dimension mismatch. Expected {self.input_dim}, got {features_tensor.shape[0]}.")
                     return torch.tensor([[0.0]], device=self.device) # Trả về dự đoán mặc định
                features_tensor = features_tensor.unsqueeze(0) # Shape: (1, input_dim)
            elif features_tensor.ndim == 2:
                # Nếu là ma trận 2D (nhiều mẫu), kiểm tra dimension cuối cùng
                if features_tensor.shape[1] != self.input_dim:
                    logging.error(f"FederatedAveraging predict: Batch dimension mismatch. Expected {self.input_dim}, got {features_tensor.shape[1]}.")
                    # Trả về tensor 0 với số dự đoán bằng số mẫu đầu vào
                    return torch.zeros((features_tensor.shape[0], 1), device=self.device)
            else:
                logging.error(f"FederatedAveraging predict: Invalid input tensor dimension: {features_tensor.ndim}. Expected 1 or 2.")
                return torch.tensor([[0.0]], device=self.device) # Dự đoán mặc định

            # Chuyển tensor đến device của model
            features_tensor = features_tensor.to(self.device)

            # Dự đoán
            prediction = self.model(features_tensor)
            return prediction # Trả về tensor dự đoán (shape [batch_size, 1])

        except Exception as e:
            logging.error(f"Error during FederatedAveraging prediction: {e}", exc_info=True)
            # Trả về tensor 0 với shape phù hợp nếu có lỗi
            batch_size = 1 if features.ndim == 1 else features.shape[0]
            return torch.zeros((batch_size, 1), device=self.device)

    def predict(self, features):
        features_tensor = torch.tensor(features, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(features, dtype=torch.float32)
        return self.model(features_tensor)

class SmartPositionSizer:
    def __init__(self, bot=None):
        if bot is None:
            raise ValueError("SmartPositionSizer requires a 'bot' instance for context (balance, exchange).")
        self.bot = bot
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"SmartPositionSizer initialized on device: {self.device}")
        try:
            # Khởi tạo với input_dim=8 (chỉ dùng RiskEncoder features)
            self.federated_model = FederatedAveraging(input_dim=8)
            self.risk_encoder = RiskFeatureEncoder()
        except Exception as e:
            logging.error(f"Failed to initialize sub-components in SmartPositionSizer: {e}", exc_info=True)
            raise
        self._load_federated_model()

    def _load_federated_model(self, model_path="federated_sizer_model.pth"):
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                self.federated_model.model.load_state_dict(state_dict)
                self.federated_model.model.to(self.device)
                self.federated_model.model.eval()
                logging.info(f"Successfully loaded pre-trained FederatedAveraging model from {model_path} to {self.device}")
            except Exception as e:
                logging.error(f"Failed to load FederatedAveraging model from {model_path}: {e}. Using untrained model.", exc_info=True)
                self.federated_model = FederatedAveraging(input_dim=8) # Khởi tạo lại với đúng dim
                self.federated_model.model.to(self.device).eval()
        else:
            logging.warning(f"FederatedAveraging model file not found at {model_path}. Using untrained model on {self.device}.")
            self.federated_model.model.to(self.device).eval()

    def calculate_size(self, strategy_params: dict, stop_loss_price: float, risk_factor: float) -> float:
        symbol = strategy_params.get("symbol", "N/A")
        entry_price = strategy_params.get("entry_price")
        if entry_price is None or entry_price <= 0: return 0.0
        if stop_loss_price is None or (stop_loss_price == entry_price) or \
           (strategy_params.get('direction') == 'LONG' and stop_loss_price >= entry_price) or \
           (strategy_params.get('direction') == 'SHORT' and stop_loss_price <= entry_price): return 0.0
        if not (0 <= risk_factor <= CONFIG.get("max_account_risk", 0.1)): return 0.0
        if risk_factor == 0: return 0.0
        try:
            #trained_fl_model = self.federated_model.model
            #trained_fl_model.eval()
            #market_conditions = { "volatility": strategy_params.get("volatility", 0.0), "atr": strategy_params.get("atr", 0.0), "rsi": strategy_params.get("rsi", 50.0), "adx": strategy_params.get("adx", 25.0), "ema_diff": strategy_params.get("ema_diff", 0.0), "volume": strategy_params.get("volume", 0.0), "sentiment": strategy_params.get("sentiment", 0.0) }
            #risk_features = self.risk_encoder.encode(risk_factor, market_conditions)
            #fl_input_features = risk_features
            #expected_dim = 8
            #if fl_input_features.shape != (expected_dim,):
            #    fl_input_features = np.pad(fl_input_features, (0, max(0, expected_dim - fl_input_features.shape[0])), mode='constant')[:expected_dim]
            #if not np.isfinite(fl_input_features).all():
             #    fl_input_features = np.nan_to_num(fl_input_features, nan=0.0, posinf=0.0, neginf=0.0)
            #features_tensor = torch.tensor(fl_input_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            #with torch.no_grad():
             #   raw_prediction = trained_fl_model(features_tensor).item()
            #raw_size_factor = 1.0 + np.tanh(raw_prediction) * 0.4
            #raw_size_factor = np.clip(raw_size_factor, 0.5, 1.5)
            #logging.debug(f"calculate_size ({symbol}): FL Raw Pred={raw_prediction:.4f} -> SizeFactor={raw_size_factor:.4f}")
            raw_size_factor = 1.0
            logging.debug(f"calculate_size ({symbol}): FL Prediction SKIPPED. Using neutral size factor: {raw_size_factor:.1f}")
            balance = self.bot.balance
            leverage = CONFIG.get("leverage", 1)
            min_qty = self._get_min_qty(symbol)
            max_qty = self._get_max_qty(symbol)
            final_size = self._apply_constraints_risk_based(
                raw_size_factor=raw_size_factor, entry_price=entry_price,
                stop_loss_price=stop_loss_price, balance=balance,
                risk_per_trade=risk_factor, leverage=leverage,
                min_qty=min_qty, max_qty=max_qty, symbol=symbol
            )
            if final_size < min_qty and final_size > 0: return 0.0
            elif final_size < 0: return 0.0
            logging.info(f"calculate_size ({symbol}): RiskFactor={risk_factor:.4f}, FLFactor={raw_size_factor:.3f} -> FinalSize={final_size:.8f}")
            return final_size
        except Exception as e:
            logging.error(f"Error during calculate_size for {symbol}: {e}", exc_info=True)
            try:
                 # ... fallback logic ...
                 sl_distance = abs(entry_price - stop_loss_price)
                 if sl_distance > 1e-9:
                      effective_risk = risk_factor if (0 < risk_factor <= CONFIG.get("max_account_risk", 0.1)) else 0.01
                      risk_amount = self.bot.balance * effective_risk * 0.5
                      fallback_size = risk_amount / sl_distance
                      min_qty = self._get_min_qty(symbol)
                      max_qty = self._get_max_qty(symbol)
                      final_fallback_size = np.clip(fallback_size, min_qty, max_qty)
                      # Làm tròn fallback size
                      if hasattr(self.bot, 'exchange') and self.bot.exchange:
                           fb_size_str = self.bot.exchange.amount_to_precision(symbol, final_fallback_size)
                           final_fallback_size = float(fb_size_str)
                           if final_fallback_size < min_qty: final_fallback_size = 0.0
                      else: # Làm tròn thủ công
                           if min_qty > 0: decimals = max(0, -int(np.floor(np.log10(min_qty))))
                           else: decimals = 8
                           final_fallback_size = round(final_fallback_size, decimals)
                           if final_fallback_size < min_qty: final_fallback_size = 0.0
                      logging.info(f"Fallback Size ({symbol}): {final_fallback_size:.8f}")
                      return max(0.0, final_fallback_size)
                 else: return 0.0
            except Exception as fallback_e:
                 logging.error(f"Error in fallback size calculation for {symbol}: {fallback_e}", exc_info=True)
                 return 0.0

    def _get_market_info(self, symbol: str) -> Optional[dict]:
        try:
            if hasattr(self.bot, 'exchange') and self.bot.exchange and \
               hasattr(self.bot.exchange, 'markets') and self.bot.exchange.markets:
                market = self.bot.exchange.market(symbol)
                return market
            else: return None
        except ccxt.BadSymbol: return None
        except Exception: return None

    def _get_min_qty(self, symbol: str) -> float:
        market_info = self._get_market_info(symbol)
        default_min_qty = 1e-8
        try:
            min_val = market_info['limits']['amount']['min']
            if min_val is not None: return max(float(min_val), default_min_qty)
            else: return default_min_qty
        except (KeyError, TypeError, AttributeError, ValueError, IndexError) :
            return default_min_qty

    def _get_max_qty(self, symbol: str) -> float:
        market_info = self._get_market_info(symbol)
        default_max_qty = 1e9
        try:
            max_val = market_info['limits']['amount']['max']
            if max_val is not None: return float(max_val)
            else: return default_max_qty
        except (KeyError, TypeError, AttributeError, ValueError, IndexError) :
            return default_max_qty

    def _apply_constraints_risk_based(self, raw_size_factor: float, entry_price: float, stop_loss_price: float, balance: float, risk_per_trade: float, leverage: int, min_qty: float, max_qty: float, symbol: str) -> float:
        sl_distance_per_unit = abs(entry_price - stop_loss_price)
        if sl_distance_per_unit < 1e-9: return 0.0
        risk_amount = balance * risk_per_trade
        if risk_amount <= 0: return 0.0
        base_position_size = risk_amount / sl_distance_per_unit
        adjusted_factor = np.clip(raw_size_factor, 0.5, 1.5)
        risk_adjusted_size = base_position_size * adjusted_factor
        max_position_value_allowed = balance * CONFIG.get("max_position_size", 0.5)
        if entry_price > 0: max_size_by_balance = max_position_value_allowed / entry_price
        else: max_size_by_balance = max_qty
        constrained_size = min(risk_adjusted_size, max_size_by_balance)
        final_size = np.clip(constrained_size, min_qty, max_qty)
        try:
            if hasattr(self.bot, 'exchange') and self.bot.exchange:
                final_size_str = self.bot.exchange.amount_to_precision(symbol, final_size)
                final_size_rounded = float(final_size_str)
                if final_size_rounded < min_qty and constrained_size >= min_qty: final_size = min_qty
                elif final_size_rounded == 0 and constrained_size > 0: final_size = 0.0
                else: final_size = final_size_rounded
            else:
                if min_qty > 0: decimals = max(0, -int(np.floor(np.log10(min_qty))))
                else: decimals = 8
                final_size = round(final_size, decimals)
                if final_size < min_qty and constrained_size >= min_qty: final_size = min_qty
                elif final_size < min_qty: final_size = 0.0
        except Exception as precision_e:
             final_size = np.clip(constrained_size, min_qty, max_qty)
             if final_size < min_qty: final_size = 0.0
        final_size = max(0.0, final_size)
        # logging.debug(f"_apply_constraints ({symbol}): ... Final(Rounded)={final_size:.8f} ...") # Log rút gọn
        return final_size


    # --- Hàm Huấn luyện FL - SỬA ĐỔI ĐỂ DÙNG NHIỀU FILE ---
    # <<< MODIFIED: Chấp nhận list file_paths >>>
    def _collect_federated_data_for_training(self,
                                             file_paths: List[str],
                                             timeframe='15m',
                                             num_nodes=5,
                                             min_data_length_per_symbol=200, # Đổi tên để rõ ràng hơn
                                             future_return_period=5):

        all_features = []
        all_labels = []
        total_samples = 0

        logging.info(f"Collecting federated data from: {file_paths}")

        # --- Lặp qua từng file trong danh sách ---
        for file_path in file_paths:
            symbol_name = file_path.split('/')[-1].split('_')[0] # Trích xuất tên symbol từ tên file (heuristic)
            logging.info(f"Processing symbol: {symbol_name} from {file_path}")
            try:
                # 1. Tải dữ liệu cho symbol hiện tại
                if not os.path.exists(file_path):
                    logging.warning(f"File not found: {file_path}. Skipping.")
                    continue
                try:
                    loaded_data = joblib.load(file_path)
                except:
                    import pickle
                    with open(file_path, 'rb') as f:
                        loaded_data = pickle.load(f)

                if not isinstance(loaded_data, dict) or timeframe not in loaded_data:
                    logging.warning(f"Invalid data format or timeframe '{timeframe}' not found in {file_path}. Skipping.")
                    continue
                df = loaded_data[timeframe]
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logging.warning(f"No valid DataFrame for timeframe '{timeframe}' in {file_path}. Skipping.")
                    continue
                logging.debug(f"Loaded {len(df)} rows for {symbol_name} ({timeframe}).")

                # 2. Chuẩn bị Features (logic giống như trước)
                feature_cols = ['volatility', 'ATR', 'RSI', 'ADX', 'EMA_50', 'EMA_200', 'volume']
                missing_cols = [col for col in feature_cols if col not in df.columns]
                if missing_cols:
                    logging.warning(f"Missing columns for {symbol_name}: {missing_cols}. Skipping.")
                    continue
                if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
                    df['ema_diff'] = (df['EMA_50'] - df['EMA_200']) / df['close'].replace(0, np.nan)
                    df['ema_diff'] = df['ema_diff'].fillna(0)
                    feature_cols.append('ema_diff')
                else: continue # Bỏ qua nếu không tính được ema_diff
                final_feature_cols = ['volatility', 'ATR', 'RSI', 'ADX', 'ema_diff', 'volume']
                df_features = df[final_feature_cols].copy()
                initial_len = len(df_features)
                df_features.dropna(inplace=True)
                dropped_rows = initial_len - len(df_features)
                # if dropped_rows > 0: logging.debug(f"Dropped {dropped_rows} rows (NaNs) for {symbol_name}.")
                if len(df_features) < min_data_length_per_symbol:
                    logging.warning(f"Not enough data ({len(df_features)}) for {symbol_name} after NaN handling. Skipping.")
                    continue
                simulated_risk = np.full(len(df_features), CONFIG.get("risk_per_trade", 0.02))
                simulated_sentiment = np.random.normal(0, 0.1, len(df_features))
                features_np = np.column_stack((
                    simulated_risk, df_features['volatility'].values, df_features['ATR'].values,
                    df_features['RSI'].values, df_features['ADX'].values, df_features['ema_diff'].values,
                    df_features['volume'].values, simulated_sentiment
                )).astype(np.float32)

                # 3. Tạo Target Labels (logic giống như trước)
                df_close_aligned = df.loc[df_features.index, 'close']
                future_returns = df_close_aligned.pct_change(future_return_period).shift(-future_return_period)
                def calculate_target_factor(ret):
                    if pd.isna(ret): return 1.0
                    if ret > 0.008: return 1.4
                    elif ret > 0.003: return 1.15
                    elif ret < -0.008: return 0.6
                    elif ret < -0.003: return 0.85
                    else: return 1.0
                target_labels_np = future_returns.apply(calculate_target_factor).values.reshape(-1, 1).astype(np.float32)

                # 4. Căn chỉnh và thêm vào danh sách tổng hợp
                if features_np.shape[0] != target_labels_np.shape[0]:
                    min_len = min(features_np.shape[0], target_labels_np.shape[0])
                    features_np = features_np[:min_len, :]
                    target_labels_np = target_labels_np[:min_len, :]
                    if min_len < min_data_length_per_symbol:
                        logging.warning(f"Not enough aligned data ({min_len}) for {symbol_name}. Skipping.")
                        continue
                logging.debug(f"Prepared {features_np.shape[0]} samples for {symbol_name}.")
                all_features.append(features_np)
                all_labels.append(target_labels_np)
                total_samples += features_np.shape[0]

            except Exception as e:
                logging.error(f"Error processing file {file_path} for symbol {symbol_name}: {e}", exc_info=True)
                continue # Bỏ qua file này nếu có lỗi

        # --- Kết hợp dữ liệu từ tất cả các symbol ---
        if not all_features:
            logging.error("No valid data collected from any provided file paths.")
            return []

        combined_features = np.concatenate(all_features, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        logging.info(f"Combined data from {len(all_features)} symbols. Total samples: {total_samples}")

        # Kiểm tra lại tổng số mẫu tối thiểu
        min_total_data = min_data_length_per_symbol # Hoặc đặt một ngưỡng tổng thể khác
        if total_samples < min_total_data:
             logging.error(f"Total combined samples ({total_samples}) is less than minimum required ({min_total_data}). Cannot proceed.")
             return []


        # --- 5. Định dạng dữ liệu cho FL (sử dụng dữ liệu tổng hợp) ---
        federated_data = []
        num_samples_per_node = total_samples // num_nodes
        if num_samples_per_node < 50:
             logging.warning(f"Samples per node ({num_samples_per_node}) is low based on total data. Using full combined dataset for each node.")
             for _ in range(num_nodes):
                  federated_data.append({"features": combined_features, "labels": combined_labels})
        else:
             # Chia dữ liệu tổng hợp (có thể cần shuffle trước khi chia)
             indices = np.arange(total_samples)
             np.random.shuffle(indices) # Xáo trộn dữ liệu trước khi chia
             shuffled_features = combined_features[indices]
             shuffled_labels = combined_labels[indices]

             for i in range(num_nodes):
                  start_idx = i * num_samples_per_node
                  end_idx = (i + 1) * num_samples_per_node if i < num_nodes - 1 else None
                  node_features = shuffled_features[start_idx:end_idx]
                  node_labels = shuffled_labels[start_idx:end_idx]
                  if len(node_features) > 0:
                       federated_data.append({"features": node_features, "labels": node_labels})
             logging.info(f"Split combined shuffled data into {len(federated_data)} nodes with ~{num_samples_per_node} samples each.")

        return federated_data


    def initial_train_federated_model(self, num_rounds=10, batch_size=32):
         """Huấn luyện ban đầu cho mô hình Federated Averaging sử dụng dữ liệu từ NHIỀU PKL."""
         # <<< MODIFIED: Truyền danh sách các file >>>
         data_files = ["BTC_USDT_data.pkl", "ETH_USDT_data.pkl"] # Danh sách các file cần dùng
         federated_training_data = self._collect_federated_data_for_training(file_paths=data_files)

         if not federated_training_data:
              logging.error("No combined training data loaded for FederatedAveraging model. Skipping training.")
              return
         try:
              self.federated_model.model.to(self.device)
              self.federated_model.model.train()
              self.federated_model.train(federated_training_data, epochs=num_rounds, batch_size=batch_size)
              logging.info("Completed initial training for FederatedAveraging model using combined PKL data.")
              torch.save(self.federated_model.model.state_dict(), "federated_sizer_model.pth")
              logging.info("Saved trained FederatedAveraging model.")
              self.federated_model.model.eval()
         except Exception as e:
              logging.error(f"Error during initial FederatedAveraging training: {e}", exc_info=True)
              self.federated_model.model.eval()

# 9. Trading Environment for DQN
class TradingEnv(Env):
    def __init__(self, bot, data: Dict[str, pd.DataFrame]):
        super().__init__()
        self.bot = bot
        self.data = data
        self.observation_space_dim = 28 # Số lượng features trong observation
        self.action_space = Discrete(3) # 3 hành động: giảm size (0.5x), giữ nguyên (1.0x), tăng size (1.5x)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32)

        # Khởi tạo step và max_steps an toàn
        self.current_step = 0
        self.max_steps = 0
        if "15m" in self.data and not self.data["15m"].empty:
            try:
                 self.start_step = self.bot.get_adaptive_warmup_period(data["15m"])
                 self.max_steps = len(data["15m"]) - 1 # Cho phép chạy đến cuối dữ liệu
                 # Đảm bảo max_steps không âm
                 self.max_steps = max(0, self.max_steps)
                 # Đảm bảo start_step không vượt quá max_steps
                 self.start_step = min(self.start_step, self.max_steps)

            except Exception as e:
                 logging.error(f"Error getting warmup/max_steps in TradingEnv init: {e}. Using defaults.")
                 self.start_step = CONFIG.get("hmm_warmup_min", 50)
                 self.max_steps = len(data.get("15m", pd.DataFrame())) - 1
                 self.max_steps = max(0, self.max_steps)
                 self.start_step = min(self.start_step, self.max_steps)
        else:
             logging.error("TradingEnv init: '15m' data is missing or empty. Cannot determine steps.")
             self.start_step = 0
             self.max_steps = 0

        self.current_step = self.start_step # Bắt đầu từ start_step
        self.scaler = None # Khởi tạo scaler
        self._initialize_scaler() # Fit scaler

        # --- Cache Market Info để tăng tốc độ ---
        self._market_info_cache = {}

    def _get_market_info(self, symbol: str) -> Optional[dict]:
        """Lấy thông tin thị trường từ cache hoặc exchange."""
        if symbol in self._market_info_cache:
             return self._market_info_cache[symbol]
        try:
             if hasattr(self.bot, 'exchange') and self.bot.exchange and self.bot.exchange.markets:
                  if symbol in self.bot.exchange.markets:
                       market = self.bot.exchange.market(symbol)
                       self._market_info_cache[symbol] = market # Lưu vào cache
                       return market
                  else:
                       logging.warning(f"Symbol {symbol} not found in loaded markets.")
                       return None
             else:
                  logging.warning(f"Cannot get market info for {symbol}: Bot or exchange not ready or markets not loaded.")
                  return None
        except ccxt.BadSymbol:
             logging.error(f"BadSymbol error getting market info for {symbol}.")
             return None
        except Exception as e:
             logging.error(f"Error getting market info for {symbol}: {e}", exc_info=True)
             return None

    def _initialize_scaler(self):
        """Khởi tạo và fit StandardScaler."""
        logging.info("TradingEnv: Initializing and fitting the observation scaler...")
        scaler_path = "trading_env_scaler.pkl" # Tên file scaler riêng cho env này

        # Cố gắng tải scaler đã lưu
        if os.path.exists(scaler_path):
             try:
                  self.scaler = joblib.load(scaler_path)
                  logging.info(f"TradingEnv: Loaded scaler from {scaler_path}")
                  # Kiểm tra xem scaler có đúng dimension không
                  if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != self.observation_space_dim:
                       logging.warning(f"Loaded scaler has incorrect dimension ({self.scaler.n_features_in_}), expected {self.observation_space_dim}. Refitting.")
                       self.scaler = None # Buộc fit lại
                  else:
                       return # Tải thành công, không cần fit lại
             except Exception as e:
                  logging.error(f"TradingEnv: Failed to load scaler from {scaler_path}: {e}. Refitting.", exc_info=True)
                  self.scaler = None

        # Nếu không tải được hoặc cần fit lại
        try:
            df_15m = self.data.get("15m")
            if df_15m is None or df_15m.empty:
                 raise ValueError("Cannot fit scaler: '15m' data is missing or empty.")

            # Dùng start_step đã tính toán
            fit_steps = min(len(df_15m) - self.start_step - 1, 5000) # Dùng self.start_step

            if fit_steps < 100:
                 logging.warning(f"TradingEnv: Insufficient data ({fit_steps} steps) to fit scaler properly. Using dummy scaler.")
                 self.scaler = StandardScaler()
                 self.scaler.fit(np.zeros((2, self.observation_space_dim)))
                 return

            sample_observations = []
            original_step = self.current_step
            start_fit_step = self.start_step
            end_fit_step = self.start_step + fit_steps

            logging.info(f"TradingEnv: Fitting scaler on steps {start_fit_step} to {end_fit_step-1}...")

            for i in range(start_fit_step, end_fit_step):
                self.current_step = i # Tạm thời đặt step
                try:
                    if i < len(df_15m):
                         # Gọi hàm lấy obs chưa chuẩn hóa
                         obs_unscaled = self._get_observation_unscaled()
                         if np.isfinite(obs_unscaled).all():
                              sample_observations.append(obs_unscaled)
                         else:
                              logging.warning(f"TradingEnv: NaN/Inf detected in unscaled observation at step {i} during scaler fitting. Skipping.")
                    else:
                        logging.warning(f"TradingEnv: Index {i} out of bounds during scaler fitting (len={len(df_15m)}).")
                        break

                except Exception as e:
                    logging.warning(f"TradingEnv: Error getting unscaled observation at step {i} during scaler fitting: {e}. Skipping.")
                    continue

            self.current_step = original_step # Khôi phục

            if not sample_observations:
                logging.error("TradingEnv: Failed to collect any valid sample observations for scaler fitting. Using dummy scaler.")
                self.scaler = StandardScaler()
                self.scaler.fit(np.zeros((2, self.observation_space_dim)))
                return

            sample_observations_np = np.array(sample_observations)

            if sample_observations_np.ndim != 2 or sample_observations_np.shape[1] != self.observation_space_dim:
                 logging.error(f"TradingEnv: Sample observations shape mismatch {sample_observations_np.shape}. Using dummy scaler.")
                 self.scaler = StandardScaler()
                 self.scaler.fit(np.zeros((2, self.observation_space_dim)))
                 return

            self.scaler = StandardScaler()
            self.scaler.fit(sample_observations_np)
            logging.info(f"TradingEnv: Scaler fitted successfully on {len(sample_observations)} samples.")
            # Lưu scaler
            try:
                 joblib.dump(self.scaler, scaler_path)
                 logging.info(f"TradingEnv: Saved scaler to {scaler_path}")
            except Exception as e:
                 logging.error(f"TradingEnv: Failed to save scaler: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"TradingEnv: Error during scaler initialization: {e}", exc_info=True)
            logging.warning("TradingEnv: Using dummy StandardScaler due to initialization error.")
            self.scaler = StandardScaler()
            self.scaler.fit(np.zeros((2, self.observation_space_dim)))


    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset môi trường về trạng thái ban đầu."""
        super().reset(seed=seed)
        logging.info("Resetting TradingEnv...")
        self.current_step = self.start_step # Reset về step bắt đầu
        self.bot.balance = CONFIG["initial_balance"] # Reset balance bot
        self.bot.trade_history = [] # Xóa lịch sử trade của bot
        # Reset exposure
        for symbol in CONFIG["symbols"]:
            self.bot.exposure_per_symbol[symbol] = 0.0
        # Reset các trạng thái nội bộ khác của env nếu có
        self._last_sentiment = 0.0

        obs = self._get_observation() # Lấy observation đầu tiên
        info = {}
        logging.info(f"TradingEnv reset complete. Current step: {self.current_step}, Balance: {self.bot.balance}")
        return obs, info

    # --- SỬA LỖI step VÀ _step_sync ---
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Thực hiện một bước đồng bộ, trả về 5 giá trị (API Gymnasium)."""
        logging.debug(f"TradingEnv Step {self.current_step}: Received action {action}")
        try:
            obs, reward, done, info = self._step_sync(action) # _step_sync trả về 4 giá trị
            terminated = done # Map done sang terminated
            truncated = False # Giả sử không có truncation riêng

            # Kiểm tra shape obs cuối cùng
            if obs.shape != self.observation_space.shape:
                 logging.error(f"TradingEnv Step {self.current_step}: Observation shape mismatch after step: {obs.shape}. Returning zeros.")
                 obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

            logging.debug(f"TradingEnv Step {self.current_step}: Returning state, reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
            # Trả về 5 giá trị
            return obs, reward, terminated, truncated, info
        except Exception as e:
            logging.critical(f"TradingEnv.step: Unhandled critical error: {e}", exc_info=True)
            obs = self._get_observation() # Cố gắng lấy obs
            if obs.shape != self.observation_space.shape:
                 obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            reward = 0.0
            terminated = True # Kết thúc episode nếu lỗi nghiêm trọng
            truncated = False
            info = {"critical_error": str(e)}
            return obs, reward, terminated, truncated, info

    def _step_sync(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Logic cốt lõi của bước đồng bộ cho TradingEnv."""
        info = {'trade_executed': False}
        pnl = 0.0
        reward = 0.0
        symbol = CONFIG["symbols"][0]
        done = False

        try:
            # Kiểm tra step hợp lệ
            if self.current_step > self.max_steps:
                 logging.warning(f"TradingEnv _step_sync: current_step {self.current_step} exceeds max_steps {self.max_steps}. Ending episode.")
                 obs = self._get_observation() # Lấy obs cuối
                 return obs, 0.0, True, {'reason': 'max_steps_exceeded'}

            # --- 1. Lấy tín hiệu cơ sở ---
            # Hàm này cần tồn tại trong self.bot và hoạt động đúng
            signal = self.bot.generate_signal_sync(symbol, {tf: df.iloc[:self.current_step+1] for tf, df in self.data.items()})

            if signal:
                # --- 2. Tính TP/SL ---
                try:
                    # Đảm bảo direction hợp lệ
                    if signal["direction"] not in ["LONG", "SHORT"]:
                         raise ValueError(f"Invalid signal direction: {signal['direction']}")

                    tp_levels, sl_price = self.bot.dynamic_stop_management(
                        signal["entry_price"], signal["atr"], signal["adx"],
                        signal["direction"], symbol
                    )
                except Exception as sl_e:
                    logging.error(f"TradingEnv _step_sync: Error calculating TP/SL for {symbol} at step {self.current_step}: {sl_e}", exc_info=True)
                    signal = None # Không thể tiếp tục

                if signal:
                    # --- 3. Tính size cơ sở (TRUYỀN sl_price) ---
                    try:
                        last_row = self.data["15m"].iloc[self.current_step]
                        current_timestamp = pd.to_datetime(last_row.name)

                        # --- LẤY SENTIMENT THỰC TẾ ---
                        active_sentiment_score = 0.0
                        if hasattr(self.bot, 'sentiment_analyzer') and self.bot.sentiment_analyzer:
                            try:
                                sentiment_details = self.bot.sentiment_analyzer.get_detailed_active_sentiment(current_timestamp)
                                active_sentiment_score = sentiment_details.get("total_score_adj_confidence", 0.0)
                            except Exception as sent_e: logging.error(f"Error getting sentiment in TradingEnv _step_sync: {sent_e}")

                        strategy_params = {
                            "symbol": symbol, "entry_price": signal["entry_price"],
                            "atr": signal["atr"], "risk": CONFIG["risk_per_trade"],
                            "volatility": last_row.get("volatility", 0.0),
                            "rsi": last_row.get("RSI", 50.0),
                            "adx": signal["adx"],
                            "ema_diff": last_row.get("EMA_50", last_row.get("close", 0)) - last_row.get("EMA_200", last_row.get("close", 0)),
                            "volume": last_row.get("volume", 0.0),
                            "sentiment": active_sentiment_score,
                            "order_book": {"bids": [], "asks": []} # Giả lập order book rỗng
                        }
                        # Gọi qua bot.position_sizer
                        base_position_size = self.bot.position_sizer.calculate_size(
                            strategy_params,
                            stop_loss_price=sl_price
                        )
                    except Exception as size_e:
                         logging.error(f"TradingEnv _step_sync: Error calculating base position size for {symbol} at step {self.current_step}: {size_e}", exc_info=True)
                         signal = None # Không thể tiếp tục

                    if signal:
                        # --- 4. Điều chỉnh size theo action DQN ---
                        adjusted_position_size = self._adjust_position_size(action, base_position_size, symbol) # Truyền symbol

                        # --- 5. Kiểm tra exposure ---
                        if adjusted_position_size > 0 and self.bot.check_position_exposure(adjusted_position_size, signal["entry_price"], symbol):
                            # --- 6. Mô phỏng trade ---
                            try:
                                exit_price, pnl = self.bot._simulate_trade_sync(
                                    self.data, self.current_step, signal, tp_levels, sl_price, adjusted_position_size
                                )
                                info['trade_executed'] = True
                                info['pnl'] = pnl

                                self.bot.exposure_per_symbol[symbol] += (adjusted_position_size * signal["entry_price"]) / max(self.bot.balance, 1.0)
                                # Cập nhật balance và history khi lệnh đóng (PnL != 0 ?)
                                self.bot.balance += pnl # Cập nhật balance
                                self.bot.equity_peak = max(self.bot.equity_peak, self.bot.balance)
                                self.bot.trade_history.append({ # Thêm vào history
                                    "symbol": symbol, "position_size": adjusted_position_size,
                                    "entry_price": signal["entry_price"], "exit_price": exit_price,
                                    "pnl": pnl, "action_dqn": action, # Lưu action DQN
                                    "timestamp": signal["timestamp"]
                                })
                                # Giảm exposure khi lệnh đóng
                                self.bot.exposure_per_symbol[symbol] -= (adjusted_position_size * signal["entry_price"]) / max(self.bot.balance, 1.0)
                                # Giữ exposure không âm
                                self.bot.exposure_per_symbol[symbol] = max(0, self.bot.exposure_per_symbol[symbol])


                            except Exception as sim_e:
                                 logging.error(f"TradingEnv _step_sync: Error simulating trade for {symbol} at step {self.current_step}: {sim_e}", exc_info=True)
                                 pnl = 0.0 # PnL = 0 nếu mô phỏng lỗi
                                 info['trade_error'] = str(sim_e)
                        else:
                            if adjusted_position_size <= 0:
                                 logging.debug(f"TradingEnv _step_sync: Trade for {symbol} skipped due to zero/negative size ({adjusted_position_size}).")
                            else:
                                 logging.warning(f"TradingEnv _step_sync: Trade for {symbol} skipped due to exposure limit.")
                            pnl = 0.0 # Không trade -> pnl = 0
            else: # Không có tín hiệu
                 pnl = 0.0

            if info.get('trade_executed', False):
                 reward = self._calculate_reward()
            else:
                 reward = 0.0 # Không có trade -> reward = 0 (hoặc phạt nhẹ?)
            info['reward'] = reward

            # --- Cập nhật trạng thái ---
            self.current_step += 1
            done = self.current_step > self.max_steps or self.bot.balance <= 0 # Dùng > thay vì >= để bước cuối cùng vẫn tính reward
            if done and self.current_step > self.max_steps: info['reason'] = 'max_steps_reached'
            if done and self.bot.balance <= 0: info['reason'] = 'balance_depleted'

            obs = self._get_observation() # Lấy observation mới
            return obs, reward, done, info # Trả về 4 giá trị

        except Exception as e:
            logging.critical(f"Critical error in TradingEnv _step_sync at step {self.current_step}: {e}", exc_info=True)
            obs = self._get_observation()
            if obs.shape != self.observation_space.shape:
                 obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return obs, 0.0, True, {"critical_error": str(e)} # Done=True nếu lỗi nghiêm trọng

    def _adjust_position_size(self, action: int, base_position_size: float, symbol: str) -> float:
        """Điều chỉnh kích thước vị thế dựa trên action của DQN."""
        multipliers = {0: 0.5, 1: 1.0, 2: 1.5}
        action_value = int(action) if isinstance(action, (int, float, np.number)) else 1 # Chuyển sang int, mặc định 1
        logging.debug(f"TradingEnv Step {self.current_step}: Raw DQN action = {action_value}")

        if action_value not in multipliers:
             logging.warning(f"Invalid action value {action_value} in _adjust_position_size. Using multiplier 1.0.")
             multiplier = 1.0
        else:
             multiplier = multipliers[action_value]

        adjusted_size = base_position_size * multiplier

        # Lấy min/max quantity từ helper của sizer
        min_qty = self.bot.position_sizer._get_min_qty(symbol)
        max_qty = self.bot.position_sizer._get_max_qty(symbol)

        # Giới hạn bởi min/max của sàn
        final_adjusted_size = np.clip(adjusted_size, min_qty, max_qty)

        # --- Làm tròn số lượng (QUAN TRỌNG) ---
        try:
            if hasattr(self.bot, 'exchange') and self.bot.exchange:
                final_size_precision = self.bot.exchange.amount_to_precision(symbol, final_adjusted_size)
                final_size_float = float(final_size_precision)
                # Đảm bảo không làm tròn xuống 0 nếu size > 0 và nhỏ hơn min_qty sau làm tròn
                if adjusted_size > 0 and final_size_float < min_qty:
                     final_adjusted_size = min_qty # Nếu làm tròn xuống dưới min, đặt lại là min
                else:
                     final_adjusted_size = max(final_size_float, 0.0) # Đảm bảo không âm
            else:
                decimals = 8 # Fallback
                final_adjusted_size = round(final_adjusted_size, decimals)
                final_adjusted_size = max(final_adjusted_size, min_qty) # Vẫn áp dụng min_qty
        except Exception as precision_e:
             logging.error(f"Error applying amount precision in _adjust_position_size for {symbol}: {precision_e}. Using unrounded value (clipped).")
             final_adjusted_size = np.clip(adjusted_size, min_qty, max_qty) # Giữ giá trị đã clip

        # Đảm bảo size cuối cùng không âm
        final_adjusted_size = max(0.0, final_adjusted_size)

        logging.debug(f"Adjusted Size (DQN): Base={base_position_size:.8f}, Action={action_value}, Multiplier={multiplier}, Adjusted={adjusted_size:.8f}, Final={final_adjusted_size:.8f}")
        return final_adjusted_size

    def _get_observation(self) -> np.ndarray:
        try:
            obs_unscaled = self._get_observation_unscaled()

            if self.scaler is None:
                logging.error("TradingEnv: Scaler is not initialized! Cannot scale observation. Returning zeros.")
                if obs_unscaled.shape != self.observation_space.shape:
                    obs_unscaled = np.zeros(self.observation_space.shape, dtype=np.float32)
                # <<< THÊM XỬ LÝ NAN/INF CHO UNCALED TRƯỚC KHI TRẢ VỀ >>>
                if not np.isfinite(obs_unscaled).all():
                    logging.warning(f"TradingEnv: NaN/Inf in unscaled obs (scaler not ready) at step {self.current_step}. Replacing with 0.")
                    obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
                return obs_unscaled.astype(np.float32)

            if obs_unscaled.shape != self.observation_space.shape:
                logging.error(f"TradingEnv _get_observation: Unscaled obs shape mismatch {obs_unscaled.shape}. Returning zeros.")
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            # <<< THÊM XỬ LÝ NAN/INF TRƯỚC KHI SCALE >>>
            if not np.isfinite(obs_unscaled).all():
                logging.warning(f"TradingEnv: NaN/Inf found in unscaled observation before scaling at step {self.current_step}. Replacing with 0.")
                obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0)

            obs_reshaped = obs_unscaled.reshape(1, -1)
            obs_scaled = self.scaler.transform(obs_reshaped)

            # <<< THÊM XỬ LÝ NAN/INF SAU KHI SCALE >>>
            if not np.isfinite(obs_scaled).all():
                logging.error(f"TradingEnv _get_observation: NaN/Inf detected after scaling at step {self.current_step}. Replacing with zeros.")
                obs_scaled = np.nan_to_num(obs_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            return obs_scaled.flatten().astype(np.float32) # Trả về đúng shape và dtype

        except Exception as e:
            logging.critical(f"TradingEnv: Critical error in _get_observation at step {self.current_step}: {e}", exc_info=True)
            # <<< TRẢ VỀ ZEROS AN TOÀN >>>
            return np.zeros(self.observation_space.shape, dtype=np.float32)


    def _get_observation_unscaled(self) -> np.ndarray:

        current_step_safe = min(self.current_step, len(self.data["15m"]) - 1)
        if current_step_safe < 0 : current_step_safe = 0

        if "15m" not in self.data or self.data["15m"].empty or current_step_safe >= len(self.data["15m"]):
            logging.error(f"TradingEnv _get_observation_unscaled: Cannot get observation at step {current_step_safe}.")
            return np.zeros(self.observation_space_dim, dtype=np.float32)

        last_row = self.data["15m"].iloc[current_step_safe]
        timestamp_for_obs = pd.to_datetime(last_row.name)
        active_sentiment_for_obs = 0.0
        if hasattr(self.bot, 'sentiment_analyzer') and self.bot.sentiment_analyzer:
            try:
                sentiment_details = self.bot.sentiment_analyzer.get_detailed_active_sentiment(timestamp_for_obs)
                active_sentiment_for_obs = sentiment_details.get("total_score_adj_confidence", 0.0)
            except Exception as sent_e: logging.error(f"Error getting sentiment for DQN observation: {sent_e}")
        obs_list = [
            last_row.get("close", 0.0),
            last_row.get("RSI", 50.0),
            last_row.get("EMA_diff", 0.0),
            last_row.get("volume_anomaly", 0.0),
            last_row.get("EMA_20", last_row.get("close", 0.0)),
            last_row.get("ATR", 0.0),
            last_row.get("volatility", 0.0),
            last_row.get("hybrid_regime", 1),
            last_row.get("VWAP", last_row.get("close", 0.0)),
            last_row.get("VAH", last_row.get("high", 0.0)),
            float(last_row.name.hour) if isinstance(last_row.name, pd.Timestamp) else 0.0, # Lấy từ index
            float(last_row.name.dayofweek) if isinstance(last_row.name, pd.Timestamp) else 0.0,
            last_row.get("VWAP_ADX_interaction", 0.0),
            last_row.get("BB_EMA_sync", 0.0),   
            last_row.get("swing_high", 0.0), # Đảm bảo cột này tồn tại từ detect_swing_points
            last_row.get("swing_low", 0.0),  # Đảm bảo cột này tồn tại
            last_row.get("MACD", 0.0),
            last_row.get("MACD_signal", 0.0),
            last_row.get("ADX", 25.0),
            last_row.get("OBV", 0.0),
            last_row.get("BB_width", 0.0),
            last_row.get("log_volume", 0.0),
            last_row.get("divergence", 0.0), # Đảm bảo cột này tồn tại từ detect_divergence
            last_row.get("rejection", 0.0), # Đảm bảo cột này tồn tại
            self.balance, # Balance hiện tại của bot
            active_sentiment_for_obs,
            last_row.get("momentum", 0.0),
            last_row.get("order_imbalance", 0.0)
        ]
        obs_unscaled = np.array(obs_list, dtype=np.float32)

        # Xử lý NaN/Inf cuối cùng
        if not np.isfinite(obs_unscaled).all():
             logging.warning(f"NaN/Inf detected in unscaled observation values at step {current_step_safe}. Replacing with 0.")
             obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0)

        if obs_unscaled.shape != (self.observation_space_dim,):
            logging.error(f"Final unscaled observation shape mismatch at step {current_step_safe}: got {obs_unscaled.shape}. Padding/Truncating.")
            obs_unscaled = np.pad(obs_unscaled, (0, self.observation_space_dim - obs_unscaled.shape[0]), mode='constant')[:self.observation_space_dim]

        return obs_unscaled
    
    def _calculate_reward_volatility(self, volatility: float, symbol_base: str) -> float: # Đổi tên symbol để tránh nhầm lẫn
        W_VOLATILITY = 0.35  # Trọng số volatility trong tổng reward

        # Làm sạch tên symbol để so sánh (chỉ lấy phần base)
        clean_symbol = symbol_base.upper().split('/')[0] # Ví dụ: "BTC/USDT" -> "BTC"

        # Ngưỡng volatility dựa trên symbol (BTC hoặc ETH)
        if clean_symbol == "BTC":
            vol_mean = 0.003
            vol_std = 0.002
        elif clean_symbol == "ETH":
            vol_mean = 0.004
            vol_std = 0.0025
        else:
            logging.warning(f"_calculate_reward_volatility: Symbol '{symbol_base}' not recognized for specific thresholds. Using default.")
            vol_mean = 0.003
            vol_std = 0.002

        # Chuẩn hóa volatility
        if pd.isna(volatility): volatility = 0.0 # Xử lý NaN trước khi tính
        vol_normalized = (volatility - vol_mean) / vol_std if vol_std > 1e-9 else 0.0 # Tránh chia cho 0

        # Hàm thưởng Gaussian-like
        reward_volatility = np.exp(-0.5 * vol_normalized**2) * W_VOLATILITY

        # Phạt bổ sung khi volatility quá cao (> 2 std)
        if vol_normalized > 2.0:
            penalty = -0.2 * W_VOLATILITY * (vol_normalized - 2.0)
            # Đảm bảo reward không âm hơn -W_VOLATILITY sau khi cộng penalty
            reward_volatility = max(reward_volatility + penalty, -W_VOLATILITY)

        # Giới hạn reward trong khoảng [-W_VOLATILITY, W_VOLATILITY]
        reward_volatility = np.clip(reward_volatility, -W_VOLATILITY, W_VOLATILITY)

        # Logging để debug
        logging.debug(f"Reward Volatility ({symbol_base}): vol={volatility:.6f}, norm={vol_normalized:.2f}, reward={reward_volatility:.3f}")

        return reward_volatility

    def _calculate_reward(self) -> float:
        if not self.bot.trade_history:
            # logging.debug("Calculate reward: No trade history yet.") # Có thể bỏ log này nếu quá nhiều
            return 0.0

        try:
            last_trade = self.bot.trade_history[-1]
            pnl = last_trade.get("pnl", 0.0)
            symbol = last_trade.get("symbol")
            position_size = last_trade.get("position_size", 0.0)
            entry_price = last_trade.get("entry_price", 0.0)
            direction = last_trade.get("direction")
            # Lấy action DQN từ history, mặc định là 1 (giữ nguyên size) nếu không có
            action_value = last_trade.get("action_dqn", 1)
            timestamp = last_trade.get("timestamp")

            if symbol is None or timestamp is None or direction is None:
                logging.error(f"TradingEnv _calculate_reward: Missing required keys in last_trade: {last_trade}")
                return 0.0

            # --- Tìm dòng dữ liệu tương ứng với thời điểm trade ---
            try:
                df_15m = self.data.get("15m") # Lấy df 15m từ dữ liệu của Env
                if df_15m is None or df_15m.empty:
                    raise KeyError("15m data is missing or empty in TradingEnv.")

                # Cần index là DatetimeIndex để get_indexer hoạt động đúng
                if not isinstance(df_15m.index, pd.DatetimeIndex):
                    # Thử chuyển đổi nếu chưa phải
                    df_15m = df_15m.reset_index()
                    df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'], errors='coerce')
                    df_15m = df_15m.dropna(subset=['timestamp']).set_index('timestamp')
                    if not isinstance(df_15m.index, pd.DatetimeIndex):
                        raise TypeError("Failed to convert index to DatetimeIndex for reward calculation.")
                    self.data["15m"] = df_15m # Cập nhật lại data trong env nếu cần

                # Tìm vị trí của timestamp trade (hoặc vị trí ngay trước đó)
                reward_step_index = df_15m.index.get_loc(timestamp, method='ffill') # Tìm vị trí gần nhất trước hoặc bằng

            except KeyError as e:
                logging.error(f"TradingEnv _calculate_reward: KeyError {e} accessing '15m' data or specific timestamp {timestamp}.")
                return 0.0
            except Exception as idx_e:
                logging.error(f"Error finding reward step index for timestamp {timestamp}: {idx_e}. Using current_step - 1 as fallback.")
                # Fallback sử dụng step trước đó
                reward_step_index = self.current_step - 1
                reward_step_index = max(0, min(reward_step_index, len(df_15m)-1)) if df_15m is not None and not df_15m.empty else -1

            if reward_step_index < 0:
                logging.error("TradingEnv _calculate_reward: Cannot determine valid reward step index.")
                return 0.0

            last_row = df_15m.iloc[reward_step_index]

            # --- Tính toán các thành phần reward ---
            # Định nghĩa trọng số (Đảm bảo tổng bằng 1 nếu muốn diễn giải dễ dàng)
            W_PNL = 0.35
            W_VOLATILITY = 0.35
            W_TIMING = 0.20 # Giảm trọng số timing một chút
            W_RISK_ADJUSTMENT = 0.10 # Thêm trọng số cho việc điều chỉnh rủi ro/exposure
            # W_LIQUIDITY = 0.0 (Tạm bỏ qua liquidity vì khó lấy chính xác)
            # assert abs(W_PNL + W_VOLATILITY + W_TIMING + W_RISK_ADJUSTMENT - 1.0) < 1e-6

            # 1. PnL Reward (Normalized)
            # Scale PnL dựa trên rủi ro tối đa của trade hoặc % balance nhỏ
            max_potential_loss = abs(entry_price - last_trade.get('stop_loss', entry_price)) * position_size if last_trade.get('stop_loss') else abs(self.bot.balance * 0.01)
            scale_factor = max(max_potential_loss, abs(self.bot.balance * 0.005), 1.0) # Scale dựa trên rủi ro tiềm năng hoặc % nhỏ balance
            normalized_pnl = np.tanh(pnl / scale_factor)
            reward_pnl = normalized_pnl * W_PNL

            # 2. Volatility Reward (Gọi hàm helper)
            volatility = last_row.get("volatility", 0.0)
            reward_volatility = self._calculate_reward_volatility(volatility, symbol) # Đã bao gồm W_VOLATILITY bên trong

            # 3. Timing Reward
            reward_timing = 0.0
            momentum = last_row.get("momentum", 0.0)
            order_imbalance = last_row.get("order_imbalance", 0.0)
            if pd.isna(momentum): momentum = 0.0
            if pd.isna(order_imbalance): order_imbalance = 0.0

            consensus_sign = np.sign(momentum) if abs(momentum) > 1e-6 and np.sign(momentum) == np.sign(order_imbalance) else 0
            if consensus_sign != 0:
                # Ánh xạ action DQN (0: giảm, 1: giữ, 2: tăng) sang hướng (-1, 0, 1)
                dqn_direction_map = {0: -1, 1: 0, 2: 1}
                intended_direction_sign = dqn_direction_map.get(int(action_value), 0)

                # Thưởng nếu action của DQN trùng hướng momentum/imbalance
                # Phạt nếu action ngược hướng
                if intended_direction_sign == consensus_sign:
                    magnitude = min(abs(momentum) * 5 + abs(order_imbalance) * 0.2, 0.5) # Giảm hệ số một chút
                    reward_timing = magnitude * W_TIMING
                elif intended_direction_sign == -consensus_sign:
                    magnitude = min(abs(momentum) * 2 + abs(order_imbalance) * 0.1, 0.3) # Giảm hệ số một chút
                    reward_timing = -magnitude * W_TIMING
                # Không thưởng/phạt nếu DQN chọn giữ nguyên (action=1, sign=0)

            # 4. Risk Adjustment Reward (Thưởng/Phạt dựa trên action DQN)
            reward_risk_adj = 0.0
            # Thưởng nếu giảm size khi PnL âm, phạt nếu tăng size khi PnL âm
            if pnl < 0:
                if action_value == 0: # Giảm size
                    reward_risk_adj = 0.1 * W_RISK_ADJUSTMENT # Thưởng nhỏ
                elif action_value == 2: # Tăng size
                    reward_risk_adj = -0.3 * W_RISK_ADJUSTMENT # Phạt
            # Thưởng nếu tăng size khi PnL dương, phạt nhẹ nếu giảm size khi PnL dương
            elif pnl > 0:
                if action_value == 2: # Tăng size
                    reward_risk_adj = 0.2 * W_RISK_ADJUSTMENT # Thưởng
                elif action_value == 0: # Giảm size
                    reward_risk_adj = -0.05 * W_RISK_ADJUSTMENT # Phạt nhẹ

            # --- Tổng hợp Reward ---
            total_reward = reward_pnl + reward_volatility + reward_timing + reward_risk_adj
            total_reward = np.clip(total_reward, -1.0, 1.0) # Giới hạn trong khoảng [-1, 1] cho ổn định

            # Kiểm tra NaN/Inf cuối cùng
            if not np.isfinite(total_reward):
                logging.error(f"NaN/Inf in final reward calculation: PNL={reward_pnl:.3f}, Vol={reward_volatility:.3f}, Time={reward_timing:.3f}, RiskAdj={reward_risk_adj:.3f}")
                return 0.0

            logging.debug(f"Reward Calc (Step {reward_step_index}, Trade: {symbol} {direction} PNL={pnl:.4f}): R_pnl={reward_pnl:.3f}, R_vol={reward_volatility:.3f}, R_time={reward_timing:.3f}, R_risk={reward_risk_adj:.3f} => Total={total_reward:.4f}")
            return total_reward

        except Exception as e:
            logging.error(f"TradingEnv _calculate_reward: Unexpected error: {e}", exc_info=True)
            return 0.0

# Main Trading Bot Class
class EnhancedTradingBot:
    XGB_STRONG_PROB = 0.75
    XGB_MEDIUM_PROB = 0.60
    XGB_WEAK_PROB = 0.55
    XGB_SHORT_WEAK_THRESH_P0=0.45
    ADAPTIVE_WEIGHTS = {
        'high_vol': {'sac': 0.6, 'xgb': 0.4},
        'trending': {'sac': 0.4, 'xgb': 0.6},
        'default': {'sac': 0.5, 'xgb': 0.5}
    }
    # === LOCK ACQUISITION ORDER (Quan trọng!) ===
    # 1. cb_lock (Circuit Breaker - Top Level)
    # 2. balance_lock (Balance, Exposure, Equity Peak)
    # 3. positions_lock (Open Positions, Trailing Stops)
    # 4. trade_history_lock (Trade History Log)
    # 5. data_locks[symbol] (Symbol-Specific Data OHLCV, OB, Indicators)
    # 6. misc_state_lock (Other less critical shared state)
    # Khi cần nhiều lock, PHẢI acquire theo thứ tự này để tránh DEADLOCK.

    def __init__(self, mode: str = "backtest"):
        assert mode in ["live", "backtest"], "Mode must be 'live' or 'backtest'"
        self.mode = mode
        logging.info(f"Initializing Bot in {self.mode.upper()} mode.")
        self.exchange = None
        self.balance_lock = asyncio.Lock()  # Protects self.balance and self.exposure_per_symbol
        self.positions_lock = asyncio.Lock()  # Protects self.open_positions
        self.data_locks = {symbol: asyncio.Lock() for symbol in CONFIG["symbols"]}  # Per-symbol locks for self.data
        self.cb_lock = asyncio.Lock()  # Protects self.circuit_breaker_triggered
        self.trade_history_lock = asyncio.Lock()
        self.misc_state_lock = asyncio.Lock()
        self.embeddings_lock = asyncio.Lock()
        self.equity_peak = CONFIG.get("initial_balance", 10000) 
        self.risk_multiplier = 1.0 
        self.data_feeds = {}
        self.indicator_cache = {}
        self.current_embeddings = {}
        self.realtime_monitor: Optional[RealtimeMonitor] = None
        self.monitor_callbacks = {} 
        self.last_processed_candle_ts = {} 
        self.open_positions = {} 
        self._initialize_components()
        self._validate_config()
        self.mlp_action_fallback = 0.0
        self.script_dir = Path(__file__).resolve().parent
        self.embeddings_cache_dir = self.script_dir / CONFIG.get("embeddings_cache_dir", "embeddings_cache")
        self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)
        self.mlp_action_analyzer = None
        self.mlp_action_scaler = None
        self.decision_model = None
        self.hybrid_model = None
        self.combined_agent = None
        if _new_models_available:
            try:
                DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                # Tạo config riêng cho model từ CONFIG chính
                decision_config = {k.replace('decision_', ''): v for k, v in CONFIG.items() if k.startswith('decision_')}
                decision_config['input_dim'] = CONFIG['input_dim'] # Thêm input_dim gốc
                decision_config['sequence_length'] = CONFIG['DECISION_SEQ_LEN']
                decision_config['dropout_rate'] = decision_config.get('dropout_rate', 0.2) # Default dropout

                hybrid_config = {k.replace('hybrid_', ''): v for k, v in CONFIG.items() if k.startswith('hybrid_')}
                hybrid_config['input_dim'] = CONFIG['input_dim']
                hybrid_config['primary_tf'] = CONFIG['primary_tf']
                hybrid_config['required_tfs'] = CONFIG['required_tfs_hybrid']
                hybrid_config['sequence_length'] = CONFIG['DECISION_SEQ_LEN'] # Có thể cần điều chỉnh? Module gốc không dùng seq len trực tiếp.
                hybrid_config['dropout_rate'] = hybrid_config.get('dropout_rate', 0.2)

                # Tạo example input cho Hybrid Model
                example_hybrid_input_dict = {
                    tf: torch.randn(1, CONFIG['HYBRID_SEQ_LEN_MAP'][tf], CONFIG['input_dim'], device=DEVICE)
                    for tf in CONFIG['required_tfs_hybrid'] if tf in CONFIG['HYBRID_SEQ_LEN_MAP'] # Check key tồn tại
                }
                # Kiểm tra xem có đủ input cho hybrid không
                if len(example_hybrid_input_dict) != len(CONFIG['required_tfs_hybrid']):
                     logging.error("Missing sequence length in HYBRID_SEQ_LEN_MAP for required timeframes. Cannot initialize Hybrid model properly.")
                     raise ValueError("Missing hybrid sequence lengths in config")

                self.decision_model = CryptoDecisionSupportModel(decision_config, device=DEVICE).to(DEVICE)
                self.hybrid_model = CryptoTradingModel(hybrid_config, example_hybrid_input_dict, device=DEVICE).to(DEVICE)
                self.decision_model.eval()
                self.hybrid_model.eval()
                logging.info("Initialized new Decision and Hybrid models.")

                # Khởi tạo CombinedTradingAgent
                self.combined_agent = CombinedTradingAgent(
                    decision_model=self.decision_model,
                    hybrid_model=self.hybrid_model,
                    bot_config=CONFIG,
                    entry_optimizer_model=None, # Sẽ được gán sau khi init SAC
                    xgboost_models=self.models,
                    xgboost_scalers=self.scalers,
                    get_sac_state_func=self._get_sac_state_unscaled, # Truyền hàm helper
                    scale_sac_state_func=self._scale_sac_state,      # Truyền hàm helper
                    get_dynamic_adx_func=self._get_dynamic_adx_threshold,
                    temporal_features=self.temporal_features,
                )
                logging.info("Initialized CombinedTradingAgent.")

            except Exception as model_init_e:
                logging.error(f"Failed to initialize new models or agent: {model_init_e}", exc_info=True)
                self.decision_model = None
                self.hybrid_model = None
                self.combined_agent = None
        else:
            logging.warning("New models module not available. CombinedTradingAgent functionality disabled.")

        # <<< Khởi tạo Position Sizer ở đây >>>
        try:
            self.position_sizer = SmartPositionSizer(self)
            # Optional: Huấn luyện FL ban đầu nếu cần và chưa có model lưu
            if not os.path.exists("federated_sizer_model.pth"):
                 logging.info("Attempting initial training for Federated Sizer...")
                 self.position_sizer.initial_train_federated_model() # Gọi hàm đã sửa
        except Exception as e:
            logging.critical(f"Failed to initialize SmartPositionSizer: {e}", exc_info=True)
            raise # Lỗi nghiêm trọng

        self._cleanup_complete = False # Cờ trạng thái cleanup

        self.trading_env_scaler = None
        scaler_path = "trading_env_scaler.pkl" # Phải khớp với tên file lưu scaler của TradingEnv
        if os.path.exists(scaler_path):
            try:
                self.trading_env_scaler = joblib.load(scaler_path)
                logging.info(f"Loaded TradingEnv scaler from {scaler_path}")
                # Optional: Kiểm tra dimension của scaler nếu cần
                if not hasattr(self.trading_env_scaler, 'n_features_in_'):
                     logging.warning(f"Loaded scaler from {scaler_path} might be invalid (missing n_features_in_).")
                     self.trading_env_scaler = None # Coi như chưa tải được
            except Exception as e:
                logging.error(f"Failed to load TradingEnv scaler: {e}")
                self.trading_env_scaler = None
        else:
            logging.warning(f"TradingEnv scaler file not found at {scaler_path}. DQN adjustment will be skipped.")

        # Tải mô hình DQN
        self.drl_model = None
        drl_model_path = "final_drl_dqn_model.zip" # Tên file lưu model DQN
        if os.path.exists(drl_model_path):
             try:
                  # Cần import DQN từ stable_baselines3
                  from stable_baselines3 import DQN
                  self.drl_model = DQN.load(drl_model_path, device="cuda" if torch.cuda.is_available() else "cpu")
                  logging.info(f"Loaded DRL (DQN) model from {drl_model_path}")
             except Exception as e:
                  logging.error(f"Failed to load DRL model from {drl_model_path}: {e}")
                  self.drl_model = None
        else:
             logging.warning(f"DRL (DQN) model file not found at {drl_model_path}. DQN adjustment will be skipped.")

    async def initialize(self):
        """Khởi tạo kết nối exchange, tải dữ liệu ban đầu và các thành phần phụ thuộc."""
        logging.info("Initializing Bot...")
        await self._initialize_exchange()

        # Lấy dữ liệu ban đầu cho symbol đầu tiên để khởi tạo các thành phần cần data
        first_symbol = CONFIG.get("symbols", [])[0] if CONFIG.get("symbols") else None
        if not first_symbol:
             raise ValueError("No symbols defined in CONFIG.")

        logging.info(f"Fetching initial data for {first_symbol}...")
        initial_data = await self.fetch_multi_timeframe_data(first_symbol)

        # Kiểm tra dữ liệu tối thiểu
        if not initial_data or "15m" not in initial_data or initial_data["15m"].empty:
            logging.error(f"Missing or empty '15m' timeframe in initial data for {first_symbol}")
            raise ValueError(f"Insufficient initial data for {first_symbol}")
        if len(initial_data["15m"]) < CONFIG.get("hmm_warmup_max", 200): # Kiểm tra độ dài tối thiểu
             logging.error(f"Initial '15m' data for {first_symbol} has {len(initial_data['15m'])} rows, less than HMM warmup max ({CONFIG.get('hmm_warmup_max', 200)}).")
             raise ValueError(f"Insufficient initial data length for {first_symbol}")

        # Lưu dữ liệu ban đầu vào self.data
        self.data[first_symbol] = initial_data
        logging.info(f"Initial data loaded and processed for {first_symbol}.")
        tf_primary = CONFIG.get("primary_tf", "15m")
        if self.mode == "live":
            self.monitor_callbacks = {
                "THRESHOLD_HIT": self._handle_monitor_threshold_hit,
                "RR_CROSSED": self._handle_monitor_rr_crossed,
                "ABNORMAL_VOLATILITY": self._handle_monitor_volatility,
            }
            # Lấy danh sách symbol có dữ liệu hợp lệ sau khi initialize data
            valid_symbols_for_monitor = list(self.data.keys())
            if valid_symbols_for_monitor:
                current_running_loop = asyncio.get_running_loop()
                self.realtime_monitor = RealtimeMonitor(
                    symbols=valid_symbols_for_monitor,
                    event_callbacks=self.monitor_callbacks,
                    event_loop=current_running_loop,
                    per_symbol_config=CONFIG.get("monitor_per_symbol_config"), # Thêm config này nếu cần
                )
                logging.info("RealtimeMonitor instance created.")
                # Monitor sẽ được start trong hàm run()
            else:
                logging.error("No valid symbols after data initialization. RealtimeMonitor not created.")
        else: # Chế độ backtest
             self.realtime_monitor = None # Đảm bảo là None
             logging.info("RealtimeMonitor disabled for BACKTEST mode.")

        # Khởi tạo VAH Optimizer
        try:
            # Đảm bảo initial_data["15m"] có đủ cột cho VAHOptimizerEnv
            if all(col in initial_data["15m"].columns for col in ['volatility', 'volume', 'close', 'high', 'low', 'regime']):
                 self.vah_optimizer = self._init_vah_optimizer(initial_data["15m"])
                 logging.info("Successfully initialized VAH Optimizer")
            else:
                 logging.warning("Missing required columns in initial 15m data for VAH Optimizer. Skipping.")
                 self.vah_optimizer = None
        except Exception as e:
            logging.error(f"Failed to initialize VAH Optimizer: {e}", exc_info=True)
            self.vah_optimizer = None
        if self.vah_optimizer is None:
            logging.warning("VAH Optimizer is None; using default bin_size in volume profile calculations")

        # Lấy dữ liệu order book ban đầu (có thể tốn thời gian)
        order_book_data_sac = []
        logging.info("Fetching initial order book data for SAC...")
        # Chỉ lấy cho symbol đầu tiên để khởi tạo SAC Env
        async with self.data_locks[first_symbol]: # Lock khi đọc data
            ob = await self.fetch_order_book(first_symbol, limit=20)
            if not ob:
                last_close = self.data[first_symbol][tf_primary]['close'].iloc[-1] if tf_primary in self.data[first_symbol] else 1.0
                ob = {"bids": [[last_close*0.999, 1]], "asks": [[last_close*1.001, 1]], "timestamp": int(time.time()*1000)}
            order_book_data_sac.append(ob)
        logging.info("Initialized order book data for SAC.")

        # Khởi tạo EntryPointOptimizer (SAC)
        try:
            # Truyền dữ liệu của symbol đầu tiên
            self.entry_optimizer = EntryPointOptimizer(self, initial_data, order_book_data_sac)
            logging.info("EntryPointOptimizer (SAC) initialized.")
            # Việc huấn luyện sẽ diễn ra trong hàm run()
        except Exception as e:
             logging.error(f"Failed to initialize EntryPointOptimizer: {e}", exc_info=True)
             self.entry_optimizer = None # Đặt là None nếu lỗi

        # --- Khởi tạo và Tải MLPAction, Scaler, Fallback ---
        try:
            if self.mlp_action_analyzer: # Only load if initialized successfully
                # <<< UPDATE Filenames >>>
                mlp_model_path = SCRIPT_DIR_API / "mlp_action_model.pth"
                mlp_scaler_path = SCRIPT_DIR_API / "mlp_action_feature_scaler.pkl"

                # Load model and get fallback value
                loaded_ok, fallback_pred = self.mlp_action_analyzer.load_model(str(mlp_model_path))
                if loaded_ok:
                    self.mlp_action_fallback = fallback_pred if fallback_pred is not None else 0.0
                    logging.info(f"MLPAction model loaded from {mlp_model_path}. Fallback: {self.mlp_action_fallback:.4f}")
                else:
                    logging.warning(f"Failed to load MLPAction model from {mlp_model_path}. Predictions will use fallback.")
                    self.mlp_action_analyzer.is_trained = False # Ensure flag is False

                # Load Scaler
                if mlp_scaler_path.exists():
                     try:
                          self.mlp_action_scaler = joblib.load(mlp_scaler_path)
                          logging.info(f"MLPAction feature scaler loaded from {mlp_scaler_path}.")
                          # <<< VALIDATE Scaler Dimension >>>
                          expected_scaler_dim = CONFIG.get("mlp_action_input_dim", 16)
                          if not hasattr(self.mlp_action_scaler, 'n_features_in_') or self.mlp_action_scaler.n_features_in_ != expected_scaler_dim:
                               logging.error(f"MLPAction scaler dimension mismatch! Expected {expected_scaler_dim}, got {getattr(self.mlp_action_scaler, 'n_features_in_', 'N/A')}. Disabling scaler.")
                               self.mlp_action_scaler = None
                     except Exception as scale_load_e:
                          logging.error(f"Failed to load MLPAction scaler from {mlp_scaler_path}: {scale_load_e}")
                          self.mlp_action_scaler = None
                else:
                     logging.warning(f"MLPAction scaler file not found: {mlp_scaler_path}. MLP predictions might be inaccurate.")
            else:
                logging.error("MLPAction analyzer instance is None. Cannot load model/scaler.")

        except Exception as e:
            logging.error(f"Failed during MLPAction model/scaler loading: {e}", exc_info=True)
            if self.mlp_action_analyzer: self.mlp_action_analyzer.is_trained = False
            self.mlp_action_scaler = None

        logging.info("Bot initialization complete.")


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

    def _initialize_components(self):
        self.balance = CONFIG.get("initial_balance", 10000)
        self.equity_peak = self.balance # Đặt equity_peak ban đầu bằng balance
        self.trade_history = []
        self.scalers = {} # Scaler cho XGBoost features (theo symbol)
        self.models = {} # Model XGBoost (theo symbol)
        self.drl_model = None # Model DQN điều chỉnh size
        self.entry_optimizer = None # SAC model tối ưu entry
        self.sentiment_analyzer = None
        self.positions_lock = asyncio.Lock()
        self.balance_exposure_lock = asyncio.Lock()
        self.data_lock = asyncio.Lock()
        self.circuit_breaker_lock = asyncio.Lock()
        try:
            # *** KHỞI TẠO VÀ TRUYỀN KEYS ***
            self.sentiment_analyzer = NewsSentimentAnalyzer(
                self,                     # Pass bot instance
                CONFIG,                   # Pass full config (nó sẽ tự lấy sentiment_config)
                fmp_api_key=fmp_api_key,  # Pass FMP key
                av_api_key=av_api_key     # Pass AV key
            )
            logging.info("News Sentiment Analyzer initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize News Sentiment Analyzer: {e}", exc_info=True)
            self.sentiment_analyzer = None
        try:
            self.stop_system = IntelligentStopSystem() # SL động
        except Exception as e:
            logging.critical(f"Failed to initialize core components (Optimizer/Stop): {e}", exc_info=True)
            raise

        self.gap_levels = {}
        self.walk_forward_windows = []
        self.temporal_features = [
            'hour', 'day_of_week', 'is_weekend',
            'time_since_last_high', 'time_since_last_gap',
            'volatility_regime_change','hybrid_regime'
        ]
        self.circuit_breaker_triggered = False
        self.last_trade_time = None # Nên là pd.Timestamp hoặc None
        self.exposure_per_symbol = {symbol: 0.0 for symbol in CONFIG.get("symbols", [])}
        self.trailing_stops = {} # Quản lý trailing stop theo symbol/position ID
        self.open_positions = {} # Quản lý các vị thế đang mở theo symbol
        self.data = {} # Lưu trữ dataframes cho các symbol/timeframe
        for symbol in CONFIG.get("symbols", []):
            self.last_processed_candle_ts[symbol] = {}
        self.rollback_state = None # State để rollback khi lỗi nghiêm trọng

    def _init_vah_optimizer(self, initial_data: pd.DataFrame):
        """Khởi tạo môi trường và agent DQN cho VAH Optimizer."""
        # Định nghĩa Env bên trong để tránh xung đột tên
        class VAHOptimizerEnv(Env):
            metadata = {'render_modes': []} # Thêm metadata nếu dùng Gymnasium mới
            def __init__(self, data, config):
                super().__init__()
                self.data = data
                self.config = config
                self.vah_bin_range = self.config.get("vah_bin_range", (20, 60))
                # Action là index của bin size lựa chọn
                num_bin_choices = 5 # Ví dụ: 20, 30, 40, 50, 60
                self.action_space = Discrete(num_bin_choices)
                # Observation space: volatility, volume, std dev, cci, adosc, regime
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
                self.min_lookback = 100 # Cần cho VAH và các chỉ báo
                self.current_step = self.min_lookback # Bắt đầu từ điểm có đủ lookback

            def _get_bin_size(self, action: int) -> int:
                """Chuyển action index thành bin size thực tế."""
                step = (self.vah_bin_range[1] - self.vah_bin_range[0]) / (self.action_space.n - 1) if self.action_space.n > 1 else 0
                bin_size = self.vah_bin_range[0] + action * step
                return max(2, int(bin_size)) # Đảm bảo ít nhất 2 bins

            def step(self, action):
                if self.current_step >= len(self.data) - 1:
                     # Nếu đã ở cuối, không thể tính reward tương lai
                     obs = self._get_obs()
                     return obs, 0.0, True, False, {"info": "End of data reached"}

                bin_size = self._get_bin_size(action)
                reward = self._calculate_vah_reward(bin_size) # Tính reward dựa trên độ chính xác VAH
                self.current_step += 1
                done = self.current_step >= len(self.data) - 1 # Kết thúc khi duyệt gần hết data
                truncated = False # Không có giới hạn thời gian riêng
                obs = self._get_obs()
                return obs, reward, done, truncated, {}

            def reset(self, *, seed: Optional[int] = None, **kwargs):
                super().reset(seed=seed) # Gọi reset của lớp cha Gymnasium
                self.current_step = self.min_lookback
                info = {}
                return self._get_obs(), info

            def _calculate_vah_reward(self, bin_size):
                """Tính reward dựa trên việc VAH có hoạt động như kháng cự/hỗ trợ không."""
                # Sử dụng dữ liệu từ [current_step - lookback + 1, current_step] để tính VAH
                # Kiểm tra VAH trên nến tiếp theo (current_step + 1)
                reward = 0.0
                lookback_start = self.current_step - self.min_lookback + 1
                if lookback_start < 0 or self.current_step + 1 >= len(self.data):
                    return 0.0 # Không đủ dữ liệu

                data_slice = self.data.iloc[lookback_start : self.current_step + 1]
                next_candle = self.data.iloc[self.current_step + 1]

                if data_slice.empty or next_candle.empty: return 0.0

                vah = self._calc_vah(data_slice, bin_size)
                if vah is None: return 0.0

                # Ví dụ reward: +1 nếu giá bật lại từ VAH, -1 nếu xuyên qua dễ dàng
                price_low = next_candle['low']
                price_high = next_candle['high']
                price_close = next_candle['close']

                # Kiểm tra xem giá có chạm VAH không
                touched_vah = price_low <= vah <= price_high

                if touched_vah:
                    # Nếu chạm và đóng cửa bật lại (ví dụ: đóng trên VAH nếu chạm từ dưới lên)
                    if price_close > vah: reward += 0.5
                    # Nếu chạm và đóng cửa bật lại (ví dụ: đóng dưới VAH nếu chạm từ trên xuống)
                    elif price_close < vah: reward += 0.5
                    # Nếu chạm nhưng đóng cửa gần VAH (không rõ ràng)
                    else: reward += 0.1
                else:
                    # Nếu không chạm VAH, có thể phạt nhẹ hoặc không làm gì
                    reward -= 0.1

                # Có thể thêm logic phức tạp hơn dựa trên volume tại VAH, etc.
                return reward

            def _calc_vah(self, data_slice, bin_size):
                """Tính VAH (Value Area High - thường là Point of Control trong cách dùng này)."""
                try:
                    low_min = data_slice['low'].min()
                    high_max = data_slice['high'].max()
                    if high_max <= low_min: return None # Tránh lỗi bins

                    bins = np.linspace(low_min, high_max, bin_size)
                    # Sử dụng 'close' làm giá đại diện, 'volume' làm trọng số
                    hist, bin_edges = np.histogram(data_slice['close'], bins=bins, weights=data_slice['volume'])
                    poc_index = np.argmax(hist) # Index của bin có volume cao nhất (POC)
                    # Trả về cạnh trên của bin POC làm VAH (hoặc cạnh dưới làm VAL)
                    # Hoặc có thể trả về điểm giữa của bin POC
                    vah = bin_edges[poc_index + 1] # Cạnh trên của bin POC
                    return vah
                except Exception as e:
                     # logging.debug(f"Error calculating VAH: {e}") # Giảm log
                     return None # Trả về None nếu lỗi

            def _get_obs(self) -> np.ndarray:
                """Lấy observation tại current_step."""
                # Đảm bảo current_step hợp lệ
                safe_step = min(self.current_step, len(self.data) - 1)
                lookback_start = max(0, safe_step - 14) # Lookback cho CCI, ADOSC
                data_slice = self.data.iloc[lookback_start : safe_step + 1]

                if len(data_slice) < 14: # Không đủ dữ liệu cho chỉ báo
                     # Trả về giá trị mặc định an toàn
                     return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

                obs_vol = data_slice['volatility'].iloc[-1]
                obs_volume = data_slice['volume'].iloc[-1]
                obs_std = data_slice['close'].pct_change().std()
                # Tính CCI, ADOSC, cần đủ lookback
                try:
                    cci = talib.CCI(data_slice['high'], data_slice['low'], data_slice['close'], 14).iloc[-1]
                    adosc = talib.ADOSC(data_slice['high'], data_slice['low'], data_slice['close'], data_slice['volume'], fastperiod=3, slowperiod=10).iloc[-1]
                except Exception: # Lỗi talib nếu dữ liệu không đủ/không hợp lệ
                    cci = 0.0
                    adosc = 0.0
                obs_regime = data_slice['hybrid_regime'].iloc[-1]

                obs_array = np.array([
                    obs_vol, obs_volume, obs_std, cci, adosc, obs_regime
                ], dtype=np.float32)

                # Xử lý NaN/Inf
                obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
                return obs_array

        # Khởi tạo Env và Agent
        try:
             env = VAHOptimizerEnv(initial_data, CONFIG)
             # Cân nhắc các siêu tham số DQN hoặc tải model đã lưu
             model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-4, buffer_size=5000,
                         learning_starts=100, batch_size=32, train_freq=4, gradient_steps=1,
                         target_update_interval=250, exploration_final_eps=0.02)
             # Optional: Huấn luyện nhanh
             # model.learn(total_timesteps=1000)
             return model
        except Exception as e:
             logging.error(f"Failed to initialize DQN for VAH Optimizer: {e}", exc_info=True)
             return None


    async def _cleanup(self):
        async with self.misc_state_lock: # Lock để set cờ cleanup
            if self._cleanup_complete: return
            logging.info("🚀 Starting cleanup process...")
            self._cleanup_complete = True
        try:
            # 1. Stop Data Feeds Safely
            if hasattr(self, 'data_feeds') and self.data_feeds:
                logging.info(f"Stopping {sum(len(feeds) for feeds in self.data_feeds.values())} data feeds...")
                stop_tasks = []
                for symbol, feeds in self.data_feeds.items():
                    for tf, feed in feeds.items():
                        if feed is not None and hasattr(feed, 'stop'):
                            stop_tasks.append(asyncio.create_task(
                                self._stop_feed_safely(feed, symbol, tf)
                            ))
                if stop_tasks:
                    results = await asyncio.gather(*stop_tasks, return_exceptions=True)
                    for i, result in enumerate(results):
                         if isinstance(result, Exception):
                              # Lỗi đã được log trong _stop_feed_safely
                              pass # logging.error(f"Error stopping feed task {i}: {result}")

            # 2. Close Exchange Connection Safely
            if hasattr(self, 'exchange') and self.exchange and hasattr(self.exchange, 'close'):
                logging.info("Closing exchange connection...")
                try:
                    # Cho phép timeout dài hơn một chút
                    await asyncio.wait_for(self.exchange.close(), timeout=15.0)
                    logging.info("✅ Exchange connection closed.")
                except asyncio.TimeoutError:
                     logging.error("⚠️ Timeout closing exchange connection.")
                except Exception as e:
                    logging.error(f"⚠️ Failed to close exchange: {e}", exc_info=True)

            # 3. Cancel Remaining Async Tasks (ngoại trừ task hiện tại)
            current_task = asyncio.current_task()
            all_tasks = asyncio.all_tasks()
            pending = [t for t in all_tasks if t is not current_task and not t.done()]
            if pending:
                logging.info(f"🛑 Cancelling {len(pending)} pending tasks...")
                for task in pending:
                    task.cancel()
                # Chờ các task bị hủy hoàn thành (hoặc timeout)
                await asyncio.gather(*pending, return_exceptions=True)
                logging.info("Pending tasks cancelled.")

            # 4. Release Additional Resources
            await self._release_additional_resources()

        except Exception as e:
            logging.critical(f"🔥 Critical error during cleanup: {e}", exc_info=True)
        finally:
            # 5. Clear GPU Cache (nếu dùng)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Cleared CUDA cache.")

            logging.info("🛑 Cleanup process completed.")
            self._cleanup_complete = True # Đánh dấu đã cleanup


    async def _stop_feed_safely(self, feed, symbol, timeframe):
        max_retries = 3; delay = 0.5
        for attempt in range(max_retries):
            try:
                if hasattr(feed, 'stop') and callable(feed.stop):
                     feed.stop()
                     await asyncio.sleep(delay) # Chờ một chút để feed dừng hẳn
                     # Optional: Kiểm tra trạng thái running nếu có
                     # if hasattr(feed, 'running') and not feed.running:
                     #      logging.info(f"✅ Stopped feed {symbol} {timeframe}")
                     #      return True
                     logging.info(f"✅ Requested stop for feed {symbol} {timeframe}")
                     return True # Giả định stop thành công nếu không có lỗi
                else:
                     logging.warning(f"Feed object for {symbol} {timeframe} lacks a stop() method.")
                     return False
            except Exception as e:
                logging.error(f"⚠️ Attempt {attempt+1}/{max_retries} failed to stop {symbol} {timeframe} feed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2**attempt)) # Backoff
                else:
                     logging.error(f"⚠️ Final attempt failed for {symbol} {timeframe}.")
                     return False
        return False # Không thành công sau max_retries


    async def _release_additional_resources(self):
        try:
            # Xóa cache nếu có
            if hasattr(self, 'indicator_cache'): self.indicator_cache.clear()
            # Đóng kết nối DB nếu có
            # if hasattr(self, 'db_conn'): await self.db_conn.close()
            logging.info("✅ Additional resources released/cleared.")
        except Exception as e:
            logging.error(f"⚠️ Resource release error: {e}", exc_info=True)

    def _validate_config(self):
        # ... (Kiểm tra kỹ hơn) ...
        logging.info("Validating configuration...")
        symbols = CONFIG.get("symbols", [])
        if not symbols or not isinstance(symbols, list):
             raise ValueError("CONFIG['symbols'] must be a non-empty list.")

        timeframes = CONFIG.get("timeframes", [])
        valid_tfs = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"] # Các timeframe phổ biến của ccxt
        if not timeframes or not isinstance(timeframes, list) or not all(tf in valid_tfs for tf in timeframes):
             raise ValueError(f"CONFIG['timeframes'] must be a non-empty list with valid timeframes (e.g., {valid_tfs})")
        # Đảm bảo '15m' tồn tại nếu các logic khác phụ thuộc vào nó
        if '15m' not in timeframes:
             logging.warning("'15m' timeframe is not in CONFIG['timeframes'], some logic might depend on it.")


        required_numeric = ["initial_balance", "leverage", "risk_per_trade", "fee_rate",
                            "max_trades_per_day", "max_account_risk", "min_position_size",
                            "max_position_size", "prob_threshold", "sentiment_window",
                            "min_samples", "lookback_window", "hmm_warmup_min",
                            "hmm_warmup_max", "max_gap_storage",
                            "circuit_breaker_threshold", "circuit_breaker_cooldown",
                            "max_exposure", "volume_spike_multiplier",
                            "volume_sustained_multiplier", "adx_threshold",
                            "volume_ma20_threshold", "days_to_fetch", "rate_limit_delay"]
        for key in required_numeric:
             val = CONFIG.get(key)
             if val is None or not isinstance(val, (int, float)) or val < 0:
                  raise ValueError(f"CONFIG['{key}'] must be a non-negative number. Found: {val}")

        # Kiểm tra logic ràng buộc
        if CONFIG["risk_per_trade"] > CONFIG["max_account_risk"]:
            raise ValueError("CONFIG Error: risk_per_trade cannot exceed max_account_risk.")
        if CONFIG["min_position_size"] > CONFIG["max_position_size"]:
             raise ValueError("CONFIG Error: min_position_size cannot exceed max_position_size.")
        if CONFIG["hmm_warmup_min"] > CONFIG["hmm_warmup_max"]:
             raise ValueError("CONFIG Error: hmm_warmup_min cannot exceed hmm_warmup_max.")
        if not (0 < CONFIG["prob_threshold"] < 1):
             raise ValueError("CONFIG Error: prob_threshold must be between 0 and 1.")
        # Thêm các kiểm tra khác nếu cần
        logging.info("Configuration validated successfully.")


    async def fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
        """Lấy dữ liệu đa khung thời gian, sử dụng cache nếu có."""
        symbol_safe = symbol.replace('/', '_').replace(':', '')
        cache_file = f"{symbol_safe}_data.pkl"
        data: Dict[str, Optional[pd.DataFrame]] = {}
        use_cache = CONFIG.get("use_data_cache", True)
        save_cache = CONFIG.get("save_data_cache", True)

        # --- Tải từ Cache ---
        if use_cache and os.path.exists(cache_file):
            try:
                logging.info(f"Loading data from cache for {symbol}: {cache_file}")
                try:
                    data_cached = joblib.load(cache_file)
                except:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        data_cached = pickle.load(f)

                if isinstance(data_cached, dict):
                    all_timeframes_in_config = CONFIG.get("timeframes", [])
                    # Kiểm tra xem cache có đủ các timeframe cần thiết không và dữ liệu có hợp lệ không
                    cache_is_sufficient = True
                    temp_data_from_cache = {}
                    for tf_cfg in all_timeframes_in_config:
                        df_cached = data_cached.get(tf_cfg)
                        if isinstance(df_cached, pd.DataFrame) and not df_cached.empty:
                            temp_data_from_cache[tf_cfg] = df_cached # Tạm thời lưu df thô từ cache
                        else:
                            cache_is_sufficient = False # Thiếu timeframe hoặc data không hợp lệ
                            break
                    
                    if cache_is_sufficient:
                        logging.info(f"Successfully loaded {len(temp_data_from_cache)} timeframes from cache for {symbol}. Applying indicators...")
                        # Áp dụng tính toán chỉ báo cho dữ liệu từ cache
                        for tf_cached, df_from_cache in temp_data_from_cache.items():
                            try:
                                # Đảm bảo df_from_cache là bản sao để không sửa đổi cache gốc nếu không muốn
                                data[tf_cached] = self.calculate_advanced_indicators(df_from_cache.copy())
                            except Exception as calc_e_cache:
                                logging.error(f"Error calculating indicators for cached {symbol} {tf_cached}: {calc_e_cache}")
                                data[tf_cached] = None # Đánh dấu lỗi
                        return data # Trả về nếu đủ từ cache và đã xử lý
                    else:
                        logging.warning(f"Cache for {symbol} is incomplete or invalid. Refetching necessary timeframes.")
                        # data sẽ được xây dựng lại từ đầu hoặc kết hợp
                        data = {} # Reset data nếu cache không đủ/lỗi để đảm bảo fetch lại

                else:
                    logging.warning(f"Invalid data format in cache file {cache_file}. Refetching.")
                    data = {}

            except Exception as e:
                logging.error(f"Error loading data from cache {cache_file}: {e}. Refetching.")
                data = {}

        # --- Fetch mới nếu không dùng cache hoặc cache thiếu/lỗi ---
        logging.info(f"Fetching fresh data for {symbol} if needed...")
        fetch_tasks = []
        timeframes_to_fetch_config = CONFIG.get("timeframes", [])
        
        # Xác định những timeframe thực sự cần fetch (chưa có trong `data` hoặc `data[tf]` là None)
        required_tfs_to_fetch_now = [tf for tf in timeframes_to_fetch_config if data.get(tf) is None]

        if not required_tfs_to_fetch_now:
            logging.info(f"All required timeframes already loaded and processed for {symbol} (possibly from valid cache).")
            # Kiểm tra lại xem data có thực sự chứa tất cả các tf không (phòng trường hợp logic trên có lỗi)
            if all(tf in data and isinstance(data[tf], pd.DataFrame) for tf in timeframes_to_fetch_config):
                return data
            else: # Nếu data không đủ dù không có gì để fetch, có thể là lỗi logic cache
                logging.warning(f"Data for {symbol} seems incomplete despite no TFs marked for fetching. Re-evaluating fetch needs.")
                data = {} # Reset để fetch lại toàn bộ nếu logic cache trước đó có vấn đề
                required_tfs_to_fetch_now = timeframes_to_fetch_config


        if required_tfs_to_fetch_now:
            logging.info(f"Fetching timeframes for {symbol}: {required_tfs_to_fetch_now}")
            for tf_fetch in required_tfs_to_fetch_now:
                # Giả sử _fetch_ohlcv lấy dữ liệu từ CONFIG["days_to_fetch"] ngày trước
                days_to_fetch = CONFIG.get("days_to_fetch", 365)
                since_timestamp = self.exchange.milliseconds() - (days_to_fetch * 24 * 60 * 60 * 1000)
                fetch_tasks.append(self._fetch_ohlcv(symbol, tf_fetch, since=since_timestamp))

            results_raw = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            newly_fetched_data_processed = {}
            for i, raw_result_item in enumerate(results_raw):
                tf_current_fetch = required_tfs_to_fetch_now[i]
                df_for_processing = None # DataFrame sau khi _process_ohlcv

                if isinstance(raw_result_item, list) and raw_result_item: # Kiểm tra result là list và không rỗng
                    try:
                        df_for_processing = self._process_ohlcv(raw_result_item) # <<<< SỬA LỖI Ở ĐÂY
                    except Exception as proc_e:
                        logging.error(f"Error processing raw OHLCV for {symbol} {tf_current_fetch}: {proc_e}")
                        df_for_processing = None # Đảm bảo là None nếu xử lý lỗi
                elif isinstance(raw_result_item, Exception):
                    logging.error(f"Error fetching data for {symbol} {tf_current_fetch}: {raw_result_item}")
                    df_for_processing = None
                else: # Trường hợp trả về None hoặc empty list từ _fetch_ohlcv
                    logging.warning(f"No data or invalid raw data fetched for {symbol} {tf_current_fetch}.")
                    df_for_processing = None

                # Bây giờ df_for_processing là DataFrame hoặc None
                if isinstance(df_for_processing, pd.DataFrame) and not df_for_processing.empty:
                    logging.debug(f"Processing fetched DataFrame for {symbol} {tf_current_fetch}...")
                    min_len = CONFIG.get("hmm_warmup_max", 200)
                    if len(df_for_processing) >= min_len:
                        try:
                            # Resample (nếu cần, logic này có thể được giữ lại hoặc bỏ đi tùy yêu cầu)
                            freq_map = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "h", "2h": "2h", "4h": "4h", "1d": "d"}
                            freq = freq_map.get(tf_current_fetch)
                            df_resampled_or_original = df_for_processing # Mặc định là df gốc
                            if freq:
                                try:
                                    df_resampled_or_original = df_for_processing.resample(freq).agg({
                                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                                    }).ffill()
                                except Exception as resample_e:
                                     logging.warning(f"Could not resample {symbol} {tf_current_fetch}: {resample_e}. Using original fetched data.")
                                     # df_resampled_or_original vẫn là df_for_processing

                            df_processed_with_indicators = self.calculate_advanced_indicators(df_resampled_or_original.copy())
                            newly_fetched_data_processed[tf_current_fetch] = df_processed_with_indicators
                            logging.debug(f"Successfully processed fetched data for {symbol} {tf_current_fetch}.")
                        except Exception as calc_e_fetch:
                            logging.error(f"Error calculating indicators for fetched {symbol} {tf_current_fetch}: {calc_e_fetch}", exc_info=True)
                            newly_fetched_data_processed[tf_current_fetch] = None
                    else:
                        logging.warning(f"Fetched data for {symbol} {tf_current_fetch} has {len(df_for_processing)} rows, less than minimum required {min_len}. Skipping.")
                        newly_fetched_data_processed[tf_current_fetch] = None
                else: # df_for_processing là None hoặc empty
                    newly_fetched_data_processed[tf_current_fetch] = None
            
            # Kết hợp dữ liệu từ cache (nếu có và hợp lệ) với dữ liệu mới fetch và đã xử lý
            # `data` có thể đã chứa dữ liệu từ cache nếu cache không đủ và cần fetch thêm
            # `newly_fetched_data_processed` chứa dữ liệu mới fetch và đã xử lý
            final_data = {**data, **newly_fetched_data_processed} # Dữ liệu mới sẽ ghi đè
        else: # Không có gì cần fetch
            final_data = data # Dữ liệu hoàn toàn từ cache (đã được xử lý ở phần cache)

        # --- Lưu vào Cache (dữ liệu đã xử lý với indicators) ---
        valid_data_to_save = {tf_save: df_save for tf_save, df_save in final_data.items() if isinstance(df_save, pd.DataFrame) and not df_save.empty}
        if save_cache and valid_data_to_save: # Chỉ lưu nếu có dữ liệu hợp lệ và được phép
            try:
                # Kiểm tra xem có thực sự cần lưu không (ví dụ, nếu dữ liệu không thay đổi so với cache)
                # Điều này hơi phức tạp, tạm thời cứ lưu nếu có dữ liệu
                logging.info(f"Saving combined and processed data to cache for {symbol}: {cache_file}")
                joblib.dump(valid_data_to_save, cache_file)
            except Exception as e_save:
                logging.error(f"Error saving data to cache {cache_file}: {e_save}")

        # Trả về dữ liệu cuối cùng (có thể chứa None cho các timeframe lỗi)
        return final_data


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


    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[dict]:
        # ... (Giữ nguyên, có thể thêm retry) ...
        max_retries = 3; delay = 2
        for attempt in range(max_retries):
            try:
                order_book = await self.exchange.fetch_order_book(symbol, limit=limit)
                # Validate order book structure
                if isinstance(order_book, dict) and 'bids' in order_book and 'asks' in order_book:
                    # Optional: Thêm kiểm tra sâu hơn về format của bids/asks
                    return { "bids": order_book["bids"], "asks": order_book["asks"], "timestamp": order_book.get("timestamp", int(time.time()*1000)) } # Lấy ts nếu có
                else:
                     logging.warning(f"Invalid order book structure received for {symbol}.")
                     return None
            except (ccxt.NetworkError, ccxt.RequestTimeout, aiohttp.ClientError) as e:
                logging.warning(f"Network error fetching OB for {symbol} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1: await asyncio.sleep(delay * (attempt + 1))
                else: return None
            except ccxt.RateLimitExceeded as e:
                logging.warning(f"Rate limit exceeded fetching OB for {symbol} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1: await asyncio.sleep(delay * 2 * (attempt + 1)) # Chờ lâu hơn
                else: return None
            except ccxt.ExchangeError as e:
                 logging.error(f"Exchange error fetching OB for {symbol}: {e}")
                 return None # Lỗi sàn, không retry
            except Exception as e:
                logging.error(f"Error fetching order book for {symbol}: {e}", exc_info=True)
                return None # Lỗi khác, không retry
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
        if not isinstance(df, pd.DataFrame) or df.empty:
            # logging.warning("calculate_advanced_indicators: Input DataFrame is empty.")
            return df # Trả về df rỗng/None
        if len(df) < 2:
            # logging.warning("calculate_advanced_indicators: DataFrame too small for most indicators.")
            return df # Trả về nếu quá nhỏ
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
            df.loc[:, "VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan) # Tránh chia cho 0

            # Các chỉ báo tùy chỉnh
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

            # EMA_diff
            if 'EMA_50' in df.columns and 'EMA_200' in df.columns and 'close' in df.columns:
                close_safe = df['close'].replace(0, np.nan)
                df['EMA_diff'] = (df['EMA_50'] - df['EMA_200']) / close_safe
            else: df['EMA_diff'] = 0 # Hoặc np.nan

            # log_volume
            if 'volume' in df.columns:
                df['log_volume'] = np.log1p(df['volume'].clip(lower=0))
            else: df['log_volume'] = 0 # Hoặc np.nan

            # BB_width
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns and 'BB_middle' in df.columns:
                bb_middle_safe = df['BB_middle'].replace(0, np.nan)
                df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_middle_safe
            else: df['BB_width'] = 0 # Hoặc np.nan

            # VWAP_ADX_interaction
            if 'VWAP' in df.columns and 'ADX' in df.columns:
                df['VWAP_ADX_interaction'] = df['VWAP'] * df['ADX'] / 100 # Chia 100 để giữ scale nhỏ hơn?
            else: df['VWAP_ADX_interaction'] = 0

            # BB_EMA_sync
            if 'BB_width' in df.columns and 'EMA_diff' in df.columns: # Cần tính 2 cái trên trước
                df['BB_EMA_sync'] = df['BB_width'] * df['EMA_diff']
            else: df['BB_EMA_sync'] = 0

            # volume_anomaly
            if 'log_volume' in df.columns: # Cần log_volume
                vol_anomaly_window = 24*3 # ~3 ngày
                log_vol_mean = df['log_volume'].rolling(window=vol_anomaly_window, min_periods=vol_anomaly_window//2).mean()
                log_vol_std = df['log_volume'].rolling(window=vol_anomaly_window, min_periods=vol_anomaly_window//2).std()
                df['volume_anomaly'] = (df['log_volume'] > log_vol_mean + 2*log_vol_std).astype(int)
            else: df['volume_anomaly'] = 0

            symbol_for_precision = df.attrs.get('symbol', list(self.data.keys())[0] if self.data else None) # Get symbol if stored in attrs, else fallback
            if symbol_for_precision:
                 df.attrs['price_precision_digits'] = self._safe_get_price_precision_digits(symbol_for_precision)
                 df.attrs['amount_precision_digits'] = self._safe_get_amount_precision_digits(symbol_for_precision)

        except Exception as e:
             logging.error(f"Error during indicator calculation: {e}", exc_info=True)

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

        return time_since.fillna(0) # Fill NaN ở đầu

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


    def _get_vah_state(self, df: pd.DataFrame) -> np.ndarray:
        try:
             vol = df['volatility'].mean()
             volume = df['volume'].mean()
             std_pct = df['close'].pct_change().std()
             # Tính CCI, ADOSC với lookback phù hợp (14 cho CCI, 3/10 cho ADOSC)
             cci = talib.CCI(df['high'], df['low'], df['close'], 14).iloc[-1]
             adosc = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10).iloc[-1]
             regime = df['hybrid_regime'].iloc[-1]

             state_array = np.array([vol, volume, std_pct, cci, adosc, regime], dtype=np.float32)
             # Xử lý NaN/Inf
             if not np.isfinite(state_array).all():
                  # logging.warning("NaN/Inf detected in VAH state. Replacing with 0.")
                  state_array = np.nan_to_num(state_array, nan=0.0, posinf=0.0, neginf=0.0)
             return state_array
        except Exception as e:
             logging.error(f"Error calculating VAH state: {e}. Returning zeros.")
             return np.zeros(6, dtype=np.float32)
        
    async def _calculate_and_add_hybrid_regimes(self, symbol: str):
        func_name = "_calculate_and_add_hybrid_regimes_optimized" # Đổi tên để phân biệt

        # --- 1. Kiểm tra điều kiện cơ bản (NGOÀI LOCK) ---
        if not _new_models_available or not self.hybrid_model:
            logging.warning(f"{func_name} ({symbol}): Hybrid model not available. Skipping.")
            return

        tf_primary = CONFIG.get("primary_tf", "15m")
        required_tfs_hybrid = list(CONFIG.get('required_tfs_hybrid', ()))
        hybrid_seq_map = CONFIG.get('HYBRID_SEQ_LEN_MAP', {})
        if not required_tfs_hybrid or not hybrid_seq_map: return
        if tf_primary not in required_tfs_hybrid: required_tfs_hybrid.append(tf_primary)

        # === Phần 2: Đọc dữ liệu cần thiết từ self.data (BÊN TRONG LOCK NGẮN) ===
        ohlcv_data_dict = {}; index_dict = {}; primary_df_len = 0; min_len_ok = True
        max_seq_len_needed = 0; start_calc_idx_primary = -1; primary_index = None
        lock_acquired_read = False
        try:
            logging.debug(f"Acquiring data_lock for reading ({symbol})...")
            await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            lock_acquired_read = True
            logging.debug(f"Data_lock acquired for reading ({symbol}).")

            # Kiểm tra dữ liệu tồn tại và đủ dài
            if symbol not in self.data or not all(tf in self.data[symbol] and isinstance(self.data[symbol][tf], pd.DataFrame) and not self.data[symbol][tf].empty for tf in required_tfs_hybrid):
                min_len_ok = False # Đánh dấu không đủ data
            else:
                # Xác định phạm vi tính toán và sao chép dữ liệu cần thiết
                df_primary = self.data[symbol][tf_primary]; primary_df_len = len(df_primary)
                for tf in required_tfs_hybrid:
                    df_tf = self.data[symbol][tf]; seq_len = hybrid_seq_map.get(tf)
                    if seq_len is None or len(df_tf) < seq_len: min_len_ok = False; break
                    max_seq_len_needed = max(max_seq_len_needed, seq_len)
                    # Sao chép dữ liệu OHLCV numpy và index
                    ohlcv_data_dict[tf] = df_tf[['open', 'high', 'low', 'close', 'volume']].copy().ffill().bfill().fillna(0).values
                    index_dict[tf] = df_tf.index
                if min_len_ok:
                    start_calc_idx_primary = max_seq_len_needed - 1
                    primary_index = index_dict[tf_primary] # Lấy index chính
                    # Kiểm tra lại điều kiện bắt đầu tính toán
                    if start_calc_idx_primary >= primary_df_len: min_len_ok = False

        except asyncio.TimeoutError:
            logging.error(f"Timeout acquiring data_lock for reading ({symbol}). Skipping."); return
        except Exception as read_e:
            logging.error(f"Error reading data inside lock ({symbol}): {read_e}"); return
        finally:
            # <<< NHẢ LOCK NGAY SAU KHI ĐỌC XONG >>>
            if lock_acquired_read and self.data_locks[symbol].locked():
                self.data_locks[symbol].release(); logging.debug(f"Data_lock released after reading ({symbol}).")

        # === Thoát nếu không đủ dữ liệu ===
        if not min_len_ok:
             # Gọi hàm thêm cột default (cần lock lại)
             await self._add_default_hybrid_regime_column_safe(symbol, tf_primary)
             return
        if start_calc_idx_primary < 0 or primary_index is None: # Kiểm tra các biến cần thiết
             logging.error(f"Internal error: Invalid state after reading data ({symbol})."); return

        num_predictions_possible = primary_df_len - start_calc_idx_primary
        if num_predictions_possible <= 0: return # Không có gì để dự đoán

        # === Phần 3: Chuẩn bị và Chạy Model (NGOÀI LOCK) ===
        logging.info(f"Calculating {num_predictions_possible} hybrid regimes for {symbol} [{tf_primary}] (Optimized Lock)...")
        self.hybrid_model.eval(); DEVICE = next(self.hybrid_model.parameters()).device
        batch_size = CONFIG.get("embedding_precompute_batch_size", 128); all_regime_preds = {}

        # Vòng lặp tính toán batch (Dùng dữ liệu đã sao chép: ohlcv_data_dict, index_dict)
        with torch.no_grad():
            for i in range(0, num_predictions_possible, batch_size):
                # ... (Logic tạo batch input TỪ ohlcv_data_dict và index_dict - GIỮ NGUYÊN) ...
                batch_end_idx_in_pred_range = min(i + batch_size, num_predictions_possible); actual_batch_start_idx = start_calc_idx_primary + i; actual_batch_end_idx = start_calc_idx_primary + batch_end_idx_in_pred_range
                batch_end_timestamps = primary_index[actual_batch_start_idx : actual_batch_end_idx]; current_batch_input_dict = {}; valid_batch = True
                if len(batch_end_timestamps) == 0: continue
                for tf_h in required_tfs_hybrid:
                    seq_len_tf = hybrid_seq_map[tf_h]; tf_index = index_dict[tf_h]; tf_values = ohlcv_data_dict[tf_h]; batch_windows_tf = []
                    for end_ts in batch_end_timestamps:
                        try:
                            end_loc = tf_index.get_loc(end_ts, method='ffill'); start_loc = end_loc - seq_len_tf + 1
                            if start_loc < 0 or tf_values[start_loc : end_loc + 1].shape[0] != seq_len_tf or np.isnan(tf_values[start_loc : end_loc + 1]).any(): valid_batch=False; break
                            batch_windows_tf.append(tf_values[start_loc : end_loc + 1])
                        except Exception: valid_batch=False; break
                    if not valid_batch: break
                    if batch_windows_tf:
                        try: current_batch_input_dict[tf_h] = torch.tensor(np.array(batch_windows_tf), dtype=torch.float32).to(DEVICE); 
                        except Exception: valid_batch=False; break
                if not valid_batch or len(current_batch_input_dict) != len(required_tfs_hybrid): continue
                # Chạy model predict (NGOÀI LOCK)
                try:
                    hybrid_outputs = self.hybrid_model(current_batch_input_dict); regime_logits = hybrid_outputs.get('regime_logits')
                    if regime_logits is not None:
                        predicted_indices = torch.argmax(regime_logits, dim=-1).cpu().numpy()
                        for ts, pred_idx in zip(batch_end_timestamps, predicted_indices): all_regime_preds[ts] = pred_idx
                except Exception as batch_pred_e: logging.error(f"Error predicting batch {i} for {symbol} (outside lock): {batch_pred_e}")

                if (i // batch_size + 1) % 50 == 0: logging.info(f"    Processed {i + len(batch_end_timestamps)}/{num_predictions_possible} regime predictions for {symbol} (outside lock)...")

        # === Phần 4: Ghi kết quả vào self.data (BÊN TRONG LOCK NGẮN) ===
        if all_regime_preds:
            lock_acquired_write = False
            try:
                logging.debug(f"Acquiring data_lock for writing regimes ({symbol})...")
                await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                lock_acquired_write = True
                logging.debug(f"Data_lock acquired for writing regimes ({symbol}).")

                # Kiểm tra lại data tồn tại và ghi kết quả
                if symbol in self.data and tf_primary in self.data[symbol]:
                    df_primary = self.data[symbol][tf_primary] # Lấy DataFrame gốc
                    regime_series = pd.Series(all_regime_preds, name='hybrid_regime')
                    if 'hybrid_regime' not in df_primary.columns: df_primary['hybrid_regime'] = pd.NA
                    df_primary.update(regime_series) # Ghi vào DataFrame trong self.data
                    default_idx = self._get_default_regime_index()
                    df_primary['hybrid_regime'].fillna(default_idx, inplace=True)
                    try: df_primary['hybrid_regime'] = df_primary['hybrid_regime'].astype(int)
                    except ValueError: logging.error(f"Could not convert regime to int ({symbol}).")
                    logging.info(f"Updated 'hybrid_regime' column for {symbol} inside lock ({len(all_regime_preds)} predictions).")
                else:
                    logging.warning(f"Primary data for {symbol} became unavailable before writing regimes.")

            except asyncio.TimeoutError:
                logging.error(f"Timeout acquiring data_lock for writing regimes ({symbol}). Results lost.")
            except Exception as write_e:
                logging.error(f"Error writing regimes inside lock ({symbol}): {write_e}")
            finally:
                if lock_acquired_write and self.data_locks[symbol].locked():
                    self.data_locks[symbol].release(); logging.debug(f"Data_lock released after writing regimes ({symbol}).")
        else:
             logging.warning(f"No hybrid regime predictions generated for {symbol}.")
             await self._add_default_hybrid_regime_column_safe(symbol, tf_primary) # Gọi hàm an toàn

        logging.info(f"Finished calculating historical hybrid regimes for {symbol} (Optimized Lock).")

    def _add_default_hybrid_regime_column(self, symbol: str, tf_primary: str):
        """Helper thêm cột default (Cần caller giữ lock)."""
        if symbol in self.data and tf_primary in self.data[symbol]:
            df = self.data[symbol][tf_primary]
            if isinstance(df, pd.DataFrame) and 'hybrid_regime' not in df.columns:
                 default_idx = self._get_default_regime_index()
                 df['hybrid_regime'] = default_idx
                 try: df['hybrid_regime'] = df['hybrid_regime'].astype(int)
                 except ValueError: pass # Bỏ qua lỗi ép kiểu
                 logging.debug(f"Added default 'hybrid_regime' column for {symbol} {tf_primary}.")

    async def _update_latest_embeddings(self, symbols: List[str]):
        func_name = "_update_latest_embeddings"
        if not symbols: return
        if not self.decision_model:
            logging.warning(f"{func_name}: Decision model not available. Skipping embedding update.")
            return

        logging.info(f"{func_name}: Starting update for symbols: {symbols}")

        # Tạo task để tính embedding cho từng symbol song song
        calculation_tasks = [self._calculate_single_embedding(symbol) for symbol in symbols]
        results = await asyncio.gather(*calculation_tasks, return_exceptions=True)

        # Thu thập kết quả thành công
        new_embeddings_to_update: Dict[str, Optional[np.ndarray]] = {}
        failed_symbols = []
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, np.ndarray): # Giả sử helper trả về ndarray hoặc None
                new_embeddings_to_update[symbol] = result
            else:
                failed_symbols.append(symbol)
                if isinstance(result, Exception):
                    logging.error(f"{func_name}: Error calculating embedding for {symbol}: {result}", exc_info=result)
                # Trường hợp trả về None (do thiếu data, etc.) đã được log trong helper

        if failed_symbols:
            logging.warning(f"{func_name}: Failed to calculate embeddings for: {failed_symbols}")

        # === Cập nhật self.current_embeddings (TRONG LOCK) ===
        if new_embeddings_to_update: # Chỉ cập nhật nếu có kết quả mới
            acquired_lock = False
            try:
                logging.debug(f"{func_name}: Acquiring embeddings_lock to update {len(new_embeddings_to_update)} symbols...")
                await asyncio.wait_for(self.embeddings_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                acquired_lock = True
                logging.debug(f"{func_name}: embeddings_lock acquired.")

                # Ghi kết quả vào dict chia sẻ
                symbols_updated = []
                for symbol, embedding_np in new_embeddings_to_update.items():
                    if embedding_np is not None:
                        # Lưu dưới dạng numpy array (hoặc tensor nếu cần)
                        self.current_embeddings[symbol] = embedding_np
                        symbols_updated.append(symbol)
                    else:
                        # Nếu kết quả là None (dù ít khả năng xảy ra ở bước này), xóa key cũ
                        if symbol in self.current_embeddings:
                            del self.current_embeddings[symbol]

                if symbols_updated:
                    logging.info(f"{func_name}: Updated self.current_embeddings for: {symbols_updated}")

            except asyncio.TimeoutError:
                logging.error(f"{func_name}: Timeout acquiring embeddings_lock. Embeddings not updated for {list(new_embeddings_to_update.keys())}.")
            except Exception as e:
                logging.error(f"{func_name}: Error updating embeddings in lock: {e}", exc_info=True)
            finally:
                if acquired_lock and self.embeddings_lock.locked():
                    self.embeddings_lock.release()
                    logging.debug(f"{func_name}: embeddings_lock released.")
        else:
            logging.info(f"{func_name}: No new embeddings were successfully calculated to update.")


    async def _calculate_single_embedding(self, symbol: str) -> Optional[np.ndarray]:
        func_name = "_calculate_single_embedding"
        tf = CONFIG.get("primary_tf", "15m")
        seq_len = CONFIG.get("DECISION_SEQ_LEN", 60)
        input_dim = CONFIG.get("input_dim", 5) # OHLCV
        required_cols = ['open', 'high', 'low', 'close', 'volume'] # Cần cho DecisionModel
        ohlcv_data_np: Optional[np.ndarray] = None
        lock_acquired = False

        try:
            # === Đọc dữ liệu gần đây nhất (TRONG LOCK NGẮN) ===
            try:
                # logging.debug(f"{func_name} ({symbol}): Acquiring data_lock to read...")
                await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                lock_acquired = True
                # logging.debug(f"{func_name} ({symbol}): data_lock acquired.")

                if symbol not in self.data or tf not in self.data[symbol]:
                    # logging.warning(f"{func_name} ({symbol}): No {tf} data available.")
                    return None
                df = self.data[symbol][tf]
                if not isinstance(df, pd.DataFrame) or len(df) < seq_len:
                    # logging.warning(f"{func_name} ({symbol}): Not enough data points ({len(df) if isinstance(df, pd.DataFrame) else 0} < {seq_len}).")
                    return None
                if not all(col in df.columns for col in required_cols):
                    logging.warning(f"{func_name} ({symbol}): Missing required OHLCV columns.")
                    return None

                # Lấy seq_len dòng cuối cùng và xử lý NaN
                ohlcv_data_df = df[required_cols].iloc[-seq_len:].copy()
                if ohlcv_data_df.isnull().values.any():
                    ohlcv_data_df = ohlcv_data_df.ffill().bfill().fillna(0) # Fill NaN
                    if ohlcv_data_df.isnull().values.any(): # Kiểm tra lại
                        logging.error(f"{func_name} ({symbol}): Could not resolve NaNs in data slice.")
                        return None
                ohlcv_data_np = ohlcv_data_df.values # Lấy numpy array

            finally:
                if lock_acquired and self.data_locks[symbol].locked():
                    self.data_locks[symbol].release()
                    # logging.debug(f"{func_name} ({symbol}): data_lock released.")

            # === Tính toán Embedding (NGOÀI LOCK) ===
            if ohlcv_data_np is None: return None # Thoát nếu đọc data lỗi

            if ohlcv_data_np.shape != (seq_len, input_dim): # Kiểm tra shape lần cuối
                logging.error(f"{func_name} ({symbol}): Data slice shape mismatch {ohlcv_data_np.shape}, expected ({seq_len}, {input_dim}).")
                return None

            if not self.decision_model: return None # Kiểm tra lại model

            try:
                input_tensor = torch.tensor(ohlcv_data_np, dtype=torch.float32).unsqueeze(0).to(self.decision_model.device)
                self.decision_model.eval()
                with torch.no_grad():
                    outputs = self.decision_model(input_tensor)
                    embedding_tensor = outputs.get('embedding') # Giả sử key là 'embedding'

                if embedding_tensor is not None and embedding_tensor.shape == (1, EMBEDDING_DIM):
                    # Chuyển về numpy array trên CPU
                    embedding_np = embedding_tensor.squeeze(0).cpu().numpy()
                    if np.isfinite(embedding_np).all(): # Kiểm tra NaN/Inf
                        return embedding_np
                    else:
                        logging.error(f"{func_name} ({symbol}): Embedding contains NaN/Inf.")
                        return None
                else:
                    logging.error(f"{func_name} ({symbol}): Invalid embedding tensor received from model. Shape: {embedding_tensor.shape if embedding_tensor is not None else 'None'}")
                    return None
            except Exception as model_e:
                logging.error(f"{func_name} ({symbol}): Error during model inference: {model_e}", exc_info=True)
                return None

        except asyncio.TimeoutError:
            logging.error(f"{func_name} ({symbol}): Timeout acquiring data_lock.")
            return None
        except Exception as e:
            logging.error(f"{func_name} ({symbol}): Unexpected error: {e}", exc_info=True)
            return None

    def _get_default_regime_index(self) -> int:
        try:
            # Ưu tiên tìm index của 'SIDEWAYS'
            sideways_index = list(REGIME_MAP.keys())[list(REGIME_MAP.values()).index("SIDEWAYS")]
            if isinstance(sideways_index, int):
                 return sideways_index
        except (ValueError, IndexError):
             # Nếu không tìm thấy 'SIDEWAYS' hoặc lỗi, dùng index 1 hoặc 0
             if 1 in REGIME_MAP: return 1
             if 0 in REGIME_MAP: return 0
        return 1 # Fallback cuối cùng nếu mọi thứ thất bại

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

    async def check_circuit_breaker(self, df: pd.DataFrame) -> bool:
        """Kiểm tra và cập nhật trạng thái circuit breaker (có lock)."""
        lock_acquired_cb = False
        current_cb_state = False # Giá trị trả về mặc định nếu lỗi acquire lock

        try:
            # === Acquire cb_lock ngay từ đầu ===
            logging.debug("Acquiring cb_lock for check_circuit_breaker...")
            await asyncio.wait_for(self.cb_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            lock_acquired_cb = True
            logging.debug("cb_lock acquired for check_circuit_breaker.")

            # === Logic bên trong cb_lock ===
            current_cb_state = self.circuit_breaker_triggered # Đọc state hiện tại để trả về nếu lỗi

            # --- Kiểm tra dữ liệu đầu vào (vẫn trong lock) ---
            if df.empty or 'ATR' not in df.columns or len(df) < 20:
                # Không thay đổi state, chỉ trả về state hiện tại đọc được
                logging.debug("CB Check: Invalid input DataFrame. Returning current state.")
                return current_cb_state

            try:
                last_row_idx = df.index[-1]
                atr_current = df["ATR"].iloc[-1]
                atr_mean = df["ATR"].rolling(20).mean().iloc[-1]

                if pd.isna(atr_current) or pd.isna(atr_mean):
                    logging.warning("CB Check: NaN detected in ATR. Returning current state.")
                    return current_cb_state # Giữ trạng thái cũ

                threshold = CONFIG.get("circuit_breaker_threshold", 4.0)
                cooldown_minutes = CONFIG.get("circuit_breaker_cooldown", 60)

                # --- Logic Trigger ---
                if not self.circuit_breaker_triggered and atr_current > atr_mean * threshold:
                    logging.warning(f"CIRCUIT BREAKER TRIGGERED for {df.iloc[-1].name}: ATR {atr_current:.4f} > {threshold} * MeanATR {atr_mean:.4f}")
                    self.circuit_breaker_triggered = True # Ghi state CB
                    self.last_trade_time = last_row_idx # Ghi state CB

                    # <<< Lưu state rollback (cần acquire các lock khác NỒNG NHAU) >>>
                    rollback_success = await self._save_rollback_state_within_cb_lock()
                    if not rollback_success:
                        logging.error("Failed to save rollback state during CB trigger. CB might not be fully effective.")
                        # Quyết định: Vẫn trigger CB để dừng giao dịch, nhưng không có rollback
                    return True # Trả về True (vừa trigger)

                # --- Logic Reset ---
                if self.circuit_breaker_triggered and self.last_trade_time:
                    time_elapsed_minutes = float('inf') # Mặc định
                    # ... (Logic tính time_elapsed_minutes với xử lý timezone như cũ) ...
                    if isinstance(last_row_idx, pd.Timestamp) and isinstance(self.last_trade_time, pd.Timestamp):
                         try:
                              last_row_idx_aware = last_row_idx.tz_convert('UTC') if last_row_idx.tzinfo else last_row_idx.tz_localize('UTC')
                              last_trade_time_aware = self.last_trade_time.tz_convert('UTC') if self.last_trade_time.tzinfo else self.last_trade_time.tz_localize('UTC')
                              time_elapsed_minutes = (last_row_idx_aware - last_trade_time_aware).total_seconds() / 60
                         except Exception: pass # Bỏ qua nếu lỗi timezone, time_elapsed_minutes sẽ là inf
                    else: time_elapsed_minutes = cooldown_minutes + 1 # Nếu không phải timestamp

                    if atr_current <= atr_mean * threshold * 0.9 and time_elapsed_minutes >= cooldown_minutes:
                         logging.info(f"Circuit breaker resetting: ATR ({atr_current:.4f}) back below threshold and cooldown ({cooldown_minutes} min) passed.")
                         self.circuit_breaker_triggered = False # Ghi state CB
                         self.last_trade_time = None # Ghi state CB
                         self.rollback_state = None # Ghi state CB
                         return False # Trả về False (vừa reset)
                    else:
                         # Vẫn đang active, đọc lại state cuối cùng (đề phòng)
                         return self.circuit_breaker_triggered

                # Không trigger và không đang active, đọc lại state cuối cùng
                return self.circuit_breaker_triggered

            except Exception as inner_e:
                 logging.error(f"Error inside check_circuit_breaker lock: {inner_e}", exc_info=True)
                 # Trả về state đọc được ở đầu khối try nếu lỗi xảy ra bên trong
                 return current_cb_state

        except asyncio.TimeoutError:
            logging.error("Timeout acquiring cb_lock in check_circuit_breaker. Returning last known state.")
            return False # Giả định không active nếu không lấy được lock
        except Exception as outer_e:
            logging.error(f"Error acquiring cb_lock or unexpected error: {outer_e}", exc_info=True)
            return False # Giả định không active nếu lỗi acquire
        finally:
            # === Đảm bảo giải phóng lock ===
            if lock_acquired_cb and self.cb_lock.locked():
                self.cb_lock.release()
                logging.debug("cb_lock released for check_circuit_breaker.")

    def check_circuit_breaker_simple(self, df: pd.DataFrame) -> bool:
        atr_current = df["ATR"].iloc[-1]
        atr_mean = df["ATR"].rolling(20).mean().iloc[-1]
        if pd.isna(atr_current) or pd.isna(atr_mean):
            return False
        
        if atr_current > atr_mean * CONFIG["circuit_breaker_threshold"]:
            if not self.circuit_breaker_triggered:
                logging.warning(f"Circuit breaker triggered: ATR {atr_current:.2f} > {atr_mean * CONFIG['circuit_breaker_threshold']:.2f}")
                self.circuit_breaker_triggered = True
                self.last_trade_time = df.index[-1]
                self.rollback_state = {
                    "balance": self.balance,
                    "trade_history": self.trade_history.copy(),
                    "exposure_per_symbol": self.exposure_per_symbol.copy()
                }
            return True
        
        if self.circuit_breaker_triggered and self.last_trade_time:
            time_elapsed = (df.index[-1] - self.last_trade_time).total_seconds() / 60
            if time_elapsed >= CONFIG["circuit_breaker_cooldown"]:
                self.circuit_breaker_triggered = False
                self.rollback_state = None
                logging.info("Circuit breaker reset")
                return False
            return True
        return False

    async def _save_rollback_state_within_cb_lock(self) -> bool:
        """Helper lưu state rollback (caller ĐÃ giữ cb_lock)."""
        # <<< ASSERTION: Kiểm tra caller đã giữ cb_lock >>>
        if not self.cb_lock.locked():
             logging.critical("FATAL: _save_rollback_state called without holding cb_lock!")
             return False

        acquired_locks = [] # Chỉ lưu các lock con
        try:
            # Acquire các lock cần thiết theo đúng thứ tự NỒNG NHAU
            logging.debug("Acquiring locks for rollback state save...")
            acquired_balance = await asyncio.wait_for(self.balance_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT / 2) # Timeout ngắn hơn
            if not acquired_balance: raise asyncio.TimeoutError("Timeout balance_lock for rollback")
            acquired_locks.append(self.balance_lock)

            acquired_positions = await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT / 2)
            if not acquired_positions: raise asyncio.TimeoutError("Timeout positions_lock for rollback")
            acquired_locks.append(self.positions_lock)

            acquired_history = await asyncio.wait_for(self.trade_history_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT / 2)
            if not acquired_history: raise asyncio.TimeoutError("Timeout history_lock for rollback")
            acquired_locks.append(self.trade_history_lock)
            logging.debug("Locks acquired for rollback state save.")

            # === Bên trong tất cả các lock (bao gồm cb_lock của caller) ===
            self.rollback_state = { # Ghi state (được bảo vệ bởi cb_lock của caller)
                "balance": self.balance, # Đọc state (được bảo vệ bởi balance_lock)
                "equity_peak": self.equity_peak, # Đọc state (được bảo vệ bởi balance_lock)
                "trade_history_len": len(self.trade_history), # Đọc state (được bảo vệ bởi history_lock)
                "exposure_per_symbol": self.exposure_per_symbol.copy(), # Đọc state (được bảo vệ bởi balance_lock)
                "open_positions": copy.deepcopy(self.open_positions) # Đọc state (được bảo vệ bởi positions_lock)
            }
            logging.info("Rollback state saved successfully.")
            return True # Lưu thành công

        except asyncio.TimeoutError as te:
            logging.error(f"Timeout acquiring lock during rollback state save: {te}")
            return False
        except Exception as e:
            logging.error(f"Error saving rollback state: {e}", exc_info=True)
            return False
        finally:
            # Giải phóng các lock con theo thứ tự ngược lại
            for lock in reversed(acquired_locks):
                if lock.locked():
                    lock.release()
            logging.debug("Rollback save locks released.")

    def check_position_exposure(self, position_size: float, entry_price: float, symbol: str) -> bool:
        # ... (Tính toán an toàn hơn) ...
        if self.balance <= 0: return False # Không mở vị thế nếu balance âm/zero
        try:
            current_exposure_symbol = self.exposure_per_symbol.get(symbol, 0.0)
            # Giá trị danh nghĩa của vị thế mới
            new_position_notional = abs(position_size * entry_price)
            # Tỷ lệ exposure của vị thế mới so với balance hiện tại
            new_exposure_ratio = new_position_notional / self.balance

            # Exposure mới của symbol này
            projected_symbol_exposure = current_exposure_symbol + new_exposure_ratio

            # Exposure tổng cộng của tất cả các symbol (bao gồm vị thế mới)
            other_symbols_exposure = sum(exp for sym, exp in self.exposure_per_symbol.items() if sym != symbol)
            projected_total_exposure = other_symbols_exposure + projected_symbol_exposure

            max_sym_exposure = CONFIG.get("max_exposure", 0.75) # Max exposure cho 1 symbol
            # Max exposure tổng thể, có thể bằng max_sym_exposure * N hoặc một giới hạn riêng
            max_total_exposure = CONFIG.get("max_total_exposure", max_sym_exposure * len(CONFIG.get("symbols", [])))
            max_total_exposure = min(max_total_exposure, 1.5) # Giới hạn trên thực tế (ví dụ 150%)

            # Kiểm tra giới hạn
            if projected_symbol_exposure > max_sym_exposure:
                 logging.warning(f"Exposure limit EXCEEDED for {symbol}: Projected Symbol Exposure {projected_symbol_exposure:.2%} > Max {max_sym_exposure:.2%}")
                 return False
            if projected_total_exposure > max_total_exposure:
                 logging.warning(f"Exposure limit EXCEEDED for Portfolio: Projected Total Exposure {projected_total_exposure:.2%} > Max {max_total_exposure:.2%}")
                 return False

            # logging.debug(f"Exposure check OK for {symbol}: NewRatio={new_exposure_ratio:.2%}, ProjSym={projected_symbol_exposure:.2%}, ProjTotal={projected_total_exposure:.2%}")
            return True

        except Exception as e:
             logging.error(f"Error checking exposure for {symbol}: {e}", exc_info=True)
             return False # An toàn là không cho phép nếu lỗi


    def apply_slippage(self, price: float, direction: str, liquidity: float) -> float:
        # ... (Có thể làm phức tạp hơn) ...
        # Liquidity ở đây là gì? Cần định nghĩa rõ (ví dụ: tổng volume 5 mức giá gần nhất?)
        # Giả sử liquidity là một số dương, càng lớn càng tốt, max là 1.0
        liquidity_factor = max(0, 1.0 - min(liquidity, 1.0)) # 0 nếu liquidity tốt, 1 nếu liquidity tệ
        slippage_rate = CONFIG.get("base_slippage_rate", 0.0005) # Tỷ lệ slippage cơ bản
        slippage_amount = price * slippage_rate * liquidity_factor

        if direction.upper() == "BUY" or direction.upper() == "LONG":
            return price + slippage_amount
        elif direction.upper() == "SELL" or direction.upper() == "SHORT":
            return price - slippage_amount
        else:
            return price # Không áp dụng nếu hướng không rõ


    def validate_volume(self, df: pd.DataFrame) -> bool:
        if df.empty or 'volume' not in df.columns or len(df) < 20: return False
        try:
            current_volume = df["volume"].iloc[-1]
            ma20_volume = df["volume"].rolling(20).mean().iloc[-1]
            if pd.isna(current_volume) or pd.isna(ma20_volume) or ma20_volume <= 0: # Kiểm tra ma20 > 0
                 return False
            threshold = CONFIG.get("volume_ma20_threshold", 0.8)
            return current_volume > threshold * ma20_volume
        except Exception as e:
             logging.error(f"Error validating volume: {e}")
             return False


    def check_volume_anomaly(self, df: pd.DataFrame, symbol: str, signal: dict) -> bool:
        if df.empty or not signal or 'direction' not in signal or len(df) < 20: return False

        try:
            last_row = df.iloc[-1]
            mean_volume = df["volume"].rolling(20).mean().iloc[-1]
            current_volume = last_row["volume"]

            if pd.isna(mean_volume) or mean_volume <= 0 or pd.isna(current_volume):
                 return False # Không thể xác định nếu thiếu volume

            # 1. Volume Spike
            spike_multiplier = CONFIG.get("volume_spike_multiplier", 4.0)
            is_spike = current_volume > spike_multiplier * mean_volume
            if not is_spike: return False

            # 2. Price Break (VWAP/VAH) - Cần VAH, VWAP
            vwap = last_row.get("VWAP")
            vah = last_row.get("VAH") # Giả sử VAH là POC hoặc mức quan trọng
            if pd.isna(vwap) or pd.isna(vah): return False # Cần VWAP/VAH để xác nhận

            direction = signal['direction']
            price_break = (direction == "LONG" and last_row["close"] > max(vwap, vah)) or \
                          (direction == "SHORT" and last_row["close"] < min(vwap, vah))
            if not price_break: return False

            # 3. Trend Confirmation (ADX)
            adx = last_row.get("ADX")
            adx_threshold = CONFIG.get("adx_threshold", 25)
            if pd.isna(adx) or adx < adx_threshold: return False

            # 4. Volume Sustained (so với nến trước)
            sustained_multiplier = CONFIG.get("volume_sustained_multiplier", 1.5)
            prev_volume = df["volume"].iloc[-2] if len(df) >= 2 else 0
            volume_sustained = pd.notna(prev_volume) and current_volume > prev_volume * sustained_multiplier
            # Có thể nới lỏng điều kiện này nếu spike là khởi đầu của move mạnh

            # 5. Volatility Check (ATR tăng?)
            atr = last_row.get("ATR")
            atr_mean = df["ATR"].rolling(20).mean().iloc[-1]
            volatility_ok = pd.notna(atr) and pd.notna(atr_mean) and atr > atr_mean # ATR hiện tại cao hơn trung bình

            # Kết hợp các điều kiện
            result = is_spike and price_break and (adx >= adx_threshold) and volume_sustained and volatility_ok
            if result:
                logging.info(f"Volume ANOMALY CONFIRMED for {symbol} {direction}: Vol={current_volume:.0f} > {spike_multiplier}*Mean({mean_volume:.0f}), PriceBreak, ADX={adx:.1f}>={adx_threshold}")
            return result

        except Exception as e:
             logging.error(f"Error checking volume anomaly for {symbol}: {e}")
             return False
    
    async def precompute_and_cache_embeddings(self):
        # --- 1. Kiểm tra điều kiện cần thiết ---
        if not self.decision_model: # Kiểm tra xem Decision Model đã được khởi tạo chưa
            logging.info("Skipping embedding precomputation: Decision Model not available.")
            return
        if not CONFIG.get("precompute_embeddings_on_start", True): # Kiểm tra config
            logging.info("Skipping embedding precomputation: Disabled in config.")
            return

        logging.info("Starting embedding precomputation for all symbols...")
        DEVICE = self.decision_model.device # Lấy device từ model (GPU hoặc CPU)
        seq_len = CONFIG.get("DECISION_SEQ_LEN", 60) # Lấy sequence length yêu cầu của Decision Model
        input_dim = CONFIG.get("input_dim", 5) # Số features cơ bản (OHLCV)
        tf = CONFIG.get("primary_tf", "15m") # Timeframe chính để tính embedding

        # --- 2. Lặp qua từng symbol ---
        for symbol in CONFIG.get("symbols", []):
            logging.info(f"Processing embeddings for: {symbol} [{tf}]")
            symbol_data = self.data.get(symbol) # Lấy dữ liệu đã tải của symbol
            embedding_cache_file = self.embeddings_cache_dir / f"{symbol.replace('/', '_')}_{tf}_embeddings.pkl"

            # --- 3. Kiểm tra Cache và Dữ liệu ---
            if embedding_cache_file.exists():
                 logging.info(f"  Embeddings cache file already exists: {embedding_cache_file}. Skipping.")
                 continue # Bỏ qua nếu file cache đã tồn tại

            if not symbol_data or tf not in symbol_data or symbol_data[tf].empty:
                logging.warning(f"  No data found for {symbol} {tf}. Cannot compute embeddings.")
                continue
            if len(symbol_data[tf]) < seq_len:
                logging.warning(f"  Not enough data ({len(symbol_data[tf])} < {seq_len}) for {symbol} {tf} to compute embeddings.")
                continue

            # --- 4. Chuẩn bị Dữ liệu Đầu vào cho Model ---
            df = symbol_data[tf]
            # Chỉ lấy 5 cột OHLCV cần thiết cho Decision Model (dựa trên cấu hình input_dim=5)
            # Giả sử Decision Model chỉ cần OHLCV thô, không cần feature engineering phức tạp ở bước này
            try:
                # Đảm bảo đúng thứ tự cột OHLCV mà model mong đợi
                df_ohlcv = df[['open', 'high', 'low', 'close', 'volume']].copy()
            except KeyError as e:
                logging.error(f"  Missing required columns (OHLCV) in data for {symbol} {tf}: {e}. Skipping.")
                continue

            # Kiểm tra và xử lý NaN trước khi đưa vào model
            if df_ohlcv.isnull().values.any():
                nan_count = df_ohlcv.isnull().sum().sum()
                logging.warning(f"  NaNs found ({nan_count}) in OHLCV data for {symbol} {tf} before embedding. Filling with forward fill then 0.")
                # Có thể dùng phương pháp fill tốt hơn (ví dụ: ffill rồi bfill, hoặc fill bằng giá trị rolling mean)
                df_ohlcv = df_ohlcv.ffill().fillna(0) # Fill NaN bằng 0 sau khi ffill
                if df_ohlcv.isnull().values.any(): # Kiểm tra lại sau khi fill
                     logging.error(f"  Could not fully resolve NaNs in OHLCV data for {symbol} {tf}. Skipping.")
                     continue

            # Chuyển DataFrame thành NumPy array để xử lý cửa sổ trượt
            ohlcv_values = df_ohlcv.values

            # --- 5. Tính Embeddings bằng Cửa sổ Trượt ---
            all_embeddings = {} # Dictionary để lưu {timestamp: embedding_numpy}
            self.decision_model.eval() # Đảm bảo model ở chế độ đánh giá

            batch_size = 128 # Xử lý theo batch để tăng tốc trên GPU
            num_samples = len(ohlcv_values) - seq_len + 1
            logging.info(f"  Calculating embeddings for {num_samples} windows (Batch Size: {batch_size})...")

            # Không cần gradient khi chỉ tính embedding
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    batch_end = min(i + batch_size, num_samples)
                    batch_windows_np = []

                    # Tạo batch dữ liệu cửa sổ
                    for j in range(i, batch_end):
                        window_data = ohlcv_values[j : j + seq_len]
                        batch_windows_np.append(window_data)

                    if not batch_windows_np: continue # Bỏ qua batch rỗng

                    # Chuyển batch sang tensor
                    input_batch_tensor = torch.tensor(np.array(batch_windows_np), dtype=torch.float32).to(DEVICE)

                    try:
                        # Chạy model trên batch
                        outputs = self.decision_model(input_batch_tensor)
                        batch_embeddings = outputs.get('embedding') # Shape: (batch_size, embedding_dim)

                        if batch_embeddings is not None:
                            # Lấy timestamps tương ứng với điểm cuối của mỗi cửa sổ trong batch
                            batch_indices_range = range(i, batch_end) # range object
                            # Tạo list các integer index cần lấy timestamp
                            timestamp_integer_indices = [idx + seq_len - 1 for idx in batch_indices_range]

                            # Kiểm tra xem các index có nằm trong giới hạn của df.index không
                            max_allowable_index = len(df.index) - 1
                            valid_integer_indices = [idx for idx in timestamp_integer_indices if 0 <= idx <= max_allowable_index]

                            if len(valid_integer_indices) != len(batch_indices_range):
                                logging.warning(f"  Mismatch between expected batch size and valid indices for timestamps (Symbol: {symbol}, Batch Start: {i}). Some embeddings might be skipped.")
                                # Chỉ lấy embedding tương ứng với các index hợp lệ
                                valid_embedding_indices = [k for k, original_idx in enumerate(batch_indices_range)
                                                        if (original_idx + seq_len - 1) in valid_integer_indices]
                                if not valid_embedding_indices:
                                    logging.error(f"  No valid timestamps found for batch starting at index {i}. Skipping batch embeddings.")
                                    continue # Bỏ qua batch này nếu không có timestamp hợp lệ

                                batch_timestamps = df.index[valid_integer_indices]
                                batch_embeddings_cpu = batch_embeddings[valid_embedding_indices].cpu().numpy() # Lấy đúng embedding
                            else:
                                # Nếu tất cả index đều hợp lệ
                                batch_timestamps = df.index[valid_integer_indices] # Dùng list các integer index
                                batch_embeddings_cpu = batch_embeddings.cpu().numpy()

                            if len(batch_timestamps) == batch_embeddings_cpu.shape[0]:
                                for k, timestamp in enumerate(batch_timestamps):
                                    all_embeddings[timestamp] = batch_embeddings_cpu[k]
                            else:
                                logging.error(f"  Timestamp count ({len(batch_timestamps)}) mismatch with embedding count ({batch_embeddings_cpu.shape[0]}) for batch starting at index {i}. Skipping saving.")

                        else:
                            logging.warning(f"  Decision model did not return 'embedding' for batch starting at index {i}.")

                    except Exception as e:
                        logging.error(f"  Error computing embedding batch for {symbol} starting at index {i}: {e}")
                        # Có thể chọn dừng hoặc tiếp tục với các batch khác

                    # Log tiến độ (tùy chọn)
                    if (i // batch_size + 1) % 50 == 0: # Log mỗi 50 batch
                         logging.info(f"    Processed {batch_end}/{num_samples} windows for {symbol}...")

            # --- 6. Lưu Embeddings vào Cache ---
            if all_embeddings:
                try:
                    # Tạo thư mục cache nếu chưa có
                    self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(all_embeddings, embedding_cache_file)
                    logging.info(f"  Saved {len(all_embeddings)} embeddings to {embedding_cache_file}")
                except Exception as e:
                    logging.error(f"  Failed to save embeddings cache for {symbol} {tf}: {e}")
            else:
                 logging.warning(f"  No embeddings were successfully computed for {symbol} {tf}.")

        logging.info("Embedding precomputation finished.")
        
    def prepare_ml_data(self, df: pd.DataFrame, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if df.empty or 'ATR' not in df.columns or 'close' not in df.columns:
             logging.error(f"prepare_ml_data ({symbol}): DataFrame rỗng hoặc thiếu cột 'ATR'/'close'.")
             return None, None
        df = df.copy()
        tf = CONFIG.get("primary_tf", "15m") # Giả sử dùng timeframe chính
        embedding_cache_file = self.embeddings_cache_dir / f"{symbol.replace('/', '_')}_{tf}_embeddings.pkl"
        embeddings_dict = None

        # --- 1. Tải Embeddings đã tính toán trước ---
        if embedding_cache_file.exists():
            try:
                embeddings_dict = joblib.load(embedding_cache_file)
                logging.info(f"Loaded {len(embeddings_dict)} precomputed embeddings for {symbol} {tf}.")
            except Exception as e:
                logging.error(f"Error loading embeddings cache for {symbol} {tf}: {e}. Proceeding without embeddings.")
                embeddings_dict = None
        else:
            logging.warning(f"Embeddings cache file not found for {symbol} {tf}. Proceeding without embeddings.")

        # --- 2. Tính Target ---
        try:
            # Threshold động 
            mean_close = df["close"].mean(); mean_atr = df["ATR"].mean()
            threshold = max(0.002, (mean_atr / mean_close) * 2) if pd.notna(mean_close) and mean_close != 0 and pd.notna(mean_atr) else 0.002
            future_period = 4
            df["future_return"] = df["close"].pct_change(future_period).shift(-future_period)
            df["target"] = np.select([df["future_return"] > threshold, df["future_return"] < -threshold], [1, 0], default=-1)
            df_binary = df[df["target"] != -1].copy()
            min_samples = CONFIG.get("min_samples", 150)
            if len(df_binary) < min_samples: return None, None
        except Exception as e: logging.error(f"Error calculating target in prepare_ml_data ({symbol}): {e}"); return None, None

        # --- 3. Chọn Features Gốc và Temporal ---
        features_base = CONFIG.get("xgboost_feature_columns", [])
        temporal_features_exist = [feat for feat in self.temporal_features if feat in df_binary.columns]
        features_all_no_emb = features_base + temporal_features_exist
        features_base.append('hybrid_regime')
        missing_ml_features = [f for f in features_all_no_emb if f not in df_binary.columns]
        if missing_ml_features: logging.error(f"Missing base/temporal features for XGBoost ({symbol}): {missing_ml_features}"); return None, None
        X_raw_no_emb = df_binary[features_all_no_emb]
        y_raw = df_binary["target"]

        # --- 4. Thêm Embeddings làm Features ---
        if embeddings_dict:
            # Tạo DataFrame từ embeddings dict
            embeddings_df = pd.DataFrame.from_dict(embeddings_dict, orient='index',
                                                   columns=[f'emb_{i}' for i in range(EMBEDDING_DIM)])
            embeddings_df.index = pd.to_datetime(embeddings_df.index, utc=True) # Đảm bảo index là datetime

            # Merge embeddings vào X_raw_no_emb dựa trên index (timestamp)
            X_raw_with_emb = X_raw_no_emb.join(embeddings_df, how='left')

            # Kiểm tra xem có dòng nào không join được embedding không
            rows_without_embedding = X_raw_with_emb[f'emb_0'].isnull().sum()
            if rows_without_embedding > 0:
                logging.warning(f"{rows_without_embedding}/{len(X_raw_with_emb)} rows could not be joined with embeddings for {symbol} {tf}. Filling with 0.")
                # Fill NaN trong các cột embedding bằng 0
                emb_cols = [f'emb_{i}' for i in range(EMBEDDING_DIM)]
                X_raw_with_emb[emb_cols] = X_raw_with_emb[emb_cols].fillna(0)

            # Xử lý NaN trong features gốc và target *sau khi join*
            valid_indices = X_raw_with_emb.dropna().index.intersection(y_raw.dropna().index)
            if len(valid_indices) < min_samples: return None, None
            X = X_raw_with_emb.loc[valid_indices]
            y = y_raw.loc[valid_indices]
            logging.info(f"Added {EMBEDDING_DIM} embedding features for {symbol}. Total features: {X.shape[1]}")
        else:
            # Xử lý NaN nếu không có embedding
            valid_indices = X_raw_no_emb.dropna().index.intersection(y_raw.dropna().index)
            if len(valid_indices) < min_samples: return None, None
            X = X_raw_no_emb.loc[valid_indices]
            y = y_raw.loc[valid_indices]
            logging.info(f"Proceeding without embedding features for {symbol}.")

        # --- 5. Oversampling & Scaling ---
        try:
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X, y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_res)
            self.scalers[symbol] = scaler # Lưu scaler (đã fit trên data có/không có embedding)
            logging.info(f"Prepared ML data for {symbol}: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features. Columns (first 5): {list(X.columns[:5])}")
            return X_scaled, y_res
        except Exception as e: logging.error(f"Error in oversampling/scaling for {symbol}: {e}"); return None, None

    def train_model(self, symbol: str, X: np.ndarray, y: np.ndarray):
        # ... (Tăng cường logging, xử lý lỗi BayesianOptimization) ...
        if X is None or y is None or X.shape[0] < CONFIG.get("min_samples", 50):
            logging.error(f"Dữ liệu không đủ hoặc không hợp lệ để huấn luyện XGBoost cho {symbol}")
            return

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Thêm stratify

            # Hàm mục tiêu cho Bayesian Optimization
            def xgb_bayes_opt(n_estimators, learning_rate, max_depth, subsample, colsample_bytree, gamma, min_child_weight):
                params = {
                    "objective": "binary:logistic", # Chỉ định mục tiêu nhị phân
                    "eval_metric": "logloss", # Hoặc auc
                    "eta": learning_rate, # learning_rate
                    "max_depth": int(max_depth),
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "gamma": gamma,
                    "min_child_weight": int(min_child_weight),
                    "n_estimators": int(n_estimators), # Thêm n_estimators vào đây
                    "seed": 42
                }
                # Sử dụng early stopping để tránh overfitting và tìm số cây tối ưu
                model = XGBClassifier(**params)
                eval_set = [(X_test, y_test)]
                model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=20, verbose=False)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                # Trả về ROC AUC làm mục tiêu tối ưu hóa (phù hợp hơn F1 cho bài toán xác suất)
                return roc_auc_score(y_test, y_pred_proba)

            # Định nghĩa không gian tìm kiếm mở rộng hơn
            pbounds = {
                 "n_estimators": (50, 400),
                 "learning_rate": (0.005, 0.2),
                 "max_depth": (3, 12),
                 "subsample": (0.6, 1.0),
                 "colsample_bytree": (0.6, 1.0),
                 "gamma": (0, 0.5),
                 "min_child_weight": (1, 10)
            }

            optimizer = BayesianOptimization(f=xgb_bayes_opt, pbounds=pbounds, random_state=42, allow_duplicate_points=True) # Cho phép trùng lặp

            # Chạy tối ưu hóa
            try:
                 optimizer.maximize(init_points=5, n_iter=15) # Tăng số lần thử
                 best_params = optimizer.max["params"]
            except Exception as opt_e: # Xử lý lỗi nếu optimizer không tìm thấy kết quả tốt
                 logging.error(f"Bayesian Optimization failed for {symbol}: {opt_e}. Using default XGBoost parameters.")
                 best_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0, "min_child_weight": 1} # Params mặc định an toàn

            # Chuẩn hóa lại kiểu dữ liệu tham số
            best_params["n_estimators"] = int(best_params["n_estimators"])
            best_params["max_depth"] = int(best_params["max_depth"])
            best_params["min_child_weight"] = int(best_params["min_child_weight"])

            # Huấn luyện mô hình cuối cùng với tham số tốt nhất và early stopping
            logging.info(f"Training final XGBoost model for {symbol} with best params: {best_params}")
            final_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", seed=42, **best_params)
            eval_set = [(X_test, y_test)]
            final_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=20, verbose=False)

            # Đánh giá trên tập test
            y_pred = final_model.predict(X_test)
            y_prob = final_model.predict_proba(X_test)[:, 1]

            logging.info(f"--- XGBoost Evaluation Report for {symbol} ---")
            try:
                 report = classification_report(y_test, y_pred, target_names=["Short", "Long"])
                 logging.info(f"\n{report}")
            except ValueError: # Có thể xảy ra nếu chỉ có 1 lớp trong y_pred/y_test
                 logging.warning("Could not generate full classification report (possibly only one class predicted).")
                 logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            logging.info(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
            logging.info("---------------------------------------------")

            self.models[symbol] = final_model
            # Lưu model và scaler
            model_filename = f"{symbol.replace('/', '_')}_xgb_model.pkl"
            scaler_filename = f"{symbol.replace('/', '_')}_xgb_scaler.pkl"
            joblib.dump(final_model, model_filename)
            joblib.dump(self.scalers[symbol], scaler_filename)
            logging.info(f"Saved XGBoost model to {model_filename} and scaler to {scaler_filename}")

        except Exception as e:
             logging.error(f"Error training XGBoost model for {symbol}: {e}", exc_info=True)


    def walk_forward_validation(self, data: Dict[str, pd.DataFrame], symbol: str):
        # ... (Tăng cường kiểm tra dữ liệu) ...
        if "15m" not in data or not isinstance(data["15m"], pd.DataFrame):
             logging.error(f"Walk-forward ({symbol}): Missing or invalid '15m' data.")
             return
        full_data = data["15m"]
        n = len(full_data)
        # Điều chỉnh kích thước train/test và step
        train_ratio = 0.6
        test_ratio = 0.2
        step_ratio = 0.1 # Bước nhảy là 10% tổng dữ liệu
        min_train_size = CONFIG.get("min_samples", 50) + CONFIG.get("hmm_warmup_max", 200) # Cần đủ cho chỉ báo và ML
        min_test_size = 50 # Số mẫu tối thiểu để test có ý nghĩa

        train_size = int(n * train_ratio)
        test_size = int(n * test_ratio)
        step_size = int(n * step_ratio)

        if n < min_train_size + min_test_size:
            logging.error(f"Walk-forward ({symbol}): Not enough data (n={n}) for initial train/test split (min_train={min_train_size}, min_test={min_test_size}).")
            return
        if step_size < 1: step_size = 1 # Đảm bảo bước nhảy ít nhất là 1

        self.walk_forward_windows = [] # Reset kết quả trước khi chạy
        num_folds = 0

        for i in range(0, n - train_size - test_size + 1, step_size):
            train_start = i
            train_end = i + train_size
            test_start = train_end
            test_end = test_start + test_size

            # Đảm bảo index không vượt quá giới hạn
            if train_end > n or test_end > n: break

            train_df = full_data.iloc[train_start:train_end]
            test_df = full_data.iloc[test_start:test_end]

            # Kiểm tra kích thước fold
            if len(train_df) < min_train_size:
                logging.warning(f"Walk-forward ({symbol}): Fold {num_folds+1} - Train data too small ({len(train_df)} < {min_train_size}). Skipping.")
                continue
            if len(test_df) < min_test_size:
                logging.warning(f"Walk-forward ({symbol}): Fold {num_folds+1} - Test data too small ({len(test_df)} < {min_test_size}). Skipping.")
                continue

            logging.info(f"--- Walk-Forward Fold {num_folds+1} for {symbol} ---")
            logging.info(f"Train: {train_df.index[0]} -> {train_df.index[-1]} ({len(train_df)} rows)")
            logging.info(f"Test:  {test_df.index[0]} -> {test_df.index[-1]} ({len(test_df)} rows)")

            # Chuẩn bị dữ liệu cho fold hiện tại
            # Quan trọng: KHÔNG tính lại chỉ báo ở đây vì sẽ gây lookahead bias. Dùng chỉ báo đã có.
            X_train, y_train = self.prepare_ml_data(train_df, symbol + f"_wf_train_{num_folds+1}") # Dùng tên riêng cho scaler
            X_test, y_test = self.prepare_ml_data(test_df, symbol + f"_wf_test_{num_folds+1}") # Dùng tên riêng cho scaler

            if X_train is None or y_train is None or X_test is None or y_test is None:
                logging.warning(f"Walk-forward ({symbol}): Fold {num_folds+1} - Failed to prepare data. Skipping.")
                continue
            try:
                 temp_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", seed=42, n_estimators=100) # Model đơn giản
                 temp_model.fit(X_train, y_train) # Không dùng early stopping ở đây để đơn giản
                 fold_report = self.evaluate_model(temp_model, X_test, y_test) # Đánh giá trên tập test
                 logging.info(f"Fold {num_folds+1} Performance: {fold_report}")
                 self.walk_forward_windows.append({
                      'fold': num_folds + 1,
                      'train_start': train_df.index[0],
                      'train_end': train_df.index[-1],
                      'test_start': test_df.index[0],
                      'test_end': test_df.index[-1],
                      'performance': fold_report
                 })
                 num_folds += 1
            except Exception as wf_train_e:
                 logging.error(f"Error training/evaluating model in Walk-forward Fold {num_folds+1} for {symbol}: {wf_train_e}")
                 continue

        logging.info(f"Completed Walk-Forward Validation for {symbol} with {num_folds} folds.")
        # Optional: Tính trung bình hiệu suất qua các fold
        if self.walk_forward_windows:
             avg_perf = pd.DataFrame([w['performance'] for w in self.walk_forward_windows]).mean().to_dict()
             logging.info(f"Average Walk-Forward Performance for {symbol}: {avg_perf}")


    def evaluate_model(self, model, X_test, y_test) -> dict:
        # ... (Xử lý lỗi nếu y_test chỉ có 1 lớp) ...
        try:
             y_pred = model.predict(X_test)
             y_prob = model.predict_proba(X_test)[:, 1]
             # Kiểm tra số lớp duy nhất trong y_test và y_pred
             unique_y_test = np.unique(y_test)
             unique_y_pred = np.unique(y_pred)

             report = { 'accuracy': accuracy_score(y_test, y_pred), 'roc_auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan }

             # ROC AUC chỉ tính được nếu có cả 2 lớp trong y_test
             if len(unique_y_test) > 1:
                  report['roc_auc'] = roc_auc_score(y_test, y_prob)

             # Precision, Recall, F1 tính được nếu có lớp dương (1) trong cả y_test và y_pred
             if 1 in unique_y_test and 1 in unique_y_pred:
                  # specifying pos_label=1 might not be needed if binary, but safer
                  # zero_division=0 prevents warnings/errors if no positive predictions made
                  report['precision'] = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
                  report['recall'] = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
                  report['f1'] = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
             # Có thể tính riêng cho lớp 0 nếu cần
             # report['precision_0'] = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
             # ...

             return report
        except Exception as e:
             logging.error(f"Error evaluating model: {e}")
             return { 'accuracy': np.nan, 'roc_auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan }


    async def train_drl(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        # ... (Kiểm tra dữ liệu, xử lý lỗi Optuna/SB3) ...
        logging.info("Starting DRL (DQN - Size Adjustment) training with Optuna...")

        # Chọn symbol đầu tiên có đủ dữ liệu để tạo Env
        first_valid_symbol = next((s for s in CONFIG.get("symbols", []) if s in data and "15m" in data[s] and not data[s]["15m"].empty), None)
        if not first_valid_symbol:
             logging.error("No valid symbol data found to initialize TradingEnv for DQN training.")
             return

        try:
            # Hàm tạo môi trường, đảm bảo truyền dữ liệu đúng
            def make_env():
                 # Cần deep copy dữ liệu để mỗi env có bản sao riêng? Không cần thiết với DummyVecEnv(n_envs=1)
                 env = TradingEnv(self, data[first_valid_symbol])
                 # Monitor để theo dõi reward/thông tin tập luyện
                 env = Monitor(env)
                 return env

            # Sử dụng DummyVecEnv cho một môi trường
            vec_env = DummyVecEnv([make_env])

        except Exception as env_e:
             logging.error(f"Failed to create TradingEnv for DQN training: {env_e}", exc_info=True)
             return

        best_model_path = "best_drl_dqn_model.zip" # Đổi tên file
        best_trial_value = -float("inf") # Dùng tên rõ ràng hơn

        # --- Hàm mục tiêu Optuna ---
        def objective(trial):
            nonlocal best_trial_value # Cho phép sửa đổi biến ngoài hàm
            try:
                # Đề xuất siêu tham số
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # Dùng categorical
                gamma = trial.suggest_float("gamma", 0.95, 0.999)
                target_update = trial.suggest_int("target_update_interval", 100, 1000, step=50) # Dùng tên đúng
                buffer_size = trial.suggest_int("buffer_size", 10000, 100000)
                exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.3)
                exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
                train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
                gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2]) # -1 có thể không ổn định

                # Tạo model DQN với các tham số đề xuất
                model = DQN(
                    "MlpPolicy",
                    vec_env, # Sử dụng vec_env
                    verbose=0,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=max(1000, buffer_size // 10), # Đảm bảo learning_starts hợp lý
                    batch_size=batch_size,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    target_update_interval=target_update,
                    exploration_fraction=exploration_fraction,
                    exploration_final_eps=exploration_final_eps,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    seed=42 # Seed để tái lập
                )

                # Huấn luyện trong thời gian ngắn để đánh giá siêu tham số
                train_steps_trial = 10000 # Tăng số bước huấn luyện thử nghiệm
                self.trade_history = [] # Reset history trước mỗi trial
                self.balance = CONFIG.get("initial_balance", 10000) # Reset balance
                model.learn(total_timesteps=train_steps_trial, reset_num_timesteps=False) # Không reset timestep

                # Đánh giá trial dựa trên PnL trung bình gần đây hoặc Sharpe Ratio
                trial_value = -float("inf")
                if self.trade_history:
                     recent_trades = self.trade_history[-20:] # Lấy 20 trades cuối
                     if recent_trades:
                          pnls = [t['pnl'] for t in recent_trades]
                          avg_pnl = np.mean(pnls)
                          std_pnl = np.std(pnls)
                          # Sharpe Ratio (giả định risk-free rate = 0)
                          sharpe = avg_pnl / std_pnl if std_pnl > 1e-6 else avg_pnl # Tránh chia cho 0
                          trial_value = sharpe # Dùng Sharpe Ratio làm mục tiêu
                          logging.info(f"Trial {trial.number}: Params={trial.params}, AvgPnL={avg_pnl:.4f}, Sharpe={sharpe:.4f}")
                     else: logging.info(f"Trial {trial.number}: No trades executed.")
                else: logging.info(f"Trial {trial.number}: No trade history.")


                # Lưu model tốt nhất dựa trên trial_value
                if trial_value > best_trial_value:
                    best_trial_value = trial_value
                    model.save(best_model_path) # Lưu model SB3 bằng save
                    logging.info(f"Trial {trial.number}: New best model saved with Sharpe={trial_value:.4f}")

                # Pruning (Optional): Dừng sớm các trial không hứa hẹn
                trial.report(trial_value, step=train_steps_trial)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return trial_value # Trả về giá trị mục tiêu (Sharpe)

            except Exception as e:
                logging.error(f"Error in DQN Optuna trial {trial.number}: {e}", exc_info=True)
                return -float("inf") # Trả về giá trị tệ nếu lỗi

        # --- Chạy Optuna ---
        try:
             loop = asyncio.get_event_loop()
             study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner()) # Thêm pruner
             # Chạy optimize trong executor để không block event loop
             await loop.run_in_executor(None, lambda: study.optimize(objective, n_trials=30, n_jobs=1)) # Tăng số trial

             # Kiểm tra xem có kết quả tốt nhất không
             if study.best_trial is None or study.best_value <= -float('inf'):
                  logging.warning("Optuna finished without finding any valid trials. Using default DQN parameters.")
                  # Xác định default params an toàn
                  best_params = {'learning_rate': 5e-4, 'batch_size': 64, 'gamma': 0.99, 'target_update_interval': 500, 'buffer_size': 50000, 'exploration_fraction': 0.15, 'exploration_final_eps': 0.05, 'train_freq': 4, 'gradient_steps': 1}
             else:
                  best_params = study.best_params
                  logging.info(f"Optuna finished. Best DQN parameters found: {best_params} with value: {study.best_value:.4f}")

             # Huấn luyện model cuối cùng với params tốt nhất
             final_model_save_path = "final_drl_dqn_model"
             if os.path.exists(best_model_path) and study.best_trial is not None: # Tải model tốt nhất từ trial nếu có
                  logging.info(f"Loading best model from trial {study.best_trial.number} ({best_model_path}) for final training/saving.")
                  self.drl_model = DQN.load(best_model_path, env=vec_env) # Load model SB3
                  # Optional: Tiếp tục huấn luyện thêm một chút
                  logging.info("Continuing training on the best trial model...")
                  await loop.run_in_executor(None, lambda: self.drl_model.learn(total_timesteps=5000, reset_num_timesteps=False))
             else: 
                  logging.info("Training final DQN model from scratch with best Optuna parameters...")
                  self.drl_model = DQN(
                       "MlpPolicy", vec_env, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu", seed=42,
                       learning_rate=best_params['learning_rate'],
                       buffer_size=best_params['buffer_size'],
                       batch_size=best_params['batch_size'],
                       gamma=best_params['gamma'],
                       train_freq=best_params.get('train_freq', 4), # Dùng get với default
                       gradient_steps=best_params.get('gradient_steps', 1),
                       target_update_interval=best_params['target_update_interval'],
                       exploration_fraction=best_params.get('exploration_fraction', 0.1),
                       exploration_final_eps=best_params.get('exploration_final_eps', 0.05),
                       learning_starts=max(1000, best_params['buffer_size'] // 10)
                  )
                  total_final_steps = 50000
                  await loop.run_in_executor(None, lambda: self.drl_model.learn(total_timesteps=total_final_steps))

             self.drl_model.save(final_model_save_path)
             logging.info(f"Completed training DRL (DQN) and saved to {final_model_save_path}.zip")

        except Exception as drl_e:
             logging.error(f"Critical error during DRL (DQN) training phase: {drl_e}", exc_info=True)
             self.drl_model = None # Đặt là None nếu huấn luyện lỗi


    async def train_entry_optimizer(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        # ... (Kiểm tra dữ liệu, xử lý lỗi) ...
        logging.info("Starting Entry Optimizer (SAC) training...")

        # Chọn symbol đầu tiên có đủ dữ liệu
        first_valid_symbol = next((s for s in CONFIG.get("symbols", []) if s in data and "15m" in data[s] and not data[s]["15m"].empty), None)
        if not first_valid_symbol:
             logging.error("No valid symbol data found to initialize MarketEnvironment for SAC training.")
             return

        # Lấy dữ liệu order book (có thể dùng dữ liệu fetch ban đầu hoặc fetch lại)
        order_book_data_sac = []
        ob = await self.fetch_order_book(first_valid_symbol, limit=20)
        if ob: order_book_data_sac.append(ob)
        else: order_book_data_sac.append({"bids": [], "asks": []}) # Dùng rỗng nếu lỗi

        try:
             # Khởi tạo hoặc sử dụng EntryPointOptimizer hiện có
             if self.entry_optimizer is None:
                  logging.info(f"Initializing EntryPointOptimizer for SAC training using data from {first_valid_symbol}.")
                  self.entry_optimizer = EntryPointOptimizer(self, data[first_valid_symbol], order_book_data_sac)
             else:
                  logging.info("Using existing EntryPointOptimizer instance for SAC training.")
                  # Có thể cần cập nhật data/env bên trong optimizer nếu cần thiết
                  # self.entry_optimizer.update_data(data[first_valid_symbol], order_book_data_sac) # Giả sử có hàm update

             # Chạy huấn luyện (hàm train_entry_model đã bao gồm Optuna)
             await self.entry_optimizer.train_entry_model(n_trials=20, total_timesteps=30000) # Tăng số bước

             if not self.entry_optimizer.is_trained():
                  logging.error("EntryPointOptimizer (SAC) training failed or was interrupted. Entry signals may be unavailable.")
                  self.entry_optimizer = None # Đặt lại nếu huấn luyện không thành công
             else:
                  logging.info("Completed training EntryPointOptimizer (SAC).")

        except Exception as sac_train_e:
             logging.error(f"Critical error during Entry Optimizer (SAC) training phase: {sac_train_e}", exc_info=True)
             self.entry_optimizer = None # Đặt lại nếu lỗi

    def _get_sac_state_unscaled(self, symbol: str,
                                current_data: Optional[Dict[str, pd.DataFrame]] = None,
                                current_step_index: int = -1) -> Optional[np.ndarray]:
        # --- Lấy dữ liệu 15m ---
        data_source = current_data if current_data is not None else self.data.get(symbol)
        tf = CONFIG.get("primary_tf", "15m")

        if not data_source or tf not in data_source or data_source[tf].empty:
            logging.warning(f"_get_sac_state_unscaled: Missing or empty {tf} data for {symbol}.")
            return None
        df_tf = data_source[tf]

        actual_index = len(df_tf) - 1 if current_step_index == -1 else current_step_index
        if not (0 <= actual_index < len(df_tf)):
            logging.error(f"_get_sac_state_unscaled: Calculated index {actual_index} is out of bounds for {symbol} (DataFrame length: {len(df_tf)}).")
            return None

        try:
            last_row = df_tf.iloc[actual_index]
            timestamp_for_obs = pd.to_datetime(last_row.name)
            if pd.isna(timestamp_for_obs):
                 logging.warning(f"_get_sac_state_unscaled: Invalid timestamp (NaT) at index {actual_index} for {symbol}.")
                 return None
        except IndexError:
            logging.error(f"_get_sac_state_unscaled: Unexpected IndexError accessing index {actual_index} for {symbol} (DataFrame length: {len(df_tf)}).")
            return None
        except Exception as e:
            logging.error(f"Error getting last_row/timestamp for SAC state at index {actual_index} for {symbol}: {e}", exc_info=True)
            return None

        # --- Tính toán Dimension kỳ vọng (giống MarketEnvironment) ---
        try:
            dim_base_features = 4
            dim_ob_extra = 2
            lob_d_model = CONFIG.get("lob_d_model", 64)
            # <<< Phần lấy lob_d_model từ env giữ nguyên >>>
            if hasattr(self, 'entry_optimizer') and self.entry_optimizer and \
               hasattr(self.entry_optimizer, 'env') and self.entry_optimizer.env and \
               hasattr(self.entry_optimizer.env, 'envs') and self.entry_optimizer.env.envs:
                 monitor_env = self.entry_optimizer.env.envs[0]
                 base_env = getattr(monitor_env, 'env', monitor_env)
                 if hasattr(base_env, 'order_book_analyzer') and base_env.order_book_analyzer:
                      lob_d_model = base_env.order_book_analyzer.d_model
            dim_lob_transformer = lob_d_model
            dim_price_action = 1
            dim_additional = 2
            dim_sentiment = 1
            expected_dim = (dim_base_features + dim_ob_extra + dim_lob_transformer +
                            dim_price_action + dim_additional + dim_sentiment)
            # logging.debug(f"SAC Expected Dim: {expected_dim}") # DEBUG
        except Exception as e:
             logging.error(f"Error calculating expected dimension in _get_sac_state_unscaled: {e}")
             return None

        # --- Lấy các features cơ bản ---
        required_base_cols = ["close", "RSI", "ATR", "volatility", "momentum", "order_imbalance"]
        obs_values = {}
        for col in required_base_cols:
            value = last_row.get(col)
            obs_values[col] = float(value) if pd.notna(value) else 0.0
        base_features = np.array([obs_values["close"], obs_values["RSI"], obs_values["ATR"], obs_values["volatility"]], dtype=np.float32)

        # --- Tính Spread và Liquidity ---
        bid_ask_spread = 0.0; liquidity = 0.0
        order_book = data_source.get("order_book", {"bids": [], "asks": []})
        try:
            if order_book and order_book.get("bids") and order_book.get("asks"):
                bids = order_book["bids"]; asks = order_book["asks"]
                if bids and asks and isinstance(bids[0],(list, tuple)) and len(bids[0]) >= 2 and \
                   isinstance(asks[0],(list, tuple)) and len(asks[0]) >= 2:
                    best_bid_price = float(bids[0][0]); best_ask_price = float(asks[0][0])
                    if best_bid_price > 0: bid_ask_spread = (best_ask_price - best_bid_price) / best_bid_price
                    bids_vol = sum(float(bid[1]) for bid in bids[:5] if len(bid)==2); asks_vol = sum(float(ask[1]) for ask in asks[:5] if len(ask)==2)
                    liquidity = bids_vol + asks_vol
        except Exception: pass
        ob_extra_features = np.array([bid_ask_spread, liquidity], dtype=np.float32)

        # --- Lấy Features từ LOB Transformer ---
        lob_features = np.zeros(lob_d_model, dtype=np.float32)
        # <<< Phần lấy lob_features giữ nguyên >>>
        try:
            if hasattr(self, 'entry_optimizer') and self.entry_optimizer and \
               hasattr(self.entry_optimizer, 'env') and self.entry_optimizer.env and \
               hasattr(self.entry_optimizer.env, 'envs') and self.entry_optimizer.env.envs:
                monitor_env = self.entry_optimizer.env.envs[0]; base_env = getattr(monitor_env, 'env', monitor_env)
                if hasattr(base_env, 'order_book_analyzer') and base_env.order_book_analyzer:
                    order_book_analyzer = base_env.order_book_analyzer
                    if order_book: lob_features = order_book_analyzer.transform(order_book)
                    if lob_features.shape != (lob_d_model,): lob_features = np.zeros(lob_d_model, dtype=np.float32)
        except Exception as tf_e: logging.error(f"Error getting LOB features for SAC state: {tf_e}", exc_info=True); lob_features = np.zeros(lob_d_model, dtype=np.float32)

        # --- Lấy Features từ MLPAction ---
        mlp_input_feature_names = ["RSI", "ATR", "EMA_diff", "MACD", "MACD_signal", "ADX", "momentum", "log_volume", "VWAP", "BB_width", "VAH", "regime", "hour", "VWAP_ADX_interaction", "BB_EMA_sync", "volume_anomaly"]
        mlp_input_list_raw = []; valid_mlp_input = True
        for feat_name in mlp_input_feature_names:
            val = last_row.get(feat_name)
            mlp_input_list_raw.append(0.0 if pd.isna(val) else float(val))
        mlp_input_numpy_raw = np.array(mlp_input_list_raw, dtype=np.float32)
        mlp_action_feature_val = self.mlp_action_fallback
        # <<< Phần gọi MLPAction predict giữ nguyên >>>
        if self.mlp_action_scaler and self.mlp_action_analyzer:
            try:
                mlp_input_numpy_scaled = self.mlp_action_scaler.transform(mlp_input_numpy_raw.reshape(1, -1)).flatten()
                if np.isfinite(mlp_input_numpy_scaled).all():
                    pa_output_numpy = self.mlp_action_analyzer.predict(mlp_input_numpy_scaled, fallback_value=self.mlp_action_fallback)
                    mlp_action_feature_val = pa_output_numpy.item() if pa_output_numpy is not None and pa_output_numpy.size > 0 else self.mlp_action_fallback
            except Exception as pa_e: logging.error(f"Error scaling/predicting MLP: {pa_e}")
        mlp_output_feature = np.array([np.nan_to_num(mlp_action_feature_val)], dtype=np.float32)

        # --- Lấy Features Bổ Sung ---
        additional_features = np.array([obs_values["momentum"], obs_values["order_imbalance"]], dtype=np.float32)

        # --- Lấy Sentiment Feature ---
        active_sentiment_for_sac = 0.0
        if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
            try: sentiment_details = self.sentiment_analyzer.get_detailed_active_sentiment(timestamp_for_obs); active_sentiment_for_sac = sentiment_details.get("total_score_adj_confidence", 0.0)
            except Exception as e: logging.error(f"Error getting sentiment for SAC state: {e}")
        sentiment_feature = np.array([active_sentiment_for_sac], dtype=np.float32)

        # --- Ghép nối Features ---
        try:
            feature_parts = [base_features, ob_extra_features, lob_features, mlp_output_feature, additional_features, sentiment_feature]
            state_unscaled = np.concatenate(feature_parts) # <<< Chỉ còn state gốc >>>

            # --- KIỂM TRA SHAPE CUỐI CÙNG ---
            if state_unscaled.shape[0] != expected_dim:
                 # Log CRITICAL vì đây là lỗi nghiêm trọng
                 logging.critical(f"_get_sac_state_unscaled ({symbol}): FINAL SHAPE MISMATCH! Expected ({expected_dim},), got {state_unscaled.shape}. Returning None.")
                 return None # <<< Trả về None nếu shape sai >>>
                 # Hoặc padding nếu muốn cố gắng tiếp tục:
                 # state_unscaled = np.pad(state_unscaled, (0, max(0, expected_dim - state_unscaled.shape[0])), mode='constant', constant_values=0.0)[:expected_dim]

            # Kiểm tra NaN/Inf
            if not np.isfinite(state_unscaled).all():
                 logging.warning(f"NaN/Inf detected in final unscaled SAC state for {symbol}. Replacing with zeros.")
                 state_unscaled = np.nan_to_num(state_unscaled, nan=0.0, posinf=0.0, neginf=0.0)

            # <<< Trả về state_unscaled (74 features) >>>
            return state_unscaled.astype(np.float32)
        except ValueError as e:
             logging.error(f"Error concatenating SAC state for {symbol}: {e}", exc_info=True)
             shapes_str = ", ".join([str(getattr(p, 'shape', 'Not array')) for p in feature_parts])
             logging.error(f"Shapes of parts: {shapes_str}")
             return None
        except Exception as e:
             logging.error(f"Unexpected error creating unscaled SAC state for {symbol}: {e}", exc_info=True)
             return None

    def _scale_sac_state(self, state_unscaled: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if state_unscaled is None: return None

        # Lấy scaler của MarketEnvironment
        sac_env_scaler = None
        if self.entry_optimizer and hasattr(self.entry_optimizer, 'env') and self.entry_optimizer.env:
            try:
                monitor_env = self.entry_optimizer.env.envs[0]
                base_env = getattr(monitor_env, 'env', monitor_env)
                sac_env_scaler = getattr(base_env, 'scaler', None)
                if not (sac_env_scaler and hasattr(sac_env_scaler, 'transform') and hasattr(sac_env_scaler, 'n_features_in_')):
                     sac_env_scaler = None
            except Exception: sac_env_scaler = None

        if sac_env_scaler is None:
            logging.warning("_scale_sac_state: SAC Environment scaler not found or invalid. Returning unscaled state.")
            if not np.isfinite(state_unscaled).all(): state_unscaled = np.nan_to_num(state_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
            return state_unscaled.astype(np.float32)

        try:
            # Lấy số features kỳ vọng từ scaler
            expected_feature_dim = sac_env_scaler.n_features_in_

            # <<< Sửa lỗi: Kiểm tra dimension của state_unscaled >>>
            if state_unscaled.shape[0] != expected_feature_dim:
                logging.error(f"_scale_sac_state: Input state dimension ({state_unscaled.shape[0]}) does not match scaler expected dimension ({expected_feature_dim}). Returning unscaled.")
                if not np.isfinite(state_unscaled).all(): state_unscaled = np.nan_to_num(state_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
                return state_unscaled.astype(np.float32)

            # <<< Sửa lỗi: Chỉ cần scale state_unscaled trực tiếp >>>
            # Kiểm tra NaN/Inf trước khi scale
            if not np.isfinite(state_unscaled).all():
                 logging.warning("_scale_sac_state: NaN/Inf found in unscaled state before scaling. Replacing with 0.")
                 state_unscaled = np.nan_to_num(state_unscaled, nan=0.0, posinf=0.0, neginf=0.0)

            state_reshaped = state_unscaled.reshape(1, -1)
            state_scaled = sac_env_scaler.transform(state_reshaped).flatten()

            # Kiểm tra NaN/Inf sau khi scale
            if not np.isfinite(state_scaled).all():
                logging.warning("_scale_sac_state: NaN/Inf detected after scaling. Replacing with 0.")
                state_scaled = np.nan_to_num(state_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            return state_scaled.astype(np.float32)

        # <<< Phần xử lý state CÓ embedding đã bị xóa >>>

        except Exception as e:
            logging.error(f"Error scaling SAC state: {e}", exc_info=True)
            # Fallback: trả về state unscaled đã xử lý NaN
            if not np.isfinite(state_unscaled).all(): state_unscaled = np.nan_to_num(state_unscaled, nan=0.0, posinf=0.0, neginf=0.0)
            return state_unscaled.astype(np.float32)

    def generate_signal_sync(self, symbol: str, data: dict) -> Optional[dict]:

        tf = CONFIG.get("primary_tf", "15m")
        df = data.get(tf)
        min_lookback = 30 # Lookback tối thiểu cho các chỉ báo cơ bản
        weak_thresh = CONFIG.get("signal_weak_threshold", 0.55)
        medium_thresh = CONFIG.get("signal_medium_threshold", 0.65)
        strong_thresh = CONFIG.get("signal_strong_threshold", 0.75)
        if df is None or len(df) < min_lookback: return None
        if self.check_circuit_breaker_simple(df): return None
        # --- 2. Lấy dữ liệu dòng cuối và các giá trị cần thiết ---
        try:
            last_row = df.iloc[-1]
            # Các cột BẮT BUỘC để chạy logic này
            required_core_cols = ["close", "ATR", "ADX"]
            # Các cột cần cho việc xác định regime (nếu dùng fallback ADX/BBW)
            required_regime_fallback_cols = ["BB_upper", "BB_lower", "BB_middle"]
            # Các cột cần cho features XGBoost (lấy từ config)
            features_base_xgb = CONFIG.get("xgboost_feature_columns", [])
            temporal_features_exist = [f for f in self.temporal_features if f in last_row]
            required_xgb_features = features_base_xgb + temporal_features_exist
            # Các cột cần cho state SAC (phức tạp hơn, giả sử _get_sac_state_unscaled xử lý)
            # Cột regime đã tính trước (ƯU TIÊN SỬ DỤNG)
            required_regime_col = ['hybrid_regime'] # Sử dụng regime đã tính từ Hybrid/HMM

            all_required_cols = list(set(
                required_core_cols +
                required_regime_fallback_cols +
                required_xgb_features +
                required_regime_col
            ))

            # Kiểm tra thiếu cột HOẶC NaN trong các cột cốt lõi
            if any(col not in last_row for col in all_required_cols):
                 missing = [col for col in all_required_cols if col not in last_row]
                 logging.warning(f"generate_signal_sync ({symbol}): Missing required columns: {missing}")
                 return None
            if last_row[required_core_cols].isnull().any():
                 logging.warning(f"generate_signal_sync ({symbol}): NaN found in core columns: {last_row[required_core_cols]}")
                 return None

            entry_price = last_row["close"]
            atr = last_row["ATR"]
            adx = last_row["ADX"]
            # Lấy regime đã tính trước, fallback về giá trị mặc định (1: SIDEWAYS)
            predicted_regime_index = last_row.get("hybrid_regime", 1)
            timestamp = last_row.name

            if entry_price <= 0 or atr <= 0 or pd.isna(adx): # Kiểm tra giá trị cốt lõi hợp lệ
                 logging.warning(f"generate_signal_sync ({symbol}): Invalid core values (Price={entry_price}, ATR={atr}, ADX={adx}).")
                 return None

        except Exception as e:
            logging.error(f"generate_signal_sync ({symbol}): Error accessing data in last row: {e}", exc_info=True)
            return None

        # --- 3. Lấy tín hiệu SAC ---
        sac_direction = None
        if self.entry_optimizer and hasattr(self.entry_optimizer, 'is_trained') and self.entry_optimizer.is_trained():
            try:
                # Lấy state unscaled (KHÔNG dùng embedding trong context này)
                state_unscaled = self._get_sac_state_unscaled(symbol, current_data=data, embedding=None, current_step_index=-1)
                if state_unscaled is not None:
                    state_scaled = self._scale_sac_state(state_unscaled) # Scale state
                    if state_scaled is not None:
                        sac_action_tuple = self.entry_optimizer.predict(state_scaled, deterministic=True)
                        sac_action_value = sac_action_tuple[0].item()
                        sac_long_thresh = CONFIG.get("sac_long_threshold", 0.1)
                        sac_short_thresh = CONFIG.get("sac_short_threshold", -0.1)
                        if sac_action_value > sac_long_thresh: sac_direction = "LONG"
                        elif sac_action_value < sac_short_thresh: sac_direction = "SHORT"
                        # else: logging.debug(f"SAC Action {sac_action_value:.3f} within thresholds.")
            except Exception as sac_e:
                logging.warning(f"generate_signal_sync ({symbol}): Error getting SAC signal: {sac_e}")
                pass # Tiếp tục

        # --- 4. Lấy tín hiệu XGBoost ---
        xgb_win_prob = 0.5 # Xác suất thắng CỦA HƯỚNG ĐƯỢC CHỌN (Long hoặc Short)
        xgb_direction = None
        if symbol in self.models and symbol in self.scalers:
            try:
                # Lấy features cho XGBoost (KHÔNG dùng embedding)
                current_features_s = last_row[required_xgb_features].fillna(0) # Fill NaN còn sót
                current_features = current_features_s.values.reshape(1, -1)
                scaler = self.scalers[symbol]

                if hasattr(scaler, 'n_features_in_') and current_features.shape[1] == scaler.n_features_in_:
                    scaled_features = scaler.transform(current_features)
                    model = self.models[symbol]
                    xgb_pred_proba = model.predict_proba(scaled_features)
                    xgb_prob_long = xgb_pred_proba[0][1] # Xác suất lớp 1 (LONG)
                    xgb_prob_short = 1.0 - xgb_prob_long # Xác suất lớp 0 (SHORT)

                    # So sánh với ngưỡng tương ứng
                    if xgb_prob_long >= self.XGB_WEAK_PROB: # Ưu tiên Long nếu cả hai đều qua ngưỡng yếu
                        xgb_direction = "LONG"
                        xgb_win_prob = xgb_prob_long
                    elif xgb_prob_short >= self.XGB_SHORT_WEAK_THRESH_P0: # Ngưỡng tính trên P(Short)
                        xgb_direction = "SHORT"
                        xgb_win_prob = xgb_prob_short # Lưu xác suất thắng của Short

                else:
                    expected_dim = getattr(scaler, 'n_features_in_', 'N/A')
                    logging.warning(f"generate_signal_sync ({symbol}): XGB Feature mismatch. Expected {expected_dim}, got {current_features.shape[1]}")
            except Exception as xgb_e:
                logging.warning(f"generate_signal_sync ({symbol}): Error getting XGB signal: {xgb_e}")
                pass # Tiếp tục

        # --- 5. Kết hợp tín hiệu và xác định chất lượng ---
        final_direction = None
        signal_quality = 'NONE'
        final_win_prob = 0.5 # Xác suất tổng hợp của hướng cuối cùng

        # Xác định trọng số dựa trên regime đã tính trước (từ df)
        regime_map_local = {0: "TREND_UP", 1: "SIDEWAYS", 2: "TREND_DOWN", -1: "UNKNOWN"} # Map cục bộ nếu cần
        predicted_regime_name = regime_map_local.get(predicted_regime_index, "UNKNOWN")

        weight_key = 'default' # Mặc định
        if "TREND" in predicted_regime_name.upper():
            weight_key = 'trending'
        elif "SIDEWAYS" in predicted_regime_name.upper():
            # Kiểm tra thêm volatility nếu là SIDEWAYS (dùng BBW)
            bb_middle = last_row.get('BB_middle')
            if bb_middle is not None and bb_middle > 0 and not pd.isna(last_row.get('BB_upper')) and not pd.isna(last_row.get('BB_lower')):
                 bollinger_width = (last_row['BB_upper'] - last_row['BB_lower']) / bb_middle
                 if bollinger_width > CONFIG.get("bb_width_vol_threshold", 0.05):
                      weight_key = 'high_vol'
                 # else: weight_key = 'default' # Hoặc 'sideways_low_vol'
            # else: logging.warning(f"Cannot calculate BBW for regime check ({symbol}). Using default weights.")
        # else: weight_key = 'default' # UNKNOWN dùng default

        # Lấy trọng số từ config một cách an toàn
        current_weights = self.ADAPTIVE_WEIGHTS.get(weight_key, self.ADAPTIVE_WEIGHTS.get('default', {'sac': 0.5, 'xgb': 0.5}))
        w_sac = current_weights.get('sac', 0.5)
        w_xgb = current_weights.get('xgb', 0.5)
        # logging.debug(f"generate_signal_sync ({symbol}): Regime Idx={predicted_regime_index}, Name='{predicted_regime_name}', WeightKey='{weight_key}', W_SAC={w_sac:.2f}, W_XGB={w_xgb:.2f}")

        # Tính điểm số (giống logic trước)
        sac_long_score = 1.0 if sac_direction == "LONG" else 0.0
        sac_short_score = 1.0 if sac_direction == "SHORT" else 0.0
        xgb_long_score = xgb_win_prob if xgb_direction == "LONG" else 0.0 # win_prob là P(Long)
        xgb_short_score = xgb_win_prob if xgb_direction == "SHORT" else 0.0 # win_prob là P(Short)

        score_long = sac_long_score * w_sac + xgb_long_score * w_xgb
        score_short = sac_short_score * w_sac + xgb_short_score * w_xgb

        # Quyết định cuối cùng dựa trên điểm số và ngưỡng
        if score_long > score_short and score_long >= weak_thresh:
            final_direction = "LONG"
            final_win_prob = 0.5 + score_long / 2
            if score_long >= strong_thresh: signal_quality = 'STRONG'
            elif score_long >= medium_thresh: signal_quality = 'MEDIUM'
            else: signal_quality = 'WEAK'
        elif score_short > score_long and score_short >= weak_thresh:
            final_direction = "SHORT"
            final_win_prob = 0.5 + score_short / 2
            if score_short >= strong_thresh: signal_quality = 'STRONG'
            elif score_short >= medium_thresh: signal_quality = 'MEDIUM'
            else: signal_quality = 'WEAK'
        elif sac_direction and xgb_direction and sac_direction != xgb_direction:
            signal_quality = 'CONFLICT'
            logging.debug(f"generate_signal_sync ({symbol}): SAC/XGB Conflict.")
        else:
            signal_quality = 'NONE'
            logging.debug(f"generate_signal_sync ({symbol}): No signal generated (Scores L={score_long:.3f}, S={score_short:.3f}).")


        # --- 6. Trả về Dictionary Tín hiệu Cơ sở ---
        if final_direction:
            return {
                "direction": final_direction,
                "win_prob": np.clip(final_win_prob, 0.01, 0.99),
                "signal_quality": signal_quality,
                "entry_price": entry_price,
                "atr": atr,
                "adx": adx,
                "timestamp": timestamp,
                "symbol": symbol,
                "predicted_regime_used": predicted_regime_name 
            }
        else:
            return None


    def calculate_position_size(self, signal: dict, stop_loss_price: float, final_risk_factor: float,active_sentiment_score: float = 0.0) -> float:
        symbol = signal.get("symbol")
        entry_price = signal.get("entry_price")
        atr = signal.get("atr") # Có thể vẫn cần atr cho các logic khác trong sizer

        # --- 1. Kiểm tra dữ liệu đầu vào cơ bản ---
        if not symbol or not entry_price or entry_price <= 0 or stop_loss_price is None:
            logging.warning(f"calculate_position_size ({symbol}): Invalid basic inputs (symbol, entry, sl). Returning 0 size.")
            return 0.0

        # Kiểm tra SL hợp lệ so với entry và direction
        direction = signal.get("direction")
        if not direction or \
           (direction == 'LONG' and stop_loss_price >= entry_price) or \
           (direction == 'SHORT' and stop_loss_price <= entry_price):
            logging.warning(f"calculate_position_size ({symbol}): Invalid stop_loss_price ({stop_loss_price:.4f}) relative to entry ({entry_price:.4f}) for direction {direction}. Returning 0 size.")
            return 0.0
        signal_timestamp = signal.get("timestamp")

        # Kiểm tra final_risk_factor hợp lệ (đã được tính toán và điều chỉnh từ bên ngoài)
        if not (0 < final_risk_factor <= CONFIG.get("max_account_risk", 0.1)):
            logging.warning(f"calculate_position_size ({symbol}): Invalid final_risk_factor received: {final_risk_factor:.4f}. Returning 0 size.")
            # Có thể trả về min_qty nếu final_risk_factor rất nhỏ nhưng > 0? Tạm thời trả 0.
            return 0.0

        # --- 2. Lấy dữ liệu bổ sung cho Sizer (Market Conditions) ---
        # Cần lấy dữ liệu mới nhất để đưa vào sizer
        if symbol not in self.data or "15m" not in self.data[symbol] or self.data[symbol]["15m"].empty:
            logging.warning(f"calculate_position_size ({symbol}): Missing or empty 15m data. Cannot provide full context to sizer. Returning 0 size.")
            return 0.0

        df_15m = self.data[symbol]["15m"]
        try:
            if signal_timestamp and signal_timestamp in df_15m.index:
                last_row = df_15m.loc[signal_timestamp]
            else:
                 last_row = df_15m.iloc[-1]
            # Lấy các thông tin thị trường cần thiết
            volatility = last_row.get("volatility", 0.0)
            ema50 = last_row.get("EMA_50", entry_price) # Fallback về entry_price
            ema200 = last_row.get("EMA_200", entry_price)
            ema_diff = (ema50 - ema200) / entry_price if entry_price > 0 else 0.0
            rsi = last_row.get("RSI", 50.0) # Fallback về 50
            adx = last_row.get("ADX", 25.0) # Fallback về 25
            volume = last_row.get("volume", 0.0)
            # Lấy order book mới nhất từ self.data (được cập nhật trong _process_symbol_data_update)
            order_book = self.data[symbol].get("order_book", {"bids": [], "asks": []})

        except IndexError:
            logging.error(f"calculate_position_size ({symbol}): IndexError accessing last row of 15m data. Returning 0 size.")
            return 0.0
        except Exception as data_e:
            logging.error(f"calculate_position_size ({symbol}): Error getting market conditions: {data_e}", exc_info=True)
            return 0.0 # Không tính size nếu lỗi lấy dữ liệu

        # --- 3. Chuẩn bị tham số cho SmartPositionSizer ---
        strategy_params = {
            "symbol": symbol,
            "entry_price": entry_price,
            "atr": atr, # Truyền atr gốc từ signal
            "volatility": volatility,
            "rsi": rsi,
            "adx": adx,
            "ema_diff": ema_diff,
            "volume": volume,
            "sentiment": active_sentiment_score,
            "order_book": order_book,
            "direction": direction
        }

        # --- 4. Gọi SmartPositionSizer để tính size ---
        try:
            # Kiểm tra sizer tồn tại
            if not hasattr(self, 'position_sizer') or self.position_sizer is None:
                logging.error(f"calculate_position_size ({symbol}): SmartPositionSizer is not initialized. Returning 0 size.")
                return 0.0

            # Gọi hàm calculate_size của sizer, truyền final_risk_factor đã điều chỉnh
            calculated_size = self.position_sizer.calculate_size(
                strategy_params,
                stop_loss_price,
                final_risk_factor # Sử dụng risk factor đã được tính toán và điều chỉnh trước đó
            )

            # --- 5. Xử lý kết quả từ Sizer ---
            min_qty = self._safe_get_min_qty(symbol)

            if calculated_size < 0:
                 logging.warning(f"calculate_position_size ({symbol}): Sizer returned negative size ({calculated_size:.8f}). Setting to 0.")
                 final_size = 0.0
            elif 0 < calculated_size < min_qty:
                 # Nếu sizer trả về size dương nhưng nhỏ hơn mức tối thiểu, coi như không vào lệnh
                 logging.info(f"calculate_position_size ({symbol}): Sizer returned size ({calculated_size:.8f}) < MinQty ({min_qty:.8f}). Setting to 0.")
                 final_size = 0.0
            elif calculated_size == 0:
                 min_notional = min_qty * entry_price
                 required_margin = min_notional / CONFIG.get('leverage', 1) if CONFIG.get('leverage', 1) > 0 else min_notional
                 # Cho phép vào min_qty nếu sizer trả về 0 và đủ margin (có thể xem xét lại logic này)
                 if self.balance * final_risk_factor * 0.8 > required_margin: # Dùng hệ số nhỏ hơn (0.8) để an toàn
                      logging.warning(f"calculate_position_size ({symbol}): Sizer returned 0, but allowing minimum qty ({min_qty:.8f}) based on risk/margin.")
                      final_size = min_qty
                 else:
                      logging.info(f"calculate_position_size ({symbol}): Sizer returned 0, and not enough margin/risk allowance for min qty. Setting to 0.")
                      final_size = 0.0
            else: # calculated_size >= min_qty
                 final_size = calculated_size # Giữ nguyên size hợp lệ từ sizer

            # Đảm bảo size cuối cùng không âm
            final_size = max(0.0, final_size)

            # Log kết quả cuối cùng
            logging.info(f"Position Size Calculated ({symbol}): FinalRisk={final_risk_factor:.4f} -> Size={final_size:.8f}")

            return final_size

        except Exception as e:
             logging.error(f"Error calling or processing result from position_sizer.calculate_size for {symbol}: {e}", exc_info=True)
             # Fallback an toàn: trả về 0 để không vào lệnh nếu có lỗi nghiêm trọng trong sizer
             return 0.0


    def _safe_get_min_qty(self, symbol: str) -> float:
        default_min_qty = 1e-8 # Giá trị mặc định rất nhỏ

        try:
            market_info = self._get_market_info(symbol) # Gọi hàm helper để lấy market info

            if market_info and \
               'limits' in market_info and \
               isinstance(market_info['limits'], dict) and \
               'amount' in market_info['limits'] and \
               isinstance(market_info['limits']['amount'], dict) and \
               'min' in market_info['limits']['amount']:

                min_val = market_info['limits']['amount']['min']

                if min_val is not None:
                    try:
                        # Trả về giá trị min từ sàn, nhưng không nhỏ hơn default
                        return max(float(min_val), default_min_qty)
                    except (ValueError, TypeError) as conv_err:
                         logging.warning(f"_safe_get_min_qty ({symbol}): Could not convert min quantity '{min_val}' to float: {conv_err}. Using default.")
                         return default_min_qty
                else:
                    # Trường hợp 'min' là None
                    logging.warning(f"_safe_get_min_qty ({symbol}): Minimum amount limit is None in market info. Using default.")
                    return default_min_qty
            else:
                # Trường hợp cấu trúc market_info không đúng hoặc thiếu thông tin limits
                logging.warning(f"_safe_get_min_qty ({symbol}): Could not find valid minimum amount limit in market info. Using default.")
                return default_min_qty

        except Exception as e:
            # Bắt lỗi chung nếu có vấn đề không mong muốn
            logging.error(f"Unexpected error in _safe_get_min_qty for {symbol}: {e}", exc_info=True)
            return default_min_qty
        
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


    def dynamic_stop_management(self, signal: dict) -> Tuple[List[float], float]: # Nhận signal dict
        symbol = signal.get("symbol"); entry_price = signal.get("entry_price"); atr = signal.get("atr")
        adx = signal.get("adx"); direction = signal.get("direction"); win_prob = signal.get("win_prob", 0.5)
        signal_quality = signal.get("signal_quality", 'NONE')
        fallback_sl = entry_price * (1 - 0.02) if direction == "LONG" else entry_price * (1 + 0.02)
        fallback_tp = [entry_price * 1.01 if direction == "LONG" else entry_price * 0.99]
        if not all([symbol, entry_price, atr, adx, direction]): return fallback_tp, fallback_sl
        if symbol not in self.data or "15m" not in self.data[symbol] or self.data[symbol]["15m"].empty: return fallback_tp, fallback_sl
        df = self.data[symbol]["15m"]
        try:
            volatility = df["volatility"].iloc[-1]; current_close = df["close"].iloc[-1]
            if pd.isna(volatility) or pd.isna(current_close) or current_close == 0: raise ValueError("NaN/zero")
            atr_factor = atr / current_close
        except (IndexError, ValueError, KeyError): volatility = 0.01; atr_factor = 0.01
        if signal_quality == 'STRONG': tp_multiplier = 1.2; sl_multiplier = 1.0
        elif signal_quality == 'MEDIUM': tp_multiplier = 1.0; sl_multiplier = 0.85
        else: tp_multiplier = 0.8; sl_multiplier = 0.7
        if adx > 30: base_multiplier = 1.0 + volatility * 0.5
        elif adx > 25: base_multiplier = 0.8 + volatility * 0.3
        else: base_multiplier = 0.5 + volatility * 0.2
        fib_base = [1.272, 1.414, 1.618, 2.0, 2.618]
        tp_levels = [entry_price * (1 + level * atr_factor * base_multiplier * tp_multiplier) if direction == "LONG"
                     else entry_price * (1 - level * atr_factor * base_multiplier * tp_multiplier)
                     for level in fib_base]
        initial_sl_distance = 2 * atr * sl_multiplier
        initial_sl_adjusted = entry_price - initial_sl_distance if direction == "LONG" else entry_price + initial_sl_distance
        position_for_iss = { "symbol": symbol, "entry": entry_price, "initial_sl_adjusted": initial_sl_adjusted, "direction": direction, "duration": 0 }
        iss_data = self.data.get(symbol)
        if iss_data and "15m" in iss_data:
            try: final_sl = self.stop_system.calculate_sl(position_for_iss, iss_data)
            except Exception: final_sl = initial_sl_adjusted
        else: final_sl = initial_sl_adjusted
        if signal_quality == 'STRONG': risk_for_max_loss = 0.05
        elif signal_quality == 'MEDIUM': risk_for_max_loss = 0.03
        else: risk_for_max_loss = 0.015
        max_loss_amount = entry_price * risk_for_max_loss # Đây là khoảng cách giá trị, không phải giá SL
        if direction == "LONG": max_loss_sl_price = entry_price - max_loss_amount; final_sl = max(final_sl, max_loss_sl_price)
        else: max_loss_sl_price = entry_price + max_loss_amount; final_sl = min(final_sl, max_loss_sl_price)
        # logging.info(f"Dynamic Stop Management ({symbol}): ... FinalSL={final_sl:.4f}")
        return tp_levels, final_sl

    async def _simulate_trade(self, data: Dict[str, pd.DataFrame], current_step: int, signal: dict, tp_levels: List[float], sl: float, position_size: float) -> Tuple[float, float]:
        df = data.get("15m");
        if df is None or df.empty: return signal.get("entry_price", 0.0), 0.0
        direction = signal.get("direction"); entry_price = signal.get("entry_price"); symbol = signal.get("symbol", "N/A")
        if direction not in ["LONG", "SHORT"] or entry_price is None or position_size <= 0: return entry_price if entry_price is not None else 0.0, 0.0
        exit_price = entry_price; trade_closed = False
        if current_step >= len(df): return entry_price, 0.0
        for i in range(current_step + 1, len(df)): # Bắt đầu từ nến *sau* nến vào lệnh
            if i >= len(df): break
            row = df.iloc[i]; high, low = row.get("high"), row.get("low")
            if high is None or low is None: continue
            if direction == "LONG":
                if high >= max(tp_levels): exit_price = max(tp_levels); trade_closed = True; break
                if low <= sl: exit_price = sl; trade_closed = True; break
            elif direction == "SHORT":
                if low <= min(tp_levels): exit_price = min(tp_levels); trade_closed = True; break
                if high >= sl: exit_price = sl; trade_closed = True; break
        if not trade_closed:
             if len(df) > current_step: exit_price = df['close'].iloc[-1]
             else: return entry_price, 0.0
        if direction == "LONG": raw_pnl = (exit_price - entry_price) * position_size
        elif direction == "SHORT": raw_pnl = (entry_price - exit_price) * position_size
        else: raw_pnl = 0.0
        fees = (abs(entry_price * position_size) + abs(exit_price * position_size)) * CONFIG.get("fee_rate", 0.0005)
        pnl = raw_pnl - fees
        # logging.debug(f"Simulated async trade for {symbol}: ... PNL={pnl:.4f}")
        return exit_price, pnl

    def _simulate_trade_sync(self, data: Dict[str, pd.DataFrame], current_step: int, signal: dict, tp_levels: List[float], sl: float, position_size: float) -> Tuple[float, float]:
        df = data.get("15m");
        if df is None or df.empty: return signal.get("entry_price", 0.0), 0.0
        direction = signal.get("direction"); entry_price = signal.get("entry_price"); symbol = signal.get("symbol", "N/A")
        if direction not in ["LONG", "SHORT"] or entry_price is None or position_size <= 0: return entry_price if entry_price is not None else 0.0, 0.0
        exit_price = entry_price; trade_closed = False
        if current_step >= len(df): return entry_price, 0.0
        for i in range(current_step + 1, len(df)):
            if i >= len(df): break
            row = df.iloc[i]; high, low = row.get("high"), row.get("low")
            if high is None or low is None: continue
            if direction == "LONG":
                if high >= max(tp_levels): exit_price = max(tp_levels); trade_closed = True; break
                if low <= sl: exit_price = sl; trade_closed = True; break
            elif direction == "SHORT":
                if low <= min(tp_levels): exit_price = min(tp_levels); trade_closed = True; break
                if high >= sl: exit_price = sl; trade_closed = True; break
        if not trade_closed:
             if len(df) > current_step: exit_price = df['close'].iloc[-1]
             else: return entry_price, 0.0
        if direction == "LONG": raw_pnl = (exit_price - entry_price) * position_size
        elif direction == "SHORT": raw_pnl = (entry_price - exit_price) * position_size
        else: raw_pnl = 0.0
        fees = (abs(entry_price * position_size) + abs(exit_price * position_size)) * CONFIG.get("fee_rate", 0.0005)
        pnl = raw_pnl - fees
        # logging.debug(f"Simulated sync trade for {symbol}: ... PNL={pnl:.4f}")
        return exit_price, pnl


    async def _update_balance(self, pnl: float, position_size: float, entry_price: float, exit_price: float, symbol: str, direction: str, timestamp: Any):
        """Cập nhật số dư, lịch sử, exposure (AN TOÀN - có lock nội bộ)."""
        acquired_locks = [] # Theo dõi các lock đã acquire để giải phóng
        lock_failed = False # Cờ báo lỗi acquire lock

        try:
            # === Acquire locks theo đúng thứ tự: balance -> positions -> history ===
            logging.debug(f"Acquiring locks for balance update ({symbol})...")

            # 1. Acquire balance_lock (bao gồm exposure, equity_peak)
            acquired_balance = await asyncio.wait_for(self.balance_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_balance: raise asyncio.TimeoutError("Timeout acquiring balance_lock")
            acquired_locks.append(self.balance_lock)

            # 2. Acquire positions_lock (bao gồm open_positions, trailing_stops)
            acquired_positions = await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_positions: raise asyncio.TimeoutError("Timeout acquiring positions_lock")
            acquired_locks.append(self.positions_lock)

            # 3. Acquire trade_history_lock
            acquired_history = await asyncio.wait_for(self.trade_history_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_history: raise asyncio.TimeoutError("Timeout acquiring trade_history_lock")
            acquired_locks.append(self.trade_history_lock)

            logging.debug(f"Locks acquired for balance update ({symbol}).")

            # === Logic cập nhật state (BÊN TRONG TẤT CẢ CÁC LOCK) ===

            # Kiểm tra lại vị thế tồn tại (phòng trường hợp gọi nhiều lần)
            if symbol not in self.open_positions:
                 logging.warning(f"_update_balance: Position {symbol} already closed/missing inside lock.")
                 # Không cần làm gì thêm, sẽ giải phóng lock ở finally
                 return

            # Cập nhật số dư và equity peak (dùng balance_lock)
            balance_before = self.balance # Lưu lại để tính exposure
            self.balance += pnl
            self.equity_peak = max(self.equity_peak, self.balance)

            # Cập nhật exposure (dùng balance_lock)
            notional_value = abs(position_size * entry_price)
            exposure_reduction = notional_value / max(balance_before, 1.0) # Dùng balance_before
            self.exposure_per_symbol[symbol] = max(0.0, self.exposure_per_symbol.get(symbol, 0.0) - exposure_reduction)

            # Ghi lịch sử giao dịch (dùng trade_history_lock)
            trade_log = {
                "symbol": symbol, "direction": direction, "position_size": position_size,
                "entry_price": entry_price, "exit_price": exit_price, "pnl": pnl,
                "balance_after": self.balance, "timestamp": timestamp
            }
            self.trade_history.append(trade_log)

            # Xóa vị thế khỏi danh sách mở (dùng positions_lock)
            del self.open_positions[symbol]
            # Xóa trailing stop nếu có (dùng positions_lock)
            if symbol in self.trailing_stops:
                 del self.trailing_stops[symbol]

            # Log thông tin (vẫn trong lock)
            logging.info(f"Trade Closed ({self.mode}): {symbol} {direction} Size={position_size:.6f} Entry={entry_price:.4f} Exit={exit_price:.4f} PNL={pnl:.4f} NewBalance={self.balance:.2f}")

        except asyncio.TimeoutError as te:
            lock_failed = True
            logging.error(f"Timeout acquiring lock during balance update for {symbol}: {te}")
            # QUAN TRỌNG: Nếu không lấy được lock, không nên tiếp tục vì state có thể không nhất quán.
            # Cần có cơ chế xử lý lỗi ở đây, ví dụ: đánh dấu lỗi, thử lại, dừng bot?
            # Tạm thời chỉ log lỗi.
        except KeyError:
            # Lỗi này không nên xảy ra nếu kiểm tra `symbol not in self.open_positions` ở trên hoạt động đúng.
            logging.error(f"KeyError during _update_balance for {symbol} inside lock. Position state might be inconsistent.")
        except Exception as e:
             logging.error(f"Error inside _update_balance lock for {symbol}: {e}", exc_info=True)
             logging.error(traceback.format_exc())
        finally:
            # === Đảm bảo giải phóng TẤT CẢ các lock đã acquire theo thứ tự NGƯỢC LẠI ===
            logging.debug(f"Releasing locks for balance update ({symbol}). Failed: {lock_failed}")
            for lock in reversed(acquired_locks):
                try:
                    if lock.locked():
                        lock.release()
                except RuntimeError as rel_e: # Bắt lỗi nếu cố release lock không được giữ (ít xảy ra)
                    logging.error(f"Error releasing lock {lock}: {rel_e}")
            logging.debug(f"Locks released for balance update ({symbol}).")

    def _is_new_candle_detected(self, symbol: str, tf: str) -> bool:
        """Kiểm tra xem có nến mới cho symbol/tf không và cập nhật timestamp."""
        if symbol not in self.data or tf not in self.data[symbol] or \
           not isinstance(self.data[symbol][tf], pd.DataFrame) or self.data[symbol][tf].empty:
            return False
        try:
            latest_candle_ts = self.data[symbol][tf].index[-1]
            if not isinstance(latest_candle_ts, pd.Timestamp): return False

            last_processed = self.last_processed_candle_ts.get(symbol, {}).get(tf)
            is_new = (last_processed is None) or (latest_candle_ts > last_processed)

            if is_new:
                # Cập nhật timestamp đã xử lý
                if symbol not in self.last_processed_candle_ts: self.last_processed_candle_ts[symbol] = {}
                self.last_processed_candle_ts[symbol][tf] = latest_candle_ts
                return True
            return False
        except IndexError:
            return False
        except Exception as e:
            logging.error(f"Error in _is_new_candle_detected for {symbol} {tf}: {e}")
            return False


    def _add_open_position(self,
                           signal: Dict[str, Any],
                           position_size: float,
                           sl_price: float,
                           agent_signals: Optional[Dict[str, Any]] = None
                          ) -> bool: # <<< Thêm kiểu trả về >>>

        func_name = "_add_open_position"

        # === ASSERTION: Kiểm tra caller đã giữ lock ===
        # Quan trọng để đảm bảo an toàn concurrency
        if not self.balance_lock.locked():
             logging.critical(f"FATAL ({func_name}): Called without holding balance_lock!")
             # raise RuntimeError("Must hold balance_lock") # Có thể raise lỗi
             return False
        if not self.positions_lock.locked():
             logging.critical(f"FATAL ({func_name}): Called without holding positions_lock!")
             # raise RuntimeError("Must hold positions_lock")
             return False

        symbol = signal.get('symbol')
        entry_price = signal.get('entry_price')
        direction = signal.get('direction')
        timestamp = signal.get('timestamp')

        # --- Kiểm tra đầu vào cơ bản ---
        if not symbol or not entry_price or not direction or not timestamp or position_size <= 0 or sl_price is None:
            logging.error(f"{func_name} ({symbol}): Invalid inputs provided.")
            return False
        # --- Lấy market_classification_at_entry từ agent_signals ---
        market_class_at_entry = None
        if agent_signals and isinstance(agent_signals.get('market_classification'), dict):
            market_class_at_entry = copy.deepcopy(agent_signals['market_classification'])
            logging.debug(f"Storing market class at entry ({symbol}): {market_class_at_entry.get('type')}/{market_class_at_entry.get('strength')}")

        # --- Tạo thông tin vị thế (ghi vào self.open_positions) ---
        # Được bảo vệ bởi positions_lock của caller
        self.open_positions[symbol] = {
            "position_size": position_size, "entry_price": entry_price, "direction": direction,
            "stop_loss_price": sl_price, "trailing_tp_level": 0, "min_locked_tp_level": 0,
            "initial_atr": signal.get('atr'), "timestamp_entry": timestamp,
            "signal_quality_entry": signal.get('signal_quality'),
            "market_classification_at_entry": market_class_at_entry,
            "last_agent_signals": copy.deepcopy(agent_signals) if agent_signals else None
        }

        # --- Cập nhật exposure (đọc self.balance, ghi self.exposure_per_symbol) ---
        # Được bảo vệ bởi balance_lock của caller
        try:
            notional_value = abs(position_size * entry_price)
            exposure_increase = notional_value / max(self.balance, 1.0) # Đọc balance
            self.exposure_per_symbol[symbol] = self.exposure_per_symbol.get(symbol, 0.0) + exposure_increase # Ghi exposure
        except Exception as exp_e:
             logging.error(f"{func_name} ({symbol}): Error updating exposure: {exp_e}")
        # --- Logging cuối cùng ---
        try:
             amount_prec = self._safe_get_amount_precision_digits(symbol)
             price_prec = self._safe_get_price_precision_digits(symbol)
             logging.info(f"Position Opened STATE UPDATED ({self.mode}): {symbol} {direction} Size={position_size:.{amount_prec}f} Entry={entry_price:.{price_prec}f} SL={sl_price:.{price_prec}f}.")
        except Exception as log_e:
             logging.error(f"{func_name} ({symbol}): Error formatting log: {log_e}")

        return True

    async def _update_monitor_new_position(self, symbol, direction, entry_price, sl_price):
        """Helper async để cập nhật monitor khi mở vị thế mới (an toàn)."""
        # <<< Hàm này không cần lock vì nó gọi phương thức của monitor (monitor tự quản lý lock nội bộ) >>>
        if not self.realtime_monitor: return
        try:
            thresholds_for_monitor = {'SL': sl_price, 'entry_price': entry_price}
            max_rr_level = float(CONFIG.get('monitor_max_rr_level', 5.0)); rr_step = float(CONFIG.get('monitor_rr_step', 0.5))
            if rr_step <= 0: rr_step = 0.5
            rr_levels_to_calc = np.arange(rr_step, max_rr_level + rr_step / 2, rr_step)
            for rr_level in rr_levels_to_calc:
                rr_price = CombinedTradingAgent._calculate_rr_price(entry_price, sl_price, rr_level, direction.lower())
                if rr_price is not None: thresholds_for_monitor[f'RR_{rr_level:.1f}'.replace('.0', '')] = rr_price
            # Gọi update_thresholds của monitor
            self.realtime_monitor.update_thresholds(
                symbol=symbol, active=True, position_type=direction.lower(),
                thresholds=thresholds_for_monitor, current_trailing_tp_level=0, min_locked_tp_level=0
            )
            logging.info(f"Activated/Updated RealtimeMonitor for new {symbol} position.")
        except Exception as monitor_e:
             logging.error(f"Failed to update RealtimeMonitor for new {symbol} position: {monitor_e}", exc_info=True)

    async def run(self):
        """Vòng lặp chính của bot: khởi tạo, huấn luyện, chạy real-time."""
        try:
            # --- 1. Khởi tạo Cơ bản ---
            # Initialize chỉ khởi tạo exchange và các thành phần không phụ thuộc data nhiều
            await self.initialize() # initialize giờ đây gọn hơn
            logging.info("Bot basic initialization complete.")

            # --- Start Sentiment Task (nếu có) ---
            self.sentiment_task = None
            if self.sentiment_analyzer:
                async def sentiment_periodic_loop():
                    # ... (logic sentiment loop) ...
                    logging.info("Starting sentiment periodic task loop...")
                    fetch_interval = self.sentiment_analyzer.config.get("calendar_fetch_interval_seconds", 3600)
                    check_interval = 60
                    while True:
                        async with self.misc_state_lock: is_done = self._cleanup_complete
                        if is_done: logging.info("Sentiment loop exiting."); break
                        try:
                            await self.sentiment_analyzer.run_periodic_tasks()
                            await asyncio.sleep(check_interval)
                        except asyncio.CancelledError: logging.info("Sentiment task cancelled."); break
                        except Exception as sent_loop_e: logging.error(f"Sentiment loop error: {sent_loop_e}"); await asyncio.sleep(check_interval * 5)
                    logging.info("Sentiment loop finished.")
                self.sentiment_task = asyncio.create_task(sentiment_periodic_loop())
                logging.info("Started sentiment background task.")

            # --- 2. Lấy và Xử lý Dữ liệu Ban đầu (Song song) ---
            all_symbols_data_processed: Dict[str, Optional[Dict[str, pd.DataFrame]]] = {}
            logging.info("Fetching and processing initial data for all symbols...")
            symbols_to_process = CONFIG.get("symbols", [])
            # Tạo task gọi hàm helper mới cho mỗi symbol
            fetch_tasks = [self._fetch_and_process_initial_symbol_data(s) for s in symbols_to_process]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Thu thập kết quả và cập nhật all_symbols_data_processed (không cần cập nhật self.data nữa vì helper đã làm)
            for symbol, result in zip(symbols_to_process, results):
                if isinstance(result, dict) and result: # Kiểm tra kết quả là dict và không rỗng
                    all_symbols_data_processed[symbol] = result # Lưu lại chỉ để kiểm tra xem có data không
                    logging.info(f"Successfully processed initial data for {symbol}.")
                elif isinstance(result, Exception):
                    logging.error(f"Failed initial fetch/process for {symbol}: {result}")
                else:
                    logging.error(f"Failed initial fetch/process for {symbol} (returned None or empty).")

            # Kiểm tra xem có dữ liệu nào được xử lý thành công không
            if not any(all_symbols_data_processed.values()):
                logging.critical("No valid initial data processed for any symbol. Cannot proceed.")
                return # Dừng nếu không có dữ liệu ban đầu

            # --- 3. Huấn luyện Mô hình Ban đầu (Song song) ---
            logging.info("Training initial models...")

            # Huấn luyện XGBoost
            train_xgb_tasks = [self._train_xgb_for_symbol(symbol) for symbol in all_symbols_data_processed if all_symbols_data_processed[symbol]] # Chỉ train symbol có data
            if train_xgb_tasks:
                 logging.info(f"Starting XGBoost training for {len(train_xgb_tasks)} symbols...")
                 await asyncio.gather(*train_xgb_tasks, return_exceptions=True)
                 logging.info("XGBoost training tasks completed.")
            else:
                 logging.warning("No symbols eligible for XGBoost training.")

            # Huấn luyện DRL (DQN và SAC) - Cần dữ liệu đã xử lý trong self.data
            # Hàm train DRL/SAC cần truy cập self.data để tạo Env, nên chạy tuần tự sau khi data sẵn sàng
            if any(all_symbols_data_processed.values()): # Kiểm tra lại lần nữa
                 try:
                      logging.info("Training DRL models (DQN size adjust)...")
                      # <<< Truyền self.data vào train_drl >>>
                      await self.train_drl(self.data)
                      logging.info("Training DRL models (SAC entry)...")
                      # <<< Truyền self.data vào train_entry_optimizer >>>
                      await self.train_entry_optimizer(self.data)
                 except Exception as rl_train_e:
                      logging.error(f"Error during main RL training phase: {rl_train_e}", exc_info=True)
            else:
                 logging.error("Cannot train RL models due to lack of initial data.")

            # --- 4. Thiết lập Real-Time Data Feed (Live Mode) ---
            if self.mode == "live":
                logging.info("Setting up real-time data feeds...")
                # Lấy danh sách symbols có dữ liệu hợp lệ trong self.data
                async with self.misc_state_lock: # Tạm dùng lock này để đọc keys của self.data
                     symbols_with_data = list(self.data.keys())

                # <<< THÊM: Cập nhật Monitor với tất cả symbols có data >>>
                if self.realtime_monitor and symbols_with_data:
                     # Giả sử Monitor có hàm update_symbols hoặc tương tự
                     # Hoặc khởi tạo lại Monitor với list đầy đủ
                     # self.realtime_monitor.update_symbols(symbols_with_data)
                     logging.info(f"RealtimeMonitor tracking symbols: {symbols_with_data}")

                for symbol in symbols_with_data:
                    self.data_feeds[symbol] = {}
                    for tf in CONFIG.get("timeframes", []):
                        # Chỉ tạo feed nếu có dữ liệu ban đầu trong self.data
                        if tf in self.data.get(symbol, {}):
                            try:
                                self.data_feeds[symbol][tf] = RealTimeDataFeed(symbol, tf)
                                asyncio.create_task(self.data_feeds[symbol][tf].connect())
                                logging.info(f"Real-time feed task created for {symbol} {tf}")
                            except Exception as feed_e:
                                logging.error(f"Failed to create/connect feed for {symbol} {tf}: {feed_e}")

            # --- 5. Vòng lặp Giao dịch Chính ---
            logging.info(f"Starting main {self.mode.upper()} loop...")
            if self.mode == "live":
                await self._run_live_loop()
            else: # mode == "backtest"
                await self._run_backtest_loop()

        except asyncio.CancelledError:
             logging.info("Bot run task cancelled.") # Bắt lỗi hủy task
        except Exception as outer_e:
             logging.critical(f"CRITICAL error during bot execution: {outer_e}", exc_info=True)
             logging.critical(traceback.format_exc())
        finally:
            # === Khối finally đã sửa ở câu trả lời trước ===
            logging.info("--- Entering Main Execution Finally Block ---")
            if hasattr(self, 'sentiment_task') and self.sentiment_task and not self.sentiment_task.done():
                logging.info("Cancelling sentiment task..."); self.sentiment_task.cancel()
                try: await asyncio.wait_for(self.sentiment_task, timeout=5.0)
                except asyncio.CancelledError: logging.info("Sentiment task cancelled.")
                except Exception as e: logging.error(f"Sentiment cancel error: {e}")
            if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer and hasattr(self.sentiment_analyzer, 'close_http_session'):
                logging.info("Closing sentiment HTTP session...")
                try: await asyncio.wait_for(self.sentiment_analyzer.close_http_session(), timeout=10.0)
                except Exception as e: logging.error(f"Sentiment close error: {e}")

            cleanup_needed = False
            try:
                async with asyncio.wait_for(self.misc_state_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT):
                    if not self._cleanup_complete: cleanup_needed = True
            except Exception as e: logging.error(f"Error checking cleanup flag: {e}"); cleanup_needed = True # Assume needed if error

            if cleanup_needed:
                logging.info("Calling main cleanup function...")
                try: await self._cleanup()
                except Exception as e: logging.error(f"Main cleanup error: {e}")
            else: logging.info("Cleanup already done or check failed.")
            logging.info("--- Exiting Main Execution Finally Block ---")

    async def _fetch_and_process_initial_symbol_data(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        func_name = "_fetch_and_process_initial_symbol_data"
        processed_data: Optional[Dict[str, pd.DataFrame]] = None
        lock_acquired = False
        try:
            # === Acquire data_lock ===
            logging.debug(f"{func_name}: Acquiring data_lock for {symbol}...")
            await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT * 2) # Timeout dài hơn cho fetch/process ban đầu
            lock_acquired = True
            logging.debug(f"{func_name}: Data_lock acquired for {symbol}.")

            # === Fetch dữ liệu thô (bên trong lock) ===
            # Hàm fetch_multi_timeframe_data đọc cache/API và trả về dict df thô
            # Nó không sửa đổi self.data
            symbol_data_raw = await self.fetch_multi_timeframe_data(symbol)

            if not symbol_data_raw or not any(isinstance(df, pd.DataFrame) and not df.empty for df in symbol_data_raw.values()):
                logging.warning(f"{func_name}: No valid raw data fetched or loaded for {symbol}.")
                return None # Thoát nếu không có dữ liệu thô

            # === Tính Indicators (bên trong lock) ===
            processed_data = {} # Khởi tạo dict để lưu kết quả đã xử lý
            calculation_ok = True
            for tf, df_raw in symbol_data_raw.items():
                if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                    try:
                        # Tính toán indicators trên bản sao
                        processed_data[tf] = self.calculate_advanced_indicators(df_raw.copy())
                        # Thêm thuộc tính symbol và precision vào DataFrame để dùng sau
                        if isinstance(processed_data[tf], pd.DataFrame): # Kiểm tra lại sau khi tính
                             processed_data[tf].attrs['symbol'] = symbol
                             processed_data[tf].attrs['price_precision_digits'] = self._safe_get_price_precision_digits(symbol)
                             processed_data[tf].attrs['amount_precision_digits'] = self._safe_get_amount_precision_digits(symbol)
                    except Exception as calc_e:
                        logging.error(f"{func_name}: Error calculating indicators for {symbol} {tf}: {calc_e}")
                        processed_data[tf] = None # Đánh dấu lỗi
                        calculation_ok = False # Đánh dấu có lỗi xảy ra
                else:
                    processed_data[tf] = None # Giữ None nếu df_raw không hợp lệ

            # Nếu có lỗi tính toán indicators, có thể quyết định không tiếp tục
            if not calculation_ok:
                 logging.error(f"{func_name}: Indicator calculation failed for some timeframes of {symbol}. Aborting process for this symbol.")
                 return None

            # === Cập nhật self.data (bên trong lock) ===
            # Ghi đè hoặc thêm mới dữ liệu đã xử lý vào self.data
            if symbol not in self.data: self.data[symbol] = {}
            self.data[symbol].update(processed_data) # Cập nhật với dữ liệu đã tính indicators
            logging.debug(f"{func_name}: Updated self.data for {symbol} with processed data.")
            await self._calculate_and_add_hybrid_regimes(symbol)
            await self._update_latest_embeddings([symbol]) # Xem xét lại hàm này
            final_processed_data_copy = {tf: df.copy() for tf, df in self.data[symbol].items() if isinstance(df, pd.DataFrame)}

            logging.info(f"{func_name}: Successfully fetched and processed initial data for {symbol}.")
            return final_processed_data_copy # Trả về bản sao

        except asyncio.TimeoutError as te:
            logging.error(f"{func_name}: Timeout acquiring data_lock for {symbol}: {te}")
            return None
        except Exception as e:
            logging.error(f"{func_name}: Error processing initial data for {symbol}: {e}", exc_info=True)
            return None
        finally:
            # === Giải phóng lock ===
            if lock_acquired and self.data_locks[symbol].locked():
                self.data_locks[symbol].release()
                logging.debug(f"{func_name}: Data_lock released for {symbol}.")


    async def _train_xgb_for_symbol(self, symbol: str):
        func_name = "_train_xgb_for_symbol"
        df_primary_copy = None # Bản sao của dữ liệu timeframe chính
        all_symbol_data_copy = None # Bản sao của toàn bộ dữ liệu symbol cho WF
        lock_acquired = False

        try:
            tf_primary = CONFIG.get("primary_tf", "15m")

            # === Đọc dữ liệu (BÊN TRONG LOCK NGẮN) ===
            try:
                logging.debug(f"{func_name}: Acquiring data_lock to read for {symbol}...")
                await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                lock_acquired = True
                logging.debug(f"{func_name}: Data_lock acquired for reading {symbol}.")

                # Kiểm tra sự tồn tại của dữ liệu cho symbol
                if symbol not in self.data or not self.data[symbol]:
                    logging.warning(f"{func_name}: No data found for {symbol} in self.data.")
                    return

                # Lấy dữ liệu timeframe chính
                symbol_data_tf = self.data[symbol].get(tf_primary)
                if not isinstance(symbol_data_tf, pd.DataFrame) or symbol_data_tf.empty:
                     logging.warning(f"{func_name}: Skipping XGB training for {symbol} (no valid {tf_primary} data).")
                     return
                df_primary_copy = symbol_data_tf.copy() # Tạo bản sao để dùng ngoài lock

                # <<< Lấy bản sao TOÀN BỘ dữ liệu của symbol cho Walk Forward >>>
                # WF có thể cần các timeframe khác nữa
                all_symbol_data_copy = {tf: df.copy() for tf, df in self.data[symbol].items() if isinstance(df, pd.DataFrame)}

            finally: # Đảm bảo giải phóng lock đọc
                if lock_acquired and self.data_locks[symbol].locked():
                    self.data_locks[symbol].release()
                    logging.debug(f"{func_name}: Data_lock released after reading {symbol}.")

            # === Chuẩn bị dữ liệu ML (NGOÀI LOCK, dùng bản sao) ===
            if df_primary_copy is None: return # Thoát nếu không đọc được data chính
            logging.debug(f"{func_name}: Preparing ML data for {symbol} (outside lock)...")
            # prepare_ml_data dùng bản sao df_primary_copy
            X, y = self.prepare_ml_data(df_primary_copy, symbol)

            if X is None or y is None:
                 logging.warning(f"{func_name}: Could not prepare ML data for XGBoost ({symbol}).")
                 return

            # === Huấn luyện Model (NGOÀI LOCK) ===
            logging.info(f"{func_name}: Starting XGBoost training for {symbol} (outside lock)...")
            # train_model ghi vào self.models[symbol] và self.scalers[symbol] (Cần xem xét lock nếu cần)
            self.train_model(symbol, X, y)

            # === Walk Forward Validation (NGOÀI LOCK, dùng bản sao) ===
            # <<< Giữ lại logic WFV >>>
            if all_symbol_data_copy: # Kiểm tra xem có dữ liệu cho WF không
                logging.info(f"{func_name}: Starting Walk-Forward Validation for {symbol} (outside lock)...")
                # Hàm WF nhận dict dữ liệu đã sao chép
                self.walk_forward_validation(all_symbol_data_copy, symbol)
            else:
                 logging.warning(f"{func_name}: Skipping Walk-Forward Validation for {symbol} (no data copy available).")
            # <<< Kết thúc phần WFV >>>

        except asyncio.TimeoutError:
             logging.error(f"{func_name}: Timeout acquiring data_lock to read for {symbol}.")
        except Exception as e:
             logging.error(f"{func_name}: Error training XGB for {symbol}: {e}", exc_info=True)
             logging.error(traceback.format_exc())

    async def _process_symbol_data_update(self, symbol: str):
        func_name = "_process_symbol_data_update"
        updated_any_tf = False # Cờ chỉ cho OHLCV
        lock_acquired = False
        try:
            # === Acquire data_lock MỘT LẦN cho tất cả cập nhật của symbol này ===
            logging.debug(f"{func_name}: Acquiring data_lock for {symbol} updates...")
            await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            lock_acquired = True
            logging.debug(f"{func_name}: Data_lock acquired for {symbol} updates.")

            # === Bên trong data_lock ===

            # --- Cập nhật OHLCV cho tất cả timeframes (tuần tự hoặc song song đều được bên trong lock) ---
            # Chạy tuần tự thường đơn giản hơn và tránh tạo quá nhiều task nhỏ
            timeframes_to_update = []
            for tf in CONFIG.get("timeframes", []):
                 if symbol in self.data_feeds and tf in self.data_feeds.get(symbol, {}):
                      timeframes_to_update.append(tf)

            if not timeframes_to_update:
                 logging.debug(f"{func_name}: No active feeds for {symbol}.")
                 return # Không có gì để cập nhật

            # Thực hiện cập nhật tuần tự bên trong lock
            for tf in timeframes_to_update:
                # Gọi hàm helper (hàm này giờ KHÔNG cần tự acquire lock nữa)
                updated = await self._update_symbol_timeframe_unsafe(symbol, tf) # Gọi phiên bản "unsafe"
                if updated:
                    updated_any_tf = True

            # --- Cập nhật Order Book (vẫn trong data_lock) ---
            if updated_any_tf: # Chỉ fetch OB nếu có cập nhật OHLCV mới
                logging.debug(f"{func_name}: Fetching updated order book for {symbol}...")
                # fetch_order_book có thể gọi API, nên cân nhắc timeout
                try:
                    ob = await asyncio.wait_for(self.fetch_order_book(symbol, limit=20), timeout=5.0) # Timeout ngắn hơn cho OB
                    if ob:
                        # Ghi vào self.data (đang được bảo vệ bởi lock)
                        if symbol not in self.data: self.data[symbol] = {}
                        self.data[symbol]["order_book"] = ob
                        logging.debug(f"{func_name}: Updated order book for {symbol} in self.data.")
                    else:
                         logging.warning(f"{func_name}: Failed to fetch order book for {symbol} after update.")
                except asyncio.TimeoutError:
                    logging.warning(f"{func_name}: Timeout fetching order book for {symbol}.")
                except Exception as ob_e:
                    logging.error(f"{func_name}: Error fetching order book for {symbol}: {ob_e}")

            # --- Kết thúc khối được bảo vệ bởi data_lock ---

        except asyncio.TimeoutError:
            logging.error(f"{func_name}: Timeout acquiring data_lock for {symbol}. Updates skipped.")
        except Exception as e:
            logging.error(f"{func_name}: Error processing data updates for {symbol}: {e}", exc_info=True)
        finally:
            # === Giải phóng lock ===
            if lock_acquired and self.data_locks[symbol].locked():
                self.data_locks[symbol].release()
                logging.debug(f"{func_name}: Data_lock released for {symbol} updates.")


    async def _update_symbol_timeframe_unsafe(self, symbol: str, tf: str) -> bool:

        try:
            # Lấy dữ liệu mới từ feed (ngoài lock logic của hàm này, nhưng caller giữ lock)
            if symbol not in self.data_feeds or tf not in self.data_feeds[symbol]: return False
            # Sử dụng timeout ngắn hơn vì lock đã được giữ
            new_data = await asyncio.wait_for(self.data_feeds[symbol][tf].get_next_data(), timeout=1.0)
            if not new_data or 'timestamp' not in new_data: return False

            df_new = pd.DataFrame([new_data])
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms', utc=True, errors='coerce')
            df_new = df_new.dropna(subset=['timestamp']).set_index("timestamp")
            if df_new.empty: return False

            # === Thao tác trên self.data (AN TOÀN vì caller giữ lock) ===
            if symbol not in self.data: self.data[symbol] = {}
            current_df = self.data[symbol].get(tf) # Đọc
            if current_df is None or current_df.empty: updated_df = df_new
            else:
                if not isinstance(current_df.index, pd.DatetimeIndex): # Xử lý index
                    current_df = current_df.reset_index(); current_df['timestamp'] = pd.to_datetime(current_df['timestamp'], errors='coerce')
                    current_df = current_df.dropna(subset=['timestamp']).set_index('timestamp')
                combined_df = pd.concat([current_df, df_new])
                updated_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

            if updated_df.empty: return False

            # Tính lại chỉ báo (vẫn trong lock của caller)
            lookback_needed = 500
            df_to_process = updated_df.tail(lookback_needed)
            if len(df_to_process) < 2: return False
            updated_df_processed = self.calculate_advanced_indicators(df_to_process)

            # Ghi vào self.data (vẫn trong lock của caller)
            self.data[symbol][tf] = updated_df_processed
            logging.debug(f"Data updated for {symbol} {tf} (unsafe helper).")
            return True

        except asyncio.TimeoutError:
             #logging.debug(f"Timeout waiting for feed data ({symbol} {tf}) within lock.")
             return False # Không có dữ liệu mới
        except Exception as e:
            logging.error(f"Error in _update_symbol_timeframe_unsafe ({symbol} {tf}): {e}", exc_info=True)
            return False
        
    def _get_observation_for_dqn(self, df_15m: pd.DataFrame) -> Optional[np.ndarray]:
        if df_15m is None or df_15m.empty:
            logging.error("_get_observation_for_dqn: Input DataFrame is empty.")
            return None
        try:
            last_row = df_15m.iloc[-1] # Lấy dòng cuối cùng
            timestamp_for_obs = pd.to_datetime(last_row.name)
            active_sentiment_for_obs = 0.0
            if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
                 try:
                      # Lấy điểm đã điều chỉnh theo confidence
                      sentiment_details = self.sentiment_analyzer.get_detailed_active_sentiment(timestamp_for_obs)
                      active_sentiment_for_obs = sentiment_details.get("total_score_adj_confidence", 0.0)
                 except Exception as e:
                      self.logger.error(f"Error getting sentiment for DQN observation: {e}", exc_info=True)

            # --- Lấy các features cần thiết (phải khớp 100% với TradingEnv._get_observation_unscaled) ---
            obs_list = [
                last_row.get("close", 0.0),
                last_row.get("RSI", 50.0),
                last_row.get("EMA_diff", 0.0),
                last_row.get("volume_anomaly", 0.0),
                last_row.get("EMA_20", last_row.get("close", 0.0)),
                last_row.get("ATR", 0.0),
                last_row.get("volatility", 0.0),
                last_row.get("regime", 0.0),
                last_row.get("VWAP", last_row.get("close", 0.0)),
                last_row.get("VAH", last_row.get("high", 0.0)),
                float(last_row.name.hour) if isinstance(last_row.name, pd.Timestamp) else 0.0, # Lấy từ index
                float(last_row.name.dayofweek) if isinstance(last_row.name, pd.Timestamp) else 0.0,
                last_row.get("VWAP_ADX_interaction", 0.0),
                last_row.get("BB_EMA_sync", 0.0),   
                last_row.get("swing_high", 0.0), # Đảm bảo cột này tồn tại từ detect_swing_points
                last_row.get("swing_low", 0.0),  # Đảm bảo cột này tồn tại
                last_row.get("MACD", 0.0),
                last_row.get("MACD_signal", 0.0),
                last_row.get("ADX", 25.0),
                last_row.get("OBV", 0.0),
                last_row.get("BB_width", 0.0),
                last_row.get("log_volume", 0.0),
                last_row.get("divergence", 0.0), # Đảm bảo cột này tồn tại từ detect_divergence
                last_row.get("rejection", 0.0), # Đảm bảo cột này tồn tại
                self.balance, # Balance hiện tại của bot
                active_sentiment_for_obs,
                last_row.get("momentum", 0.0),
                last_row.get("order_imbalance", 0.0)
            ]
            obs_unscaled = np.array(obs_list, dtype=np.float32)

            # --- Kiểm tra và Xử lý NaN/Inf ---
            if not np.isfinite(obs_unscaled).all():
                logging.warning("_get_observation_for_dqn: NaN/Inf detected in unscaled observation. Replacing with 0.")
                obs_unscaled = np.nan_to_num(obs_unscaled, nan=0.0, posinf=0.0, neginf=0.0)

            # --- Kiểm tra Dimension ---
            # Lấy dimension từ scaler nếu có, nếu không dùng giá trị mặc định (ví dụ 28)
            expected_dim = getattr(self.trading_env_scaler, 'n_features_in_', 28) if self.trading_env_scaler else 28
            if obs_unscaled.shape != (expected_dim,):
                logging.error(f"_get_observation_for_dqn: Observation shape mismatch. Expected {expected_dim}, got {obs_unscaled.shape}. Padding/Truncating.")
                obs_unscaled = np.pad(obs_unscaled, (0, max(0, expected_dim - obs_unscaled.shape[0])), mode='constant')[:expected_dim]

            return obs_unscaled

        except Exception as e:
            logging.error(f"Error creating observation for DQN: {e}", exc_info=True)
            return None
        
    def _get_dynamic_adx_threshold(self, symbol: str, timeframe: str) -> int:
        default_threshold = 20 # Ngưỡng mặc định cứng nếu mọi thứ khác thất bại

        try:
            # 1. Lấy dictionary ngưỡng từ CONFIG, an toàn với dict rỗng nếu key không tồn tại
            threshold_dict = CONFIG.get("adx_thresholds", {})
            if not isinstance(threshold_dict, dict): # Kiểm tra xem có đúng là dict không
                logging.warning(f"CONFIG['adx_thresholds'] is not a dictionary. Using default threshold {default_threshold}.")
                return default_threshold

            # 2. Ưu tiên ngưỡng cụ thể cho timeframe
            tf_threshold = threshold_dict.get(timeframe)
            if tf_threshold is not None: # Nếu tìm thấy key cho timeframe
                if isinstance(tf_threshold, int) and tf_threshold >= 0: # Kiểm tra kiểu và giá trị hợp lệ (ADX >= 0)
                    # logging.debug(f"Using specific ADX threshold {tf_threshold} for {timeframe}")
                    return tf_threshold
                else:
                    logging.warning(f"Invalid ADX threshold value '{tf_threshold}' for timeframe '{timeframe}' in CONFIG. Proceeding to default.")

            # 3. Nếu không có ngưỡng riêng cho timeframe, dùng ngưỡng 'default' trong config
            default_config_threshold = threshold_dict.get("default")
            if default_config_threshold is not None: # Nếu tìm thấy key 'default'
                if isinstance(default_config_threshold, int) and default_config_threshold >= 0:
                    # logging.debug(f"Using default ADX threshold {default_config_threshold} from CONFIG['adx_thresholds']")
                    return default_config_threshold
                else:
                    logging.warning(f"Invalid ADX threshold value '{default_config_threshold}' for 'default' in CONFIG. Proceeding to hardcoded default.")

            # 4. Fallback cuối cùng nếu không tìm thấy gì hợp lệ trong CONFIG
            logging.warning(f"No valid ADX threshold found in CONFIG for '{timeframe}' or 'default'. Using hardcoded default {default_threshold}.")
            return default_threshold

        except Exception as e:
             # Bắt các lỗi không mong muốn khác khi truy cập CONFIG
             logging.error(f"Error accessing dynamic ADX threshold from CONFIG: {e}. Using hardcoded default {default_threshold}.", exc_info=True)
             return default_threshold


    def _get_higher_timeframe_trend(self,
                                    symbol: str,
                                    timeframe: str,
                                    df_htf_snapshot: Optional[pd.DataFrame] 
                                   ) -> Optional[str]:
        # --- 1. Kiểm tra dữ liệu đầu vào ---
        if not isinstance(symbol, str) or not symbol:
             logging.error("_get_higher_tf_trend: Invalid symbol provided.")
             return None
        if not isinstance(timeframe, str) or not timeframe:
             logging.error("_get_higher_tf_trend: Invalid timeframe provided.")
             return None

        # <<< THAY ĐỔI: Kiểm tra df_htf_snapshot thay vì self.data >>>
        if df_htf_snapshot is None or not isinstance(df_htf_snapshot, pd.DataFrame) or df_htf_snapshot.empty:
            logging.warning(f"_get_higher_tf_trend: Missing, empty, or invalid DataFrame snapshot provided for {symbol} {timeframe}.")
            return None
        # Cần ít nhất 2 dòng để truy cập iloc[-1] một cách an toàn
        if len(df_htf_snapshot) < 2:
            logging.warning(f"_get_higher_tf_trend: Not enough data points in snapshot for {symbol} {timeframe} (need at least 2, have {len(df_htf_snapshot)}).")
            return None

        try:
            df_htf = df_htf_snapshot # SỬ DỤNG SNAPSHOT ĐƯỢC TRUYỀN VÀO

            # Sử dụng .iloc[-1] an toàn hơn vì đã kiểm tra len >= 2
            last_row = df_htf.iloc[-1]

            # --- 2. Lấy và kiểm tra các giá trị chỉ báo ---
            # (Logic lấy close, ema50, ema200, adx, macd, macd_signal giữ nguyên)
            close = last_row.get('close', None)
            ema50 = last_row.get('EMA_50', None)
            ema200 = last_row.get('EMA_200', None)
            adx = last_row.get('ADX', None)
            macd = last_row.get('MACD', None)
            macd_signal = last_row.get('MACD_signal', None)

            required_indicators = {'close': close, 'EMA_50': ema50, 'EMA_200': ema200, 'ADX': adx}
            for name, value in required_indicators.items():
                if value is None or pd.isna(value):
                    logging.warning(f"_get_higher_tf_trend: Missing or NaN value for required indicator '{name}' for {symbol} {timeframe} from snapshot.")
                    return None

            if not all(isinstance(v, (int, float)) for v in required_indicators.values()):
                 logging.warning(f"_get_higher_tf_trend: Non-numeric value found in required indicators for {symbol} {timeframe} from snapshot.")
                 return None
            if close <= 0 or ema50 <= 0 or ema200 <= 0 or adx < 0: # type: ignore
                logging.warning(f"_get_higher_tf_trend: Invalid indicator values (<=0 or ADX<0) for {symbol} {timeframe} from snapshot: "
                                f"C={close}, E50={ema50}, E200={ema200}, ADX={adx}")
                return None
            if adx > 100: # type: ignore
                 logging.warning(f"_get_higher_tf_trend: Unusually high ADX value ({adx:.2f}) for {symbol} {timeframe} from snapshot. Treating as indeterminate.")
                 return "SIDEWAYS"

            # Lấy ngưỡng ADX động (logic này không thay đổi, nó lấy từ config)
            adx_threshold_htf = self._get_dynamic_adx_threshold(symbol, timeframe)

            # --- 3. Logic Xác Định Xu Hướng ---
            ema_bullish_cross = ema50 > ema200 # type: ignore
            ema_bearish_cross = ema50 < ema200 # type: ignore
            price_above_cloud = close > max(ema50, ema200) # type: ignore
            price_below_cloud = close < min(ema50, ema200) # type: ignore
            # price_in_cloud = not price_above_cloud and not price_below_cloud # Không dùng trực tiếp

            macd_bullish_confirm = False
            macd_bearish_confirm = False
            if pd.notna(macd) and pd.notna(macd_signal):
                 if macd > macd_signal: macd_bullish_confirm = True # type: ignore
                 if macd < macd_signal: macd_bearish_confirm = True # type: ignore

            adx_trending = adx > adx_threshold_htf # type: ignore
            trend = "SIDEWAYS"

            if adx_trending:
                if price_above_cloud and ema_bullish_cross:
                    trend = "UP"
                    if macd_bullish_confirm: trend = "UP_STRONG"
                elif price_below_cloud and ema_bearish_cross:
                    trend = "DOWN"
                    if macd_bearish_confirm: trend = "DOWN_STRONG"
            else:
                 if price_above_cloud and ema_bullish_cross and macd_bullish_confirm:
                      trend = "UP_WEAK"
                 elif price_below_cloud and ema_bearish_cross and macd_bearish_confirm:
                      trend = "DOWN_WEAK"

            # --- 4. Logging Chi Tiết ---
            price_precision_digits = df_htf.attrs.get('price_precision_digits', 4)

            log_msg = (
                f"_get_higher_tf_trend (Snapshot): {symbol} {timeframe} - "
                f"Close={close:.{price_precision_digits}f}, "
                f"EMA50={ema50:.{price_precision_digits}f}, " # type: ignore
                f"EMA200={ema200:.{price_precision_digits}f}, " # type: ignore
                f"ADX={adx:.2f} (Thresh={adx_threshold_htf}), " # type: ignore
                f"MACD={macd:.4f if pd.notna(macd) else 'N/A'}, "
                f"Signal={macd_signal:.4f if pd.notna(macd_signal) else 'N/A'} "
                f"=> Trend='{trend}'"
            )
            logging.debug(log_msg)

            # Trả về kết quả cuối cùng đơn giản hóa (logic GIỮ NGUYÊN)
            if "UP" in trend: return "UP"
            if "DOWN" in trend: return "DOWN"
            return "SIDEWAYS"

        except IndexError as idx_err:
             logging.error(f"_get_higher_tf_trend: IndexError accessing data for {symbol} {timeframe} from snapshot. Check data length and indexing.", exc_info=True)
             return None
        except Exception as e:
            logging.error(f"Unexpected error getting HTF trend for {symbol} {timeframe} from snapshot: {e}", exc_info=True)
            return None
        
    async def _process_symbol_trading_logic(self, symbol: str, current_data_source: Dict[str, pd.DataFrame]):
        func_name = "_process_symbol_trading_logic"
        acquired_locks: List[asyncio.Lock] = [] # Theo dõi lock đã acquire
        # <<< Khởi tạo các biến kết quả tính toán ngoài lock >>>
        entry_signal: Optional[Dict[str, Any]] = None
        agent_signals: Optional[Dict[str, Any]] = None
        final_risk_factor: float = 0.0
        final_sl_price: Optional[float] = None
        active_sentiment_score: float = 0.0
        position_size: float = 0.0 # Khởi tạo position_size
        current_embedding: Optional[np.ndarray] = None

        try:
            # === Phần 1: Logic NGOÀI LOCK - Kiểm tra cơ bản, chuẩn bị data, lấy tín hiệu, tính risk/SL ===

            tf_primary = CONFIG.get("primary_tf", "15m")
            df_primary = current_data_source.get(tf_primary)
            if df_primary is None or df_primary.empty: return # Thoát sớm nếu thiếu data chính

            # Kiểm tra CB (hàm con có lock nội bộ)
            # Chấp nhận rủi ro race condition nhỏ khi đọc cờ `circuit_breaker_triggered` ngoài lock
            async with self.cb_lock: cb_is_active = self.circuit_breaker_triggered
            if cb_is_active: logging.debug(f"{func_name} ({symbol}): Global CB active. Skipping."); return
            # Gọi hàm check (có await và lock nội bộ)
            if await self.check_circuit_breaker(df_primary):
                logging.debug(f"{func_name} ({symbol}): Symbol CB active. Skipping.")
                return

            # Kiểm tra đủ dữ liệu (ngoài lock)
            required_tfs_agent = list(CONFIG.get('required_tfs_hybrid', ())); required_tfs_htf = ["1h", "4h"]
            all_required_tfs = set([tf_primary] + required_tfs_agent + required_tfs_htf)
            min_lengths = { tf: CONFIG.get('HYBRID_SEQ_LEN_MAP', {}).get(tf, CONFIG.get('DECISION_SEQ_LEN', 60) if tf == tf_primary else 2) for tf in all_required_tfs }
            is_data_sufficient = all(tf in current_data_source and isinstance(current_data_source[tf], pd.DataFrame) and len(current_data_source[tf]) >= min_lengths.get(tf, 2) for tf in all_required_tfs)
            if not is_data_sufficient:
                 logging.warning(f"{func_name} ({symbol}): Skipping due to insufficient data.")
                 return

            last_row_primary = df_primary.iloc[-1]; current_price = last_row_primary['close']; current_timestamp = last_row_primary.name
            if pd.isna(current_price) or current_price <= 0:
                 logging.error(f"{func_name} ({symbol}): Invalid current_price. Skipping."); return
            
            lock_acquired_emb = False
            try:
                logging.debug(f"{func_name} ({symbol}): Acquiring embeddings_lock to read...")
                await asyncio.wait_for(self.embeddings_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT / 2) # Timeout ngắn hơn cho đọc
                lock_acquired_emb = True
                # Đọc embedding bên trong lock
                current_embedding_np = self.current_embeddings.get(symbol) # Trả về None nếu key không tồn tại
                # Sao chép nếu cần dùng lâu hoặc chuyển đổi kiểu
                if current_embedding_np is not None:
                     current_embedding = current_embedding_np.copy() # Hoặc torch.from_numpy(..).to(DEVICE) nếu Agent cần Tensor
            except asyncio.TimeoutError:
                 logging.warning(f"{func_name} ({symbol}): Timeout acquiring embeddings_lock. Proceeding without embedding.")
                 current_embedding = None # Đảm bảo là None nếu lỗi lock
            except Exception as emb_read_e:
                 logging.error(f"{func_name} ({symbol}): Error reading embedding: {emb_read_e}")
                 current_embedding = None
            finally:
                if lock_acquired_emb and self.embeddings_lock.locked():
                    self.embeddings_lock.release()
                    logging.debug(f"{func_name} ({symbol}): embeddings_lock released after reading.")

            # Chuẩn bị input và gọi Agent (ngoài lock)
            if self.combined_agent is None or self.decision_model is None or self.hybrid_model is None:
                 logging.warning(f"{func_name} ({symbol}): Agent/Models not available."); return
            # ... (logic chuẩn bị x_decision, x_hybrid_dict) ...
            try: # Tạo tensors
                x_decision = None; x_hybrid_dict = {}; DEVICE = self.decision_model.device
                decision_seq_len = CONFIG.get('DECISION_SEQ_LEN', 60); hybrid_seq_map = CONFIG.get('HYBRID_SEQ_LEN_MAP', {})
                if len(df_primary) >= decision_seq_len:
                    decision_data = df_primary[['open', 'high', 'low', 'close', 'volume']].iloc[-decision_seq_len:].values
                    if not np.isnan(decision_data).any(): x_decision = torch.tensor(decision_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                valid_hybrid = True
                for tf_h in required_tfs_agent:
                    df_tf_h = current_data_source.get(tf_h); seq_len_tf_h = hybrid_seq_map.get(tf_h)
                    if df_tf_h is not None and seq_len_tf_h is not None and len(df_tf_h) >= seq_len_tf_h:
                        hybrid_data_tf_h = df_tf_h[['open', 'high', 'low', 'close', 'volume']].iloc[-seq_len_tf_h:].values
                        if not np.isnan(hybrid_data_tf_h).any(): x_hybrid_dict[tf_h] = torch.tensor(hybrid_data_tf_h, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    else: valid_hybrid = False; break
                if not valid_hybrid: x_hybrid_dict = {}
            except Exception as tensor_e: logging.error(f"Error preparing tensors for {symbol}: {tensor_e}"); return
            if not (x_decision is not None and x_hybrid_dict): logging.warning(f"Cannot call agent for {symbol}: Missing input tensors."); return

            # Gọi agent (ngoài lock)
            try:
                agent_signals = self.combined_agent.get_trade_signals(
                    symbol=symbol, x_decision=x_decision, x_hybrid_dict=x_hybrid_dict,
                    current_price=current_price, current_data=current_data_source,
                    open_position=None, embedding=current_embedding # <<< Truyền embedding >>>
                )
                entry_signal = agent_signals.get('entry_signal')
            except Exception as agent_call_e: logging.error(f"Agent call error for {symbol}: {agent_call_e}"); return

            # Xử lý tín hiệu (ngoài lock)
            if not (entry_signal and isinstance(entry_signal, dict) and entry_signal.get("direction") and entry_signal.get("signal_quality") not in ['NONE', 'CONFLICT']):
                signal_status = agent_signals.get('entry_signal', {}).get('signal_quality', 'NO_SIGNAL') if isinstance(agent_signals.get('entry_signal'), dict) else 'NO_SIGNAL'
                logging.debug(f"{func_name} ({symbol}): No valid entry signal at {current_timestamp}. Status: {signal_status}")
                return # Không có tín hiệu hợp lệ, thoát

            # ... (Phần logic lọc tín hiệu, tính risk, tính SL - từ 5.1 đến 5.6 - NGOÀI LOCK) ...
            signal_direction = entry_signal['direction']; win_prob = entry_signal.get("win_prob", 0.0); signal_quality = entry_signal.get("signal_quality", 'NONE'); signal_timestamp = entry_signal.get('timestamp', current_timestamp)
            logging.info(f"--- [{self.mode.upper()}] Agent Signal Received [{symbol} @ {signal_timestamp}] ---"); logging.info(f"  Direction: {signal_direction}, Q: {signal_quality}, P: {win_prob:.3f}")
            # HTF Filter
            df_1h_snapshot = current_data_source.get.get("1h") # Lấy DataFrame cho "1h"
            trend_1h = self._get_higher_timeframe_trend(symbol, "1h", df_1h_snapshot)

            df_4h_snapshot = current_data_source.get.get("4h") # Lấy DataFrame cho "4h"
            trend_4h = self._get_higher_timeframe_trend(symbol, "4h", df_4h_snapshot)

            htf_confirmation_factor = 1.0
            filter_by_htf = False
            is_strong_contradiction = (signal_direction == "LONG" and trend_1h == "DOWN" and trend_4h == "DOWN") or (signal_direction == "SHORT" and trend_1h == "UP" and trend_4h == "UP")
            if is_strong_contradiction: logging.warning(f"Trade Filtered (HTF) for {symbol}."); return
            is_partial_contradiction = (signal_direction == "LONG" and (trend_1h == "DOWN" or trend_4h == "DOWN")) or (signal_direction == "SHORT" and (trend_1h == "UP" or trend_4h == "UP"))
            is_strong_agreement = (signal_direction == "LONG" and trend_1h == "UP" and trend_4h == "UP") or (signal_direction == "SHORT" and trend_1h == "DOWN" and trend_4h == "DOWN")
            if is_partial_contradiction and not is_strong_agreement: htf_confirmation_factor = 0.6; 
            elif is_strong_agreement: htf_confirmation_factor = 1.1
            # Sentiment Filter
            active_sentiment_score = 0.0
            if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer is not None:
                try: sentiment_details = self.sentiment_analyzer.get_detailed_active_sentiment(signal_timestamp); active_sentiment_score = sentiment_details.get("total_score_adj_confidence", 0.0)
                except Exception: pass
            veto_thresh = CONFIG.get("sentiment_veto_threshold", 0.65)
            if (active_sentiment_score < -veto_thresh and signal_direction == "LONG") or (active_sentiment_score > veto_thresh and signal_direction == "SHORT"): logging.warning(f"Trade Filtered (Sentiment Veto) for {symbol}."); return
            # DQN Filter
            dqn_action = 1; reduce_risk_aggressively = False
            if self.drl_model and self.trading_env_scaler and CONFIG.get("dqn_size_adj_enabled", True):
                obs_dqn = self._get_observation_for_dqn(df_primary)
                if obs_dqn is not None:
                    try:
                        expected_dqn_dim = getattr(self.trading_env_scaler, 'n_features_in_', None)
                        if expected_dqn_dim is not None and obs_dqn.shape[0] == expected_dqn_dim:
                            scaled_obs = self.trading_env_scaler.transform(obs_dqn.reshape(1,-1)); act, _ = self.drl_model.predict(scaled_obs, deterministic=True); dqn_action = int(act.item())
                        else: dqn_action = 1
                    except Exception: dqn_action = 1
                win_prob_override = CONFIG.get("win_prob_dqn_override", 0.80)
                if dqn_action == 0 and win_prob < win_prob_override: logging.warning(f"Trade Filtered (DQN Veto) for {symbol}."); return
                elif dqn_action == 0: reduce_risk_aggressively = True
            # Risk Factor Calc
            base_risk = 0.0; risk_map = {'STRONG': 0.05, 'MEDIUM': 0.03, 'WEAK': 0.015}; base_risk = risk_map.get(signal_quality, 0.0)
            if base_risk <= 0: logging.info(f"Skipping trade {symbol} (base_risk=0)."); return
            risk_adj_wp = np.clip(base_risk * (1 + (win_prob - 0.65) * 0.5), 0.01, CONFIG.get("max_account_risk", 0.1)); risk_adj_htf = risk_adj_wp * htf_confirmation_factor
            sent_inf = CONFIG.get("sentiment_influence_factor", 0.15); sent_mult = np.clip(1.0 + active_sentiment_score * sent_inf, 0.6, 1.4); risk_adj_sent = risk_adj_htf * sent_mult
            dqn_adj = 1.0; boost_thresh = CONFIG.get("win_prob_dqn_boost_thresh", 0.70)
            if reduce_risk_aggressively: dqn_adj = 0.4; 
            elif dqn_action == 2: dqn_adj = 1.25 if win_prob >= boost_thresh else 1.1
            risk_step_d = risk_adj_sent * dqn_adj; mods = CONFIG.get("symbol_risk_modifiers", {}); def_mod = mods.get("DEFAULT", 1.0); base_curr = symbol.split('/')[0].upper(); sym_mod = mods.get(base_curr, def_mod)
            final_risk_factor = np.clip(risk_step_d * sym_mod, 0.005, CONFIG.get("max_account_risk", 0.1))
            # SL Calc
            potential_sl = entry_signal.get('potential_sl_price'); final_sl_price = potential_sl
            if final_sl_price is None: logging.error(f"Missing potential SL for {symbol}. Skipping."); return
            if self.stop_system and CONFIG.get("use_intelligent_stop_system", True):
                iss_data = current_data_source
                if iss_data and tf_primary in iss_data and not iss_data[tf_primary].empty:
                    try:
                        pos_iss = { "symbol": symbol, "entry": entry_signal['entry_price'], "initial_sl_adjusted": potential_sl, "direction": signal_direction, "duration": 0 }
                        refined_sl = self.stop_system.calculate_sl(pos_iss, iss_data); is_valid_iss = False; p_prec = self._safe_get_price_precision_digits(symbol)
                        if signal_direction == 'LONG' and refined_sl < entry_signal['entry_price']: is_valid_iss = True; final_sl_price = max(final_sl_price, refined_sl)
                        elif signal_direction == 'SHORT' and refined_sl > entry_signal['entry_price']: is_valid_iss = True; final_sl_price = min(final_sl_price, refined_sl)
                    except Exception: pass # Use agent SL if ISS errors
            if reduce_risk_aggressively: # Tighten SL by DQN
                 entry_sl = entry_signal['entry_price']; dir_sl = signal_direction; tight_f = 0.5; p_prec_sl = self._safe_get_price_precision_digits(symbol); min_dist = 10**(-p_prec_sl) * 5
                 orig_sl = final_sl_price
                 if dir_sl == 'LONG': tight_sl_val = final_sl_price + abs(entry_sl - final_sl_price) * tight_f; final_sl_price = min(tight_sl_val, entry_sl - min_dist)
                 elif dir_sl == 'SHORT': tight_sl_val = final_sl_price - abs(entry_sl - final_sl_price) * tight_f; final_sl_price = max(tight_sl_val, entry_sl + min_dist)
                 final_sl_price = round(final_sl_price, p_prec_sl)
            try: # Max loss rule
                risk_max = risk_map.get(signal_quality, 0.015); max_loss = entry_signal['entry_price'] * risk_max; sl_b4_max = final_sl_price
                if signal_direction == "LONG": max_loss_sl = entry_signal['entry_price'] - max_loss; final_sl_price = max(final_sl_price, max_loss_sl)
                elif signal_direction == "SHORT": max_loss_sl = entry_signal['entry_price'] + max_loss; final_sl_price = min(final_sl_price, max_loss_sl)
            except Exception: final_sl_price = sl_b4_max
            if (signal_direction == 'LONG' and final_sl_price >= entry_signal['entry_price']) or (signal_direction == 'SHORT' and final_sl_price <= entry_signal['entry_price']): logging.error(f"Invalid Final SL for {symbol}. Skipping."); return


            # === Phần 2: Acquire Locks cho Critical Section ===
            logging.debug(f"{func_name}: Acquiring locks for entry decision ({symbol})...")
            acquired_balance = await asyncio.wait_for(self.balance_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_balance: raise asyncio.TimeoutError("Timeout acquiring balance_lock")
            acquired_locks.append(self.balance_lock)

            acquired_positions = await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_positions: raise asyncio.TimeoutError("Timeout acquiring positions_lock")
            acquired_locks.append(self.positions_lock)
            logging.debug(f"{func_name}: Locks acquired for entry decision ({symbol}).")

            # === Phần 3: Logic BÊN TRONG LOCK ===

            # --- Kiểm tra lại vị thế ---
            if symbol in self.open_positions:
                logging.debug(f"{func_name} ({symbol}): Position opened concurrently. Skipping.")
                return # finally sẽ giải phóng lock

            # --- Tính Size ---
            # Hàm này đọc self.balance (đã có lock)
            position_size = self.calculate_position_size(
                entry_signal, final_sl_price, final_risk_factor, active_sentiment_score
            )

            # --- Kiểm tra Size và Exposure ---
            if position_size <= 0: logging.info(f"Skipping {symbol}: Final size <= 0."); return
            min_qty = self._safe_get_min_qty(symbol)
            if position_size < min_qty: logging.info(f"Skipping {symbol}: Size < MinQty."); return
            # Hàm này đọc self.balance, self.exposure (đã có lock)
            if not self.check_position_exposure(position_size, entry_signal["entry_price"], symbol):
                logging.warning(f"Skipping {symbol}: Exposure limit."); return

            # --- Log và Cập nhật State (Mở Vị Thế) ---
            logging.info(f"--- [{self.mode.upper()}] Attempting {signal_direction} {symbol} ---")
            # ... (Log chi tiết size, entry, SL, context như cũ) ...
            amount_prec = self._safe_get_amount_precision_digits(symbol); price_prec = self._safe_get_price_precision_digits(symbol)
            logging.info(f"  Calculated Size: {position_size:.{amount_prec}f}"); logging.info(f"  Target Entry: ~{entry_signal['entry_price']:.{price_prec}f}"); logging.info(f"  Calculated SL: {final_sl_price:.{price_prec}f}")
            logging.info(f"  Context: Q:{signal_quality}, P:{win_prob:.3f}, HTF(F:{htf_confirmation_factor:.2f}), Sent(S:{active_sentiment_score:.2f}), DQN(A:{dqn_action}), FinalRisk({final_risk_factor:.4f})")


            # <<< Gọi _add_open_position (cần caller giữ locks) >>>
            position_added = self._add_open_position(
                signal=entry_signal,
                position_size=position_size,
                sl_price=final_sl_price,
                agent_signals=agent_signals # Truyền agent_signals đầy đủ
            )

            if not position_added:
                 logging.error(f"{func_name}: Failed to add open position state for {symbol}!")
                 # Xử lý lỗi nghiêm trọng ở đây nếu cần
            else:
                # --- Gửi lệnh live (VẪN TRONG LOCK - Tạm thời comment out) ---
                if self.mode == 'live':
                     logging.info(f"Submitting LIVE entry order placeholder for {symbol}...")
                     # --- TODO: Thêm logic gửi lệnh thực tế ---
                     # order_result = await self._execute_entry_order(...)
                     # Xử lý order_result:
                     # if order_result and order_result.get('status') in ['closed', 'filled']:
                     #     actual_entry_price = order_result.get('average', entry_signal['entry_price'])
                     #     actual_filled_size = order_result.get('filled', position_size)
                     #     # Có thể cần cập nhật lại state trong self.open_positions[symbol] với giá/size thực tế
                     #     # Hoặc xử lý PnL ngay lập tức nếu lệnh khớp một phần?
                     #     logging.info(f"LIVE order filled simulation: {symbol} {signal_direction}. Avg Price: {actual_entry_price}, Filled: {actual_filled_size}")
                     # else:
                     #     logging.error(f"LIVE entry order for {symbol} failed or not filled. Status: {order_result.get('status') if order_result else 'N/A'}")
                     #     # <<< QUAN TRỌNG: Nếu lệnh live lỗi, cần rollback state đã thêm? >>>
                     #     # Ví dụ: Gọi hàm _remove_open_position(symbol) (cần lock)
                     #     logging.warning(f"Attempting to remove position state for {symbol} due to failed live order.")
                     #     # self._remove_open_position(symbol) # Hàm này cần được tạo và quản lý lock
                     pass # Giữ placeholder

        except asyncio.TimeoutError as te:
            logging.error(f"{func_name} ({symbol}): Timeout acquiring lock: {te}. Skipping entry.")
        except Exception as e:
            logging.error(f"CRITICAL Error in {func_name} ({self.mode}) for {symbol}: {e}", exc_info=True)
            logging.error(traceback.format_exc()) # In traceback đầy đủ
        finally:
            # === Giải phóng locks đã acquire ===
            for lock in reversed(acquired_locks):
                if lock.locked():
                    lock.release()
            logging.debug(f"{func_name}: Locks released for {symbol}.")

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


    async def _manage_open_positions(self, current_data_source: Dict[str, Dict[str, pd.DataFrame]]):
        """Quản lý các vị thế đang mở (Có lock đọc keys)."""
        func_name = "_manage_open_positions"
        if self.combined_agent is None: logging.warning(f"{func_name}: Agent not available."); return

        symbols_to_manage: List[str] = []
        acquired_lock = False
        try:
            # === Lock ngắn gọn để đọc danh sách keys ===
            logging.debug(f"{func_name}: Acquiring positions_lock to get keys...")
            await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            acquired_lock = True
            logging.debug(f"{func_name}: positions_lock acquired.")
            # Lấy bản sao của list keys
            symbols_to_manage = list(self.open_positions.keys())
        except asyncio.TimeoutError:
            logging.error(f"{func_name}: Timeout acquiring positions_lock to get keys. Skipping management cycle.")
            return # Không thể lấy list an toàn, bỏ qua chu kỳ này
        except Exception as e:
            logging.error(f"{func_name}: Error getting open position keys: {e}")
            return # Bỏ qua nếu lỗi
        finally:
            if acquired_lock and self.positions_lock.locked():
                self.positions_lock.release()
                logging.debug(f"{func_name}: positions_lock released after getting keys.")

        # Nếu không có vị thế nào để quản lý
        if not symbols_to_manage:
            # logging.debug(f"{func_name}: No open positions to manage.")
            return

        # Tạo task để xử lý từng symbol song song
        management_tasks = [
            self._process_single_symbol_management(symbol, current_data_source)
            for symbol in symbols_to_manage
        ]

        if management_tasks:
            logging.debug(f"{func_name}: Dispatching management tasks for {len(management_tasks)} symbols.")
            results = await asyncio.gather(*management_tasks, return_exceptions=True)
            # Log lỗi từ các task quản lý nếu cần
            for symbol, result in zip(symbols_to_manage, results):
                 if isinstance(result, Exception):
                      logging.error(f"Error managing symbol {symbol}: {result}", exc_info=result)


    async def _process_single_symbol_management(self, symbol: str, current_data_source: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Xử lý quản lý chi tiết cho một symbol (Quản lý lock nội bộ).
        """
        func_name = "_process_single_symbol_management"
        position_info_copy: Optional[Dict[str, Any]] = None
        backtest_close_info: Optional[Dict[str, Any]] = None
        agent_signals: Optional[Dict[str, Any]] = None
        acquired_locks: List[asyncio.Lock] = []

        try:
            # === Bước 1: Đọc state và kiểm tra Backtest SL/TP (Trong lock đọc) ===
            lock_acquired_read = False
            try:
                logging.debug(f"{func_name} ({symbol}): Acquiring positions_lock for read/backtest check...")
                await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                lock_acquired_read = True
                logging.debug(f"{func_name} ({symbol}): positions_lock acquired for read/backtest check.")

                position_info_locked = self.open_positions.get(symbol)
                if not position_info_locked:
                    logging.debug(f"{func_name} ({symbol}): Position closed before management could start.")
                    return # Đã đóng, thoát sớm

                # Sao chép thông tin cần thiết ra ngoài lock
                position_info_copy = copy.deepcopy(position_info_locked)

                # --- Kiểm tra Backtest SL/TP Hit (bên trong lock đọc) ---
                if self.mode == 'backtest':
                    symbol_data_slice = current_data_source.get(symbol) # Lấy slice data
                    tf_primary = CONFIG.get("primary_tf", "15m")
                    if symbol_data_slice and tf_primary in symbol_data_slice and not symbol_data_slice[tf_primary].empty:
                        df_primary = symbol_data_slice[tf_primary]
                        try:
                            current_candle = df_primary.iloc[-1]; candle_high = current_candle.get('high'); candle_low = current_candle.get('low'); current_timestamp = current_candle.name
                            if not (any(v is None for v in [candle_high, candle_low]) or pd.isna(current_timestamp)):
                                sl_price = position_info_locked.get('stop_loss_price'); trailing_tp = position_info_locked.get('trailing_tp_level', 0); min_locked = position_info_locked.get('min_locked_tp_level', 0); direction = position_info_locked.get('direction', '').lower(); entry = position_info_locked.get('entry_price')
                                if sl_price is not None and entry is not None and direction:
                                    min_tp = CombinedTradingAgent._calculate_rr_price(entry, sl_price, min_locked, direction) if min_locked > 0 else None
                                    trail_tp = CombinedTradingAgent._calculate_rr_price(entry, sl_price, trailing_tp, direction) if trailing_tp > 0 else None
                                    exit_p_bt = None; exit_r_bt = None
                                    if direction == 'l':
                                        if candle_low <= sl_price: exit_p_bt, exit_r_bt = sl_price, "SL_HIT_BT"
                                        elif min_tp is not None and candle_low <= min_tp: exit_p_bt, exit_r_bt = min_tp, "MIN_TP_FLOOR_HIT_BT"
                                        elif trail_tp is not None and candle_low <= trail_tp and trailing_tp >= min_locked: exit_p_bt, exit_r_bt = trail_tp, f"TRAILING_TP_{trailing_tp}_HIT_BT"
                                    elif direction == 's':
                                        if candle_high >= sl_price: exit_p_bt, exit_r_bt = sl_price, "SL_HIT_BT"
                                        elif min_tp is not None and candle_high >= min_tp: exit_p_bt, exit_r_bt = min_tp, "MIN_TP_FLOOR_HIT_BT"
                                        elif trail_tp is not None and candle_high >= trail_tp and trailing_tp >= min_locked: exit_p_bt, exit_r_bt = trail_tp, f"TRAILING_TP_{trailing_tp}_HIT_BT"
                                    if exit_p_bt is not None:
                                        backtest_close_info = {'symbol': symbol, 'exit_price': exit_p_bt, 'reason': exit_r_bt, 'timestamp': current_timestamp, 'position_info': copy.deepcopy(position_info_locked)}
                                        logging.info(f"Backtest: Marked {symbol} for closure ({exit_r_bt})")
                        except Exception as bt_e: logging.error(f"Error in Backtest check for {symbol}: {bt_e}")
            finally:
                if lock_acquired_read and self.positions_lock.locked():
                    self.positions_lock.release(); logging.debug(f"{func_name} ({symbol}): positions_lock released after read/backtest check.")

            # === Bước 2: Xử lý đóng lệnh Backtest (NGOÀI lock đọc, cần lock ghi) ===
            if backtest_close_info:
                await self._handle_backtest_close(backtest_close_info) # Hàm này tự quản lý lock ghi
                return # Kết thúc xử lý cho symbol này

            # === Bước 3: Gọi Agent (NGOÀI LOCK) ===
            if position_info_copy is None: return # Không có thông tin vị thế để quản lý
            if self.combined_agent is None: logging.warning(f"{func_name} ({symbol}): Agent not available."); return

            symbol_data_slice = current_data_source.get(symbol) # Lấy lại slice data (nếu cần)
            tf_primary = CONFIG.get("primary_tf", "15m")
            # Kiểm tra lại dữ liệu slice cần cho agent
            required_tfs_agent = list(CONFIG.get('required_tfs_hybrid', ())); all_req_agent_tfs = set([tf_primary] + required_tfs_agent)
            if not symbol_data_slice or not all(tf in symbol_data_slice and isinstance(symbol_data_slice[tf], pd.DataFrame) and not symbol_data_slice[tf].empty for tf in all_req_agent_tfs):
                logging.warning(f"{func_name} ({symbol}): Missing data slice for Agent."); return

            df_primary = symbol_data_slice[tf_primary]; current_price = df_primary['close'].iloc[-1]; current_timestamp_agent = df_primary.index[-1]
            if pd.isna(current_price): logging.error(f"{func_name} ({symbol}): Invalid current price for Agent."); return

            # Chuẩn bị tensors (ngoài lock)
            x_decision = None; x_hybrid_dict = {} # ... (logic tạo tensor như cũ) ...
            try:
                DEVICE = self.decision_model.device; decision_seq_len = CONFIG.get('DECISION_SEQ_LEN', 60); hybrid_seq_map = CONFIG.get('HYBRID_SEQ_LEN_MAP', {})
                if len(df_primary) >= decision_seq_len:
                    d_data = df_primary[['open','high','low','close','volume']].iloc[-decision_seq_len:].values
                    if not np.isnan(d_data).any(): x_decision = torch.tensor(d_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                v_hybrid = True
                for tf_h in required_tfs_agent:
                    df_h = symbol_data_slice.get(tf_h); seq_h = hybrid_seq_map.get(tf_h)
                    if df_h is not None and seq_h is not None and len(df_h) >= seq_h:
                        h_data = df_h[['open','high','low','close','volume']].iloc[-seq_h:].values
                        if not np.isnan(h_data).any(): x_hybrid_dict[tf_h] = torch.tensor(h_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        else: v_hybrid=False; break
                    else: v_hybrid=False; break
                if not v_hybrid: x_hybrid_dict = {}
            except Exception as te: logging.error(f"{func_name} ({symbol}): Error preparing tensors: {te}"); return
            if not (x_decision is not None and x_hybrid_dict): logging.warning(f"{func_name} ({symbol}): Missing tensors for Agent."); return

            # Gọi agent (ngoài lock)
            try:
                agent_signals = self.combined_agent.get_trade_signals(
                    symbol=symbol, x_decision=x_decision, x_hybrid_dict=x_hybrid_dict,
                    current_price=current_price, current_data=symbol_data_slice, # Truyền slice
                    open_position=position_info_copy # Truyền bản copy
                )
            except Exception as agent_e: logging.error(f"{func_name} ({symbol}): Agent call error: {agent_e}"); return


            # === Bước 4: Cập nhật State từ Agent (TRONG LOCK GHI) ===
            if agent_signals: # Chỉ cập nhật nếu agent trả về kết quả
                lock_acquired_write = False
                try:
                    logging.debug(f"{func_name} ({symbol}): Acquiring positions_lock for agent update...")
                    await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                    lock_acquired_write = True
                    acquired_locks.append(self.positions_lock) # Thêm vào list để giải phóng
                    logging.debug(f"{func_name} ({symbol}): positions_lock acquired for agent update.")

                    # Kiểm tra lại vị thế tồn tại bên trong lock
                    current_pos_state = self.open_positions.get(symbol)
                    if not current_pos_state:
                        logging.debug(f"{func_name} ({symbol}): Position closed before agent results could be applied.")
                        return # Thoát

                    # Cập nhật state
                    new_trailing_tp = agent_signals.get('trailing_tp_level', current_pos_state.get('trailing_tp_level', 0))
                    new_min_locked_tp = agent_signals.get('min_locked_tp_level', current_pos_state.get('min_locked_tp_level', 0))
                    should_update_monitor = False
                    if new_trailing_tp != current_pos_state.get('trailing_tp_level'):
                        current_pos_state['trailing_tp_level'] = new_trailing_tp; should_update_monitor = True
                        logging.info(f"[{self.mode.upper()}] Updating Trailing TP for {symbol}: -> {new_trailing_tp}")
                    if new_min_locked_tp != current_pos_state.get('min_locked_tp_level'):
                        current_pos_state['min_locked_tp_level'] = new_min_locked_tp; should_update_monitor = True
                        logging.info(f"[{self.mode.upper()}] Updating Min Locked TP for {symbol}: -> {new_min_locked_tp}")
                    current_pos_state['last_agent_signals'] = copy.deepcopy(agent_signals)

                    # --- Cập nhật Monitor (NGOÀI LOCK, tạo task) ---
                    if should_update_monitor and self.realtime_monitor and self.mode == 'live':
                        asyncio.create_task(self._update_monitor_managed_position(symbol))

                except asyncio.TimeoutError:
                    logging.error(f"{func_name} ({symbol}): Timeout acquiring positions_lock for agent update. State not updated.")
                    # Không thể cập nhật state, nhưng logic đóng lệnh agent vẫn có thể chạy nếu agent_signals tồn tại
                except Exception as write_e:
                    logging.error(f"{func_name} ({symbol}): Error updating position state from agent: {write_e}")

            # === Bước 5: Xử lý đóng lệnh từ Agent (NGOÀI LOCK) ===
            if agent_signals: # Chỉ xử lý nếu có kết quả từ agent
                exit_signal_agent = agent_signals.get('exit_signal', 'hold')
                if exit_signal_agent != 'hold' and not exit_signal_agent.startswith('close_at_t'):
                    logging.info(f"[{self.mode.upper()}] Agent requested position close for {symbol}: {exit_signal_agent}.")
                    # Tạo task để xử lý đóng lệnh (hàm này tự quản lý lock)
                    asyncio.create_task(self._handle_exit_signal(symbol, exit_signal_agent, current_price))

        except asyncio.TimeoutError as te_outer:
             # Lỗi timeout ở lần acquire lock đầu tiên (đọc/backtest)
             logging.error(f"{func_name} ({symbol}): Timeout acquiring initial positions_lock: {te_outer}")
        except Exception as e:
            logging.error(f"{func_name}: Error managing position {symbol}: {e}", exc_info=True)
            logging.error(traceback.format_exc())
        finally:
             # Giải phóng tất cả các lock đã acquire trong acquired_locks
             for lock in reversed(acquired_locks):
                  if lock.locked():
                       lock.release()
             logging.debug(f"{func_name}: Management locks released for {symbol}.")
    
    async def _handle_backtest_close(self, close_info: dict):
         """Helper xử lý đóng lệnh trong backtest (có lock)."""
         func_name = "_handle_backtest_close"
         symbol = close_info['symbol']
         acquired_locks: List[asyncio.Lock] = []
         try:
             # === Acquire locks ===
             logging.debug(f"{func_name}: Acquiring locks for {symbol}...")
             acquired_balance = await asyncio.wait_for(self.balance_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
             if not acquired_balance: raise asyncio.TimeoutError("Timeout balance_lock")
             acquired_locks.append(self.balance_lock)

             acquired_positions = await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
             if not acquired_positions: raise asyncio.TimeoutError("Timeout positions_lock")
             acquired_locks.append(self.positions_lock)

             acquired_history = await asyncio.wait_for(self.trade_history_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
             if not acquired_history: raise asyncio.TimeoutError("Timeout history_lock")
             acquired_locks.append(self.trade_history_lock)
             logging.debug(f"{func_name}: Locks acquired for {symbol}.")

             # === Logic bên trong locks ===
             if symbol in self.open_positions: # Kiểm tra lại trong lock
                  pos_info = close_info['position_info'] # Lấy thông tin từ lúc hit
                  exit_p = close_info['exit_price']
                  reason = close_info['reason']
                  timestamp_close = close_info['timestamp']

                  # --- Tính PnL ---
                  entry_p = pos_info['entry_price']
                  size = pos_info['position_size']
                  dir_orig = pos_info['direction']
                  fee_rate = CONFIG.get("fee_rate", 0.0005)
                  pnl = 0.0 # Khởi tạo

                  if entry_p is None or size is None or dir_orig is None:
                       logging.error(f"{func_name} ({symbol}): Missing critical position info for PnL calc.")
                  elif size <= 0:
                       logging.warning(f"{func_name} ({symbol}): Position size is zero or negative, PnL = 0.")
                  else:
                       if dir_orig == "LONG":
                           raw_pnl = (exit_p - entry_p) * size
                       elif dir_orig == "SHORT":
                           raw_pnl = (entry_p - exit_p) * size
                       else:
                           raw_pnl = 0.0
                           logging.warning(f"{func_name} ({symbol}): Unknown direction '{dir_orig}'. PnL = 0.")
                       # Tính phí
                       fees = (abs(entry_p * size) + abs(exit_p * size)) * fee_rate
                       pnl = raw_pnl - fees
                       logging.info(f"{func_name} ({symbol}): Closing due to {reason}. Entry={entry_p:.4f}, Exit={exit_p:.4f}, Size={size:.6f}, RawPnL={raw_pnl:.4f}, Fees={fees:.4f}, NetPnL={pnl:.4f}")

                  # --- Gọi hàm cập nhật state ---
                  # Hàm _update_balance sẽ log lại thông tin đóng lệnh chi tiết
                  await self._update_balance(
                      pnl, size if size is not None else 0.0, # Truyền size hoặc 0 nếu lỗi
                      entry_p if entry_p is not None else exit_p, # Truyền entry hoặc exit nếu lỗi
                      exit_p, symbol, dir_orig if dir_orig else "UNKNOWN",
                      timestamp_close
                  )
             else:
                  # Vị thế đã bị đóng bởi một logic khác trước đó
                  logging.warning(f"{func_name}: Position {symbol} already closed before backtest hit ({close_info['reason']}) could be processed.")

         except asyncio.TimeoutError as te:
             logging.error(f"{func_name}: Timeout acquiring lock for {symbol}: {te}")
             # State có thể không nhất quán nếu lỗi xảy ra sau khi một phần state đã được cập nhật (dù khó xảy ra ở đây)
         except KeyError as ke:
             logging.error(f"{func_name}: Missing key in close_info for {symbol}: {ke}")
         except Exception as e:
             logging.error(f"{func_name}: Error closing backtest position {symbol}: {e}", exc_info=True)
         finally:
             # === Giải phóng locks ===
             logging.debug(f"{func_name}: Releasing locks for {symbol}.")
             for lock in reversed(acquired_locks):
                  try:
                      if lock.locked(): lock.release()
                  except RuntimeError as rel_e: logging.error(f"{func_name}: Error releasing lock {lock}: {rel_e}")
             logging.debug(f"{func_name}: Locks released for {symbol}.")

    async def _handle_exit_signal(self, symbol: str, reason: str, exit_price: float):
        """Xử lý đóng lệnh từ Agent hoặc Monitor (có lock)."""
        logging.info(f"Handling exit signal for {symbol}. Reason: {reason}. Price: {exit_price}")
        acquired_locks = []
        try:
            acquired_balance = await asyncio.wait_for(self.balance_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_balance: raise asyncio.TimeoutError("Timeout balance_lock")
            acquired_locks.append(self.balance_lock)
            acquired_positions = await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_positions: raise asyncio.TimeoutError("Timeout positions_lock")
            acquired_locks.append(self.positions_lock)
            acquired_history = await asyncio.wait_for(self.trade_history_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_history: raise asyncio.TimeoutError("Timeout history_lock")
            acquired_locks.append(self.trade_history_lock)

            if symbol not in self.open_positions: logging.warning(f"_handle_exit_signal: Position {symbol} already closed."); return
            position_info = self.open_positions[symbol]
            # ... (Tính PnL) ...
            entry_p=position_info['entry_price']; size=position_info['position_size']; dir_orig=position_info['direction']; fee=CONFIG.get("fee_rate", 0.0005); ts=pd.Timestamp.now(tz='UTC')
            if dir_orig=="LONG": raw_pnl=(exit_price-entry_p)*size;
            elif dir_orig=="SHORT": raw_pnl=(entry_p-exit_price)*size; 
            else: raw_pnl=0.0
            fees=(abs(entry_p*size)+abs(exit_price*size))*fee; pnl=raw_pnl-fees
            await self._update_balance(pnl, size, entry_p, exit_price, symbol, dir_orig, ts) # Cập nhật state
            # --- TODO: Gửi lệnh đóng live ---

        except asyncio.TimeoutError as te: logging.error(f"Timeout handling exit signal for {symbol}: {te}")
        except Exception as e: logging.error(f"Error handling exit signal for {symbol}: {e}", exc_info=True)
        finally:
            for lock in reversed(acquired_locks):
                if lock.locked(): lock.release()
            # Deactivate Monitor (ngoài lock)
            if self.realtime_monitor and self.mode == 'live':
                self.realtime_monitor.update_thresholds(symbol, active=False)

    async def _handle_monitor_threshold_hit(self, signal_data: Dict[str, Any]):
        func_name = "_handle_monitor_threshold_hit"
        symbol = signal_data.get("symbol")
        threshold_type = signal_data.get("threshold_type")
        level_or_price = signal_data.get("level_or_price") # Giá trị ngưỡng bị chạm
        current_price = signal_data.get("current_price") # Giá hiện tại khi event (dùng làm fallback)
        timestamp_ms = signal_data.get("timestamp")

        # === Bước 1: Kiểm tra đầu vào cơ bản (NGOÀI LOCK) ===
        if not symbol: logging.error(f"{func_name}: Received signal without symbol."); return
        if level_or_price is None: logging.error(f"{func_name} ({symbol}): Received signal without level/price."); return
        if current_price is None: logging.warning(f"{func_name} ({symbol}): Received signal without current_price (used as fallback)."); # Vẫn có thể tiếp tục nếu level_or_price có

        # <<< THÊM: Kiểm tra và xử lý timestamp_ms >>>
        timestamp: pd.Timestamp
        if timestamp_ms is None or not isinstance(timestamp_ms, (int, float)) or timestamp_ms <= 0:
            logging.warning(f"{func_name} ({symbol}): Invalid or missing timestamp_ms ({timestamp_ms}). Using current time.")
            timestamp = pd.Timestamp.now(tz='UTC')
        else:
            try:
                timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True)
            except (ValueError, TypeError) as ts_e:
                logging.error(f"{func_name} ({symbol}): Error converting timestamp_ms {timestamp_ms} to datetime: {ts_e}. Using current time.")
                timestamp = pd.Timestamp.now(tz='UTC')

        acquired_locks: List[asyncio.Lock] = []
        position_info_local: Optional[Dict[str, Any]] = None
        pnl_calculated: Optional[float] = None
        exit_price: Optional[float] = None # Khởi tạo exit_price

        try:
            # === Bước 2: Acquire locks ===
            logging.debug(f"{func_name} ({symbol}): Acquiring locks...")
            acquired_balance = await asyncio.wait_for(self.balance_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT); acquired_locks.append(self.balance_lock)
            acquired_positions = await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT); acquired_locks.append(self.positions_lock)
            acquired_history = await asyncio.wait_for(self.trade_history_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT); acquired_locks.append(self.trade_history_lock)
            logging.debug(f"{func_name} ({symbol}): Locks acquired.")

            # === Bước 3: Logic bên trong locks ===
            if symbol not in self.open_positions:
                logging.warning(f"{func_name} ({symbol}): Position already closed before threshold hit ({threshold_type}) could be processed inside lock.")
                return

            position_info_local = self.open_positions[symbol].copy()
            logging.warning(f"--- MONITOR EVENT PROCESSING [{symbol}] --- Type: {threshold_type}, Trigger Price: {level_or_price}, Timestamp: {timestamp}")

            # --- Xác định Exit Price ---
            exit_price = level_or_price # Mặc định dùng giá chạm ngưỡng

            if threshold_type == "TRAILING_TP":
                logging.debug(f"{func_name} ({symbol}): Threshold is Trailing TP, attempting to get actual price for level {level_or_price}...")
                # level_or_price ở đây là RR level (ví dụ: 1.0, 1.5)
                actual_tp_price = self._get_price_for_tp_level(symbol, float(level_or_price)) # Chuyển level sang float
                if actual_tp_price is not None:
                    exit_price = actual_tp_price
                    logging.info(f"{func_name} ({symbol}): Using actual TP price {exit_price:.4f} for level {level_or_price}")
                else:
                    # <<< THÊM: Sử dụng current_price làm fallback nếu không lấy được giá TP >>>
                    exit_price = current_price if current_price is not None else level_or_price # Ưu tiên current_price nếu có
                    logging.warning(f"{func_name} ({symbol}): Could not get exact price for TP level {level_or_price}. Using fallback price {exit_price:.4f}.")

            # --- Tính PnL ---
            entry_price = position_info_local.get('entry_price')
            position_size = position_info_local.get('position_size')
            direction = position_info_local.get('direction')
            fee_rate = CONFIG.get("fee_rate", 0.0005)
            pnl = 0.0

            if entry_price is None or position_size is None or direction is None or position_size <= 0:
                logging.error(f"{func_name} ({symbol}): Invalid position info for PnL calc.")
            elif exit_price is None: # Kiểm tra exit_price
                 logging.error(f"{func_name} ({symbol}): Could not determine valid exit price. Cannot calculate PnL.")
            else:
                # Tính toán PnL chỉ khi có đủ thông tin
                if direction == "LONG": raw_pnl = (exit_price - entry_price) * position_size
                elif direction == "SHORT": raw_pnl = (entry_price - exit_price) * position_size
                else: raw_pnl = 0.0
                fees = (abs(entry_price * position_size) + abs(exit_price * position_size)) * fee_rate
                pnl = raw_pnl - fees
                pnl_calculated = pnl
                logging.info(f"{func_name} ({symbol}): PnL calculated: {pnl:.4f}")

            # --- Gọi hàm cập nhật state ---
            # _update_balance sẽ log thông tin đóng lệnh chi tiết và xóa khỏi open_positions
            await self._update_balance(
                pnl_calculated if pnl_calculated is not None else 0.0, # Truyền PnL đã tính hoặc 0
                position_size if position_size else 0.0,
                entry_price if entry_price else (exit_price if exit_price is not None else 0.0), # Giá trị mặc định an toàn
                exit_price if exit_price is not None else 0.0, # Giá trị mặc định an toàn
                symbol, direction if direction else "UNKNOWN",
                timestamp
            )
            # <<< THÊM: Log xác nhận vị thế đã được xử lý (sau khi _update_balance chạy) >>>
            logging.info(f"{func_name} ({symbol}): Position state updated and removed due to {threshold_type} hit.")

        except asyncio.TimeoutError as te:
            logging.error(f"{func_name} ({symbol}): Timeout acquiring lock: {te}. State update failed.")
        except KeyError as ke:
            logging.error(f"{func_name} ({symbol}): Missing key in position info: {ke}. State update failed.")
        except Exception as e:
            logging.error(f"{func_name} ({symbol}): Error processing threshold hit: {e}", exc_info=True)
            logging.error(traceback.format_exc())
        finally:
            # === Giải phóng locks ===
            logging.debug(f"{func_name} ({symbol}): Releasing locks...")
            for lock in reversed(acquired_locks):
                try:
                    if lock.locked(): lock.release()
                except RuntimeError as rel_e: logging.error(f"{func_name}: Error releasing lock {lock}: {rel_e}")
            logging.debug(f"{func_name} ({symbol}): Locks released.")

            # === Cập nhật Monitor và Gửi Lệnh (NGOÀI LOCK) ===
            if position_info_local is not None: # Chỉ chạy nếu đã đọc được state ban đầu
                # Deactivate Monitor
                if self.realtime_monitor and self.mode == 'live':
                    try:
                        self.realtime_monitor.update_thresholds(symbol, active=False)
                        logging.info(f"{func_name}: Deactivated RealtimeMonitor for {symbol}.")
                    except Exception as mon_e: logging.error(f"{func_name} ({symbol}): Error deactivating monitor: {mon_e}")

                # --- TODO: Gửi lệnh đóng thật ra sàn ---
                if self.mode == 'live' and exit_price is not None: # Chỉ gửi lệnh nếu có giá đóng hợp lệ
                    logging.warning(f"--- ACTION NEEDED (LIVE): Close position {symbol} due to {threshold_type} hit at {exit_price:.4f} ---")
                    # await self._execute_live_order(symbol, 'market', ...) # Gửi lệnh với giá exit_price (hoặc market)
                    pass

    async def _handle_monitor_rr_crossed(self, signal_data: Dict[str, Any]):
        func_name = "_handle_monitor_rr_crossed"
        symbol = signal_data.get("symbol")
        crossed_level = signal_data.get("level")
        direction_cross = signal_data.get("direction")
        current_price = signal_data.get("current_price")

        if not symbol: logging.error(f"{func_name}: Received signal without symbol."); return
        if crossed_level is None: logging.error(f"{func_name} ({symbol}): Received signal without crossed level."); return

        position_info_copy: Optional[Dict[str, Any]] = None
        current_data_slice: Optional[Dict[str, pd.DataFrame]] = None
        acquired_locks: List[asyncio.Lock] = []
        agent_management_result: Optional[Dict[str, Any]] = None

        try:
            # === Bước 1: Lock để đọc state ban đầu và data slice ===
            lock_acquired_read = False
            try:
                logging.debug(f"{func_name} ({symbol}): Acquiring positions_lock and data_lock for read...")
                # Acquire positions_lock trước (theo thứ tự ưu tiên nếu cần cả hai)
                await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                acquired_locks.append(self.positions_lock)
                lock_acquired_read = True # Đánh dấu đã lấy được lock này

                # Kiểm tra vị thế tồn tại bên trong lock
                position_info_locked = self.open_positions.get(symbol)
                if not position_info_locked:
                    logging.debug(f"{func_name} ({symbol}): Position closed before RR cross could be processed.")
                    return # Thoát sớm

                # Sao chép thông tin vị thế ra ngoài
                position_info_copy = copy.deepcopy(position_info_locked)

                # Acquire data_lock để đọc data slice
                await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                acquired_locks.append(self.data_locks[symbol])
                logging.debug(f"{func_name} ({symbol}): Locks acquired for read.")

                # Lấy data slice bên trong data_lock
                current_data_slice = {tf: df.copy() for tf, df in self.data.get(symbol, {}).items() if isinstance(df, pd.DataFrame)}

            finally:
                # Giải phóng locks đọc theo thứ tự ngược lại
                for lock in reversed(acquired_locks):
                    if lock.locked(): lock.release()
                logging.debug(f"{func_name} ({symbol}): Read locks released.")
                # Reset acquired_locks sau khi giải phóng
                acquired_locks = []

            # === Bước 2: Chuẩn bị Input và Gọi Agent (NGOÀI LOCK) ===
            if position_info_copy is None or current_data_slice is None:
                logging.error(f"{func_name} ({symbol}): Failed to retrieve necessary position info or data slice.")
                return

            logging.info(f"--- MONITOR EVENT PROCESSING [{symbol}] --- RR Level: {crossed_level} {direction_cross}, Price: {current_price}")

            if not self.combined_agent or not self.decision_model: # Kiểm tra các thành phần cần thiết
                logging.warning(f"{func_name} ({symbol}): Agent or DecisionModel not available."); return

            # --- Chuẩn bị dữ liệu và tensor ---
            tf_primary = CONFIG.get("primary_tf", "15m"); required_tfs_agent = list(CONFIG.get('required_tfs_hybrid', ()))
            all_req_agent_tfs = set([tf_primary] + required_tfs_agent)
            # Kiểm tra dữ liệu slice đủ cho agent không
            if not all(tf in current_data_slice and isinstance(current_data_slice[tf], pd.DataFrame) and not current_data_slice[tf].empty for tf in all_req_agent_tfs):
                logging.warning(f"{func_name} ({symbol}): Missing required data for Agent in data slice."); return

            df_primary = current_data_slice[tf_primary]
            x_decision = None; x_hybrid_dict = {} # ... (logic tạo tensor từ current_data_slice) ...
            try:
                DEVICE = self.decision_model.device; decision_seq_len = CONFIG.get('DECISION_SEQ_LEN', 60); hybrid_seq_map = CONFIG.get('HYBRID_SEQ_LEN_MAP', {})
                if len(df_primary) >= decision_seq_len:
                    d_data = df_primary[['open','high','low','close','volume']].iloc[-decision_seq_len:].values
                    if not np.isnan(d_data).any(): x_decision = torch.tensor(d_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                v_hybrid = True
                for tf_h in required_tfs_agent:
                    df_h = current_data_slice.get(tf_h); seq_h = hybrid_seq_map.get(tf_h)
                    if df_h is not None and seq_h is not None and len(df_h) >= seq_h:
                        h_data = df_h[['open','high','low','close','volume']].iloc[-seq_h:].values
                        if not np.isnan(h_data).any(): x_hybrid_dict[tf_h] = torch.tensor(h_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        else: v_hybrid=False; break
                    else: v_hybrid=False; break
                if not v_hybrid: x_hybrid_dict = {}
            except Exception as te: logging.error(f"{func_name} ({symbol}): Error preparing tensors: {te}"); return
            if not (x_decision is not None and x_hybrid_dict): logging.warning(f"{func_name} ({symbol}): Missing tensors for Agent."); return

            # --- Gọi Agent ---
            try:
                # Giả sử có hàm riêng hoặc logic trong get_trade_signals để xử lý RR cross
                # Cần truyền position_info_copy (state trước khi quản lý)
                agent_management_result = self.combined_agent.get_trade_signals( # Hoặc hàm manage_open_position
                     symbol=symbol, x_decision=x_decision, x_hybrid_dict=x_hybrid_dict,
                     current_price=current_price,
                     current_data=current_data_slice, # Truyền slice data
                     open_position=position_info_copy, # Truyền state copy
                     # Có thể thêm tham số để chỉ rõ đây là trigger từ RR cross:
                     # trigger_event={'type': 'RR_CROSS', 'level': crossed_level, 'direction': direction_cross}
                )
            except Exception as agent_call_e:
                 logging.error(f"{func_name} ({symbol}): Agent call error: {agent_call_e}", exc_info=True); return

            # === Bước 3: Cập nhật State và Monitor (TRONG LOCK GHI) ===
            if agent_management_result:
                lock_acquired_write = False
                try:
                    logging.debug(f"{func_name} ({symbol}): Acquiring positions_lock for agent update...")
                    await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                    lock_acquired_write = True
                    acquired_locks.append(self.positions_lock) # Thêm vào list để giải phóng ở finally ngoài cùng
                    logging.debug(f"{func_name} ({symbol}): positions_lock acquired for agent update.")

                    # Kiểm tra lại vị thế tồn tại
                    current_pos_state = self.open_positions.get(symbol)
                    if not current_pos_state:
                        logging.debug(f"{func_name} ({symbol}): Position closed before agent results could be applied after RR cross.")
                        return # Thoát

                    # Xử lý kết quả và cập nhật state
                    new_trailing_tp = agent_management_result.get('trailing_tp_level', current_pos_state.get('trailing_tp_level', 0))
                    new_min_locked = agent_management_result.get('min_locked_tp_level', current_pos_state.get('min_locked_tp_level', 0))
                    # Giả sử agent không đổi SL ở đây, nếu có cần thêm logic cập nhật SL
                    should_update_monitor = False
                    if new_trailing_tp != current_pos_state.get('trailing_tp_level'):
                        current_pos_state['trailing_tp_level'] = new_trailing_tp; should_update_monitor = True
                        logging.info(f"[{self.mode.upper()}] Updating Trailing TP for {symbol} (RR Cross): -> {new_trailing_tp}")
                    if new_min_locked != current_pos_state.get('min_locked_tp_level'):
                        current_pos_state['min_locked_tp_level'] = new_min_locked; should_update_monitor = True
                        logging.info(f"[{self.mode.upper()}] Updating Min Locked TP for {symbol} (RR Cross): -> {new_min_locked}")
                    current_pos_state['last_agent_signals'] = copy.deepcopy(agent_management_result) # Lưu lại tín hiệu mới nhất

                    # --- Cập nhật Monitor (NGOÀI LOCK, tạo task) ---
                    if should_update_monitor and self.realtime_monitor and self.mode == 'live':
                        asyncio.create_task(self._update_monitor_managed_position(symbol))

                    # --- Xử lý đóng lệnh từ Agent (NGOÀI LOCK, tạo task) ---
                    # Cần lấy giá hiện tại từ signal_data vì current_price có thể đã cũ
                    exit_signal_agent = agent_management_result.get('exit_signal', 'hold')
                    if exit_signal_agent != 'hold' and not exit_signal_agent.startswith('close_at_t'):
                         logging.info(f"[{self.mode.upper()}] Agent requested position close for {symbol} after RR Cross: {exit_signal_agent}.")
                         # Sử dụng current_price từ signal_data làm giá đóng giả định
                         asyncio.create_task(self._handle_exit_signal(symbol, f"AGENT_CLOSE_AFTER_RR_{crossed_level}", current_price))

                except asyncio.TimeoutError:
                    logging.error(f"{func_name} ({symbol}): Timeout acquiring positions_lock for agent update. State not updated.")
                except Exception as write_e:
                    logging.error(f"{func_name} ({symbol}): Error updating position state from agent after RR cross: {write_e}")
            else:
                logging.warning(f"{func_name} ({symbol}): Agent did not return management results after RR cross.")

        except asyncio.TimeoutError as te_outer:
            logging.error(f"{func_name} ({symbol}): Timeout acquiring initial lock: {te_outer}")
        except Exception as e:
            logging.error(f"{func_name}: Error managing position {symbol} after RR cross: {e}", exc_info=True)
            logging.error(traceback.format_exc())
        finally:
             # Giải phóng tất cả các lock đã acquire trong acquired_locks
             for lock in reversed(acquired_locks):
                  if lock.locked(): lock.release()
             logging.debug(f"{func_name}: Management locks released for {symbol} after RR cross.")

    async def _handle_monitor_volatility(self, signal_data: Dict[str, Any]):
        func_name = "_handle_monitor_volatility"
        symbol = signal_data.get("symbol")
        current_price = signal_data.get("current_price")
        change_percent = signal_data.get("change_percent")

        if not symbol: logging.error(f"{func_name}: Received signal without symbol."); return
        if current_price is None: logging.warning(f"{func_name} ({symbol}): Received signal without current_price.");

        logging.warning(f"--- MONITOR EVENT PROCESSING [{symbol}] --- ABNORMAL VOLATILITY --- Change %: {change_percent:.2f}, Price: {current_price}")

        position_info_copy: Optional[Dict[str, Any]] = None
        current_data_slice: Optional[Dict[str, pd.DataFrame]] = None
        acquired_locks_read: List[asyncio.Lock] = []
        agent_risk_assessment: Optional[Dict[str, Any]] = None

        try:
            # === Bước 1: Lock để đọc state ban đầu và data slice ===
            lock_acquired_read_success = False
            try:
                logging.debug(f"{func_name} ({symbol}): Acquiring positions_lock and data_lock for read...")
                await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT); acquired_locks_read.append(self.positions_lock)
                position_info_locked = self.open_positions.get(symbol)
                if not position_info_locked: return # Đã đóng
                position_info_copy = copy.deepcopy(position_info_locked)
                await asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT); acquired_locks_read.append(self.data_locks[symbol])
                current_data_slice = {tf: df.copy() for tf, df in self.data.get(symbol, {}).items() if isinstance(df, pd.DataFrame)}
                lock_acquired_read_success = True
            finally:
                for lock in reversed(acquired_locks_read): # Giải phóng lock đọc
                    if lock.locked(): lock.release()
                logging.debug(f"{func_name} ({symbol}): Read locks released.")

            # === Bước 2: Chuẩn bị Input và Gọi Agent (NGOÀI LOCK) ===
            if not lock_acquired_read_success or position_info_copy is None or current_data_slice is None: return
            if not self.combined_agent or not self.decision_model: return

            # ... (Logic chuẩn bị tensor x_decision, x_hybrid_dict từ current_data_slice như cũ) ...
            tf_primary = CONFIG.get("primary_tf", "15m"); required_tfs_agent = list(CONFIG.get('required_tfs_hybrid', ()))
            all_req_agent_tfs = set([tf_primary] + required_tfs_agent)
            if not all(tf in current_data_slice and isinstance(current_data_slice[tf], pd.DataFrame) and not current_data_slice[tf].empty for tf in all_req_agent_tfs): return
            df_primary = current_data_slice[tf_primary]
            x_decision = None; x_hybrid_dict = {}
            try:
                DEVICE = self.decision_model.device; decision_seq_len = CONFIG.get('DECISION_SEQ_LEN', 60); hybrid_seq_map = CONFIG.get('HYBRID_SEQ_LEN_MAP', {})
                if len(df_primary) >= decision_seq_len:
                    d_data = df_primary[['open','high','low','close','volume']].iloc[-decision_seq_len:].values
                    if not np.isnan(d_data).any(): x_decision = torch.tensor(d_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                v_hybrid = True
                for tf_h in required_tfs_agent:
                    df_h = current_data_slice.get(tf_h); seq_h = hybrid_seq_map.get(tf_h)
                    if df_h is not None and seq_h is not None and len(df_h) >= seq_h:
                        h_data = df_h[['open','high','low','close','volume']].iloc[-seq_h:].values
                        if not np.isnan(h_data).any(): x_hybrid_dict[tf_h] = torch.tensor(h_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        else: v_hybrid=False; break
                    else: v_hybrid=False; break
                if not v_hybrid: x_hybrid_dict = {}
            except Exception as te: logging.error(f"{func_name} ({symbol}): Error preparing tensors: {te}"); return
            if not (x_decision is not None and x_hybrid_dict): logging.warning(f"{func_name} ({symbol}): Missing tensors for Agent."); return

            # --- Gọi Agent đánh giá rủi ro ---
            try:
                # Giả sử hàm này chỉ trả về new_sl_price (nếu có)
                agent_risk_assessment = self.combined_agent.assess_volatility_risk(
                     symbol=symbol, x_decision=x_decision, x_hybrid_dict=x_hybrid_dict,
                     current_price=current_price, open_position=position_info_copy
                )
            except Exception as agent_e: logging.error(f"{func_name} ({symbol}): Agent volatility assessment error: {agent_e}"); return

            # === Bước 3: Xử lý kết quả Agent - Chỉ cập nhật SL (TRONG LOCK GHI) ===
            if agent_risk_assessment:
                new_sl_price = agent_risk_assessment.get('new_sl_price')
                should_update_monitor = False
                acquired_locks_write: List[asyncio.Lock] = [] # List riêng cho lock ghi

                if new_sl_price is not None: # Chỉ xử lý nếu agent đề xuất SL mới
                    lock_acquired_write_success = False
                    try:
                        logging.debug(f"{func_name} ({symbol}): Acquiring positions_lock for SL update...")
                        await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
                        acquired_locks_write.append(self.positions_lock) # << Thêm vào list lock ghi
                        lock_acquired_write_success = True
                        logging.debug(f"{func_name} ({symbol}): positions_lock acquired for SL update.")

                        current_pos_state = self.open_positions.get(symbol)
                        if current_pos_state:
                            entry_p = current_pos_state.get('entry_price'); direction = current_pos_state.get('direction')
                            current_sl = current_pos_state.get('stop_loss_price', 0)
                            if entry_p and direction and abs(new_sl_price - current_sl) > 1e-9 and \
                               ((direction == 'LONG' and new_sl_price < entry_p) or \
                                (direction == 'SHORT' and new_sl_price > entry_p)):
                                logging.warning(f"{func_name} ({symbol}): Agent suggested tightening SL to {new_sl_price:.4f} due to volatility.")
                                current_pos_state['stop_loss_price'] = new_sl_price
                                should_update_monitor = True # << Đánh dấu cần cập nhật monitor
                            elif abs(new_sl_price - current_sl) > 1e-9:
                                 logging.warning(f"{func_name} ({symbol}): Agent suggested SL {new_sl_price:.4f} ignored (invalid or same).")
                        else:
                             logging.debug(f"{func_name} ({symbol}): Position closed before SL update could be applied.")

                    except asyncio.TimeoutError:
                         logging.error(f"{func_name} ({symbol}): Timeout acquiring positions_lock for SL update. SL not updated.")
                    except Exception as write_e:
                         logging.error(f"{func_name} ({symbol}): Error updating SL: {write_e}")
                    finally:
                         # Chỉ giải phóng lock ghi
                         for lock in reversed(acquired_locks_write):
                              if lock.locked(): lock.release()
                         logging.debug(f"{func_name} ({symbol}): SL update lock released.")

                    # --- Cập nhật Monitor (NGOÀI LOCK, tạo task) ---
                    if should_update_monitor and self.realtime_monitor and self.mode == 'live':
                        asyncio.create_task(self._update_monitor_managed_position(symbol)) # Hàm này lấy state mới nhất

        except asyncio.TimeoutError as te_outer:
            logging.error(f"{func_name} ({symbol}): Timeout acquiring initial locks: {te_outer}")
        except Exception as e:
            logging.error(f"{func_name}: Error managing position {symbol} after volatility event: {e}", exc_info=True)
            logging.error(traceback.format_exc())

    # --- Helper để lấy giá TP Level (cần thiết cho callback THRESHOLD_HIT) ---
    def _get_price_for_tp_level(self, symbol: str, level: float) -> Optional[float]:
        """Lấy giá TP tương ứng với level từ state của Monitor."""
        if self.realtime_monitor:
             with self.realtime_monitor.position_states_lock:
                 state = self.realtime_monitor.position_states.get(symbol)
                 if state and state.get('active'):
                     tp_key = f'RR_{level:.1f}'.replace('.0', '')
                     # Kiểm tra xem key có tồn tại và giá trị có hợp lệ không
                     price = state.get('thresholds', {}).get(tp_key)
                     if isinstance(price, (int, float)) and price > 0:
                          return price
                     else:
                          logging.warning(f"Invalid or missing price for TP key '{tp_key}' in Monitor state for {symbol}.")
        position = self.open_positions.get(symbol)
        if position:
             entry = position.get('entry_price')
             sl = position.get('stop_loss_price')
             p_type = position.get('direction', '').lower()
             if entry and sl and p_type:
                  recalculated_price = RealtimeMonitor._calculate_rr_price(entry, sl, level, p_type)
                  if recalculated_price is not None:
                       logging.warning(f"Recalculated price for TP level {level} for {symbol}: {recalculated_price:.4f}")
                       return recalculated_price
        logging.error(f"Could not retrieve or recalculate price for TP level {level} for {symbol}")
        return None
    
    async def _run_live_loop(self):
        """Vòng lặp chính cho chế độ LIVE (Sử dụng helper)."""
        logging.info("Starting main LIVE trading loop...")
        # <<< Bắt đầu Monitor ở đây >>>
        if self.realtime_monitor:
            try:
                await self.realtime_monitor.start() # Giả sử start là async
                logging.info("RealtimeMonitor started.")
            except Exception as monitor_start_e:
                logging.error(f"Failed to start RealtimeMonitor: {monitor_start_e}", exc_info=True)
                # Có thể quyết định dừng bot nếu monitor không start được
                # return
        else:
            logging.warning("RealtimeMonitor not available for live mode.")

        while True:
            # === Kiểm tra cờ cleanup (trong lock ngắn) ===
            async with self.misc_state_lock:
                if self._cleanup_complete:
                    logging.info("Cleanup flag set, exiting live loop.")
                    break

            try:
                start_time = time.monotonic()
                active_symbols = CONFIG.get("symbols", [])
                if not active_symbols:
                    logging.debug("No active symbols configured, sleeping.")
                    await asyncio.sleep(30); continue

                # --- 1. Cập nhật Dữ liệu (Song song - hàm helper có lock nội bộ) ---
                # update_tasks sẽ gọi _process_symbol_data_update cho mỗi symbol
                update_tasks = [self._process_symbol_data_update(s) for s in active_symbols]
                await asyncio.gather(*update_tasks, return_exceptions=True) # Log lỗi nếu cần thiết

                # --- 2. Kiểm tra Global CB (Có lock nội bộ) ---
                # Chỉ cần kiểm tra một lần sau khi cập nhật dữ liệu
                cb_active_global = False
                if active_symbols:
                     first_sym = active_symbols[0]
                     async with self.data_locks[first_sym]: # Lock đọc data
                          df_cb_check = self.data.get(first_sym, {}).get(CONFIG.get("primary_tf", "15m"))
                     if df_cb_check is not None and not df_cb_check.empty:
                          cb_active_global = await self.check_circuit_breaker(df_cb_check)

                if cb_active_global:
                    logging.warning("Global CB ACTIVE. Loop iteration skipped.")
                    await asyncio.sleep(CONFIG.get("main_loop_sleep", 10)); continue

                # --- 3. Xử lý Logic cho Từng Symbol (Song song) ---
                # Tạo một task gọi hàm helper _process_single_symbol_live_logic cho mỗi symbol
                processing_tasks = [
                    self._process_single_symbol_live_logic(symbol)
                    for symbol in active_symbols
                ]

                if processing_tasks:
                    logging.debug(f"Dispatching processing tasks for {len(processing_tasks)} symbols...")
                    results = await asyncio.gather(*processing_tasks, return_exceptions=True)
                    # Log lỗi từ các task xử lý symbol nếu cần
                    for symbol, result in zip(active_symbols, results):
                         if isinstance(result, Exception):
                              logging.error(f"Error processing symbol {symbol} in live loop: {result}", exc_info=result)
                else:
                    logging.debug("No symbols eligible for processing in this loop.")


                # --- 4. Sleep ---
                processing_time = time.monotonic() - start_time
                sleep_duration = max(0.1, CONFIG.get("main_loop_sleep", 10) - processing_time)
                # logging.debug(f"Loop duration: {processing_time:.2f}s, Sleeping for {sleep_duration:.2f}s")
                await asyncio.sleep(sleep_duration)

            except asyncio.CancelledError:
                logging.info("Bot run loop cancelled.")
                break
            except Exception as loop_e:
                logging.error(f"Unexpected error in main trading loop: {loop_e}", exc_info=True)
                logging.error(traceback.format_exc())
                # Quyết định có nên dừng hẳn không?
                logging.critical("CRITICAL ERROR in main loop. Stopping.")
                break # Dừng vòng lặp nếu có lỗi nghiêm trọng

    async def _process_single_symbol_live_logic(self, symbol: str):
        func_name = "_process_single_symbol_live_logic"
        tf_primary = CONFIG.get("primary_tf", "15m")
        all_necessary_tfs: Set[str] = set([tf_primary] + list(CONFIG.get('required_tfs_hybrid', ())) + ["1h", "4h"])
        min_lengths: Dict[str, int] = { tf: CONFIG.get('HYBRID_SEQ_LEN_MAP', {}).get(tf, CONFIG.get('DECISION_SEQ_LEN', 60) if tf == tf_primary else 2) for tf in all_necessary_tfs }
        is_new_primary_candle = False
        current_data_slice: Optional[Dict[str, pd.DataFrame]] = None # Slice dữ liệu để dùng chung

        try:
            # === 1. Kiểm tra data, CB cục bộ, và nến mới (trong data_lock) ===
            async with asyncio.wait_for(self.data_locks[symbol].acquire(), timeout=LOCK_ACQUIRE_TIMEOUT):
                # Kiểm tra data tồn tại và đủ dài
                if symbol not in self.data or not all(tf in self.data[symbol] and isinstance(self.data[symbol][tf], pd.DataFrame) and len(self.data[symbol][tf]) >= min_lengths.get(tf, 2) for tf in all_necessary_tfs):
                    # logging.debug(f"{func_name} ({symbol}): Insufficient/invalid data.")
                    return # Không đủ dữ liệu, bỏ qua symbol này

                # Kiểm tra CB cục bộ (hàm check_circuit_breaker có lock nội bộ)
                df_check_cb = self.data[symbol].get(tf_primary)
                if df_check_cb is not None and not df_check_cb.empty:
                    if await self.check_circuit_breaker(df_check_cb):
                        # logging.debug(f"{func_name} ({symbol}): Circuit breaker active.")
                        return # CB Active cho symbol này, bỏ qua

                # Kiểm tra nến mới (cần lock)
                is_new_primary_candle = self._is_new_candle_detected(symbol, tf_primary)

                # <<< Lấy slice dữ liệu hiện tại để truyền đi (vẫn trong lock) >>>
                current_data_slice = {tf: df.copy() for tf, df in self.data[symbol].items() if isinstance(df, pd.DataFrame)}

            # === 2. Thực hiện tác vụ định kỳ NẾU có nến mới (ngoài data_lock) ===
            if is_new_primary_candle and current_data_slice:
                logging.debug(f"{func_name}: Processing periodic tasks for new {tf_primary} candle in {symbol}")
                periodic_tasks = [
                    self._calculate_and_add_hybrid_regimes(symbol), # Có lock nội bộ
                    self._update_latest_embeddings([symbol]) # Nên có lock nội bộ nếu cần
                ]
                # <<< Quản lý vị thế mở (gọi hàm có lock nội bộ) >>>
                async with self.positions_lock: is_pos_open = symbol in self.open_positions # Chỉ đọc
                if is_pos_open:
                    # Truyền slice dữ liệu vào hàm quản lý
                    periodic_tasks.append(self._manage_open_positions({symbol: current_data_slice}))

                if periodic_tasks:
                    results = await asyncio.gather(*periodic_tasks, return_exceptions=True)
                    for i, res in enumerate(results): # Log lỗi nếu có
                         if isinstance(res, Exception): logging.error(f"Error in periodic task {i} for {symbol}: {res}")

            # === 3. Thực hiện logic VÀO LỆNH MỚI (ngoài data_lock) ===
            if current_data_slice: # Chỉ chạy nếu có dữ liệu slice hợp lệ
                async with self.positions_lock: is_open = symbol in self.open_positions # Chỉ đọc
                if not is_open:
                    # <<< Gọi hàm vào lệnh (có lock nội bộ), truyền slice >>>
                    await self._process_symbol_trading_logic(symbol, current_data_slice) # Hàm này xử lý logic và lock bên trong

        except asyncio.TimeoutError:
            logging.error(f"{func_name} ({symbol}): Timeout acquiring data_lock. Skipping processing.")
        except Exception as e:
            logging.error(f"{func_name}: Error processing symbol {symbol}: {e}", exc_info=True)

    async def _run_backtest_loop(self):
        """Vòng lặp chính cho Backtest (chạy theo nến, dùng chung hàm logic với live)."""
        logging.info("Starting BACKTEST Loop (Processing candle by candle)...")
        tf_primary = CONFIG.get("primary_tf", "15m") # <<< Nhất quán key

        # --- 1. Kiểm tra dữ liệu và Xác định Index ---
        if not self.data: logging.error("Backtest loop requires pre-loaded data."); return
        first_symbol = next((s for s in self.data if tf_primary in self.data[s] and not self.data[s][tf_primary].empty), None)
        if not first_symbol: logging.error(f"Backtest loop requires '{tf_primary}' data."); return
        backtest_index = self.data[first_symbol][tf_primary].index
        try: # Xác định start_index an toàn
             df_warmup_check = self.data[first_symbol][tf_primary]; min_warmup_len = CONFIG.get("hmm_warmup_min", 50)
             if len(df_warmup_check) < min_warmup_len: raise ValueError(f"Not enough data ({len(df_warmup_check)}) for warmup ({min_warmup_len})")
             start_index = self.get_adaptive_warmup_period(df_warmup_check); assert start_index < len(backtest_index), f"Start index ({start_index}) out of bounds"
             logging.info(f"Backtest starting from index {start_index} (Timestamp: {backtest_index[start_index]}).")
        except Exception as e: logging.error(f"Error determining start index: {e}", exc_info=True); return

        # --- 2. Vòng lặp qua từng nến trong backtest ---
        for i in range(start_index, len(backtest_index)):
            current_timestamp = backtest_index[i]
            logging.debug(f"--- Processing Backtest Candle: {current_timestamp} (Index: {i}) ---")
            async with self.misc_state_lock:
                if self._cleanup_complete:
                    logging.info("Cleanup flag set during backtest, exiting loop.")
                    break

            # --- 2a. Lấy dữ liệu slice ---
            current_data_slice = {}
            available_symbols_in_slice = []
            all_necessary_tfs_bt = set([tf_primary] + list(CONFIG.get('required_tfs_hybrid', ())) + ["1h", "4h"])
            min_lengths_bt = { tf: CONFIG.get('HYBRID_SEQ_LEN_MAP', {}).get(tf, CONFIG.get('DECISION_SEQ_LEN', 60) if tf == tf_primary else 2) for tf in all_necessary_tfs_bt }
            max_lookback_needed = max(200, CONFIG.get("DECISION_SEQ_LEN", 60), *CONFIG.get('HYBRID_SEQ_LEN_MAP', {}).values())

            for symbol in self.data.keys():
                symbol_data_slice = {}; symbol_data_valid = True
                if symbol not in self.data: continue
                # <<< KIỂM TRA DỮ LIỆU ĐẦY ĐỦ TRƯỚC KHI SLICE >>>
                if not all(tf in self.data[symbol] and isinstance(self.data[symbol][tf], pd.DataFrame) for tf in all_necessary_tfs_bt): continue

                for tf in all_necessary_tfs_bt:
                    df_full = self.data[symbol][tf]; min_len_req = min_lengths_bt.get(tf, 2)
                    try:
                        loc = df_full.index.get_loc(current_timestamp, method='ffill')
                        start_loc = max(0, loc - max_lookback_needed + 1)
                        df_slice = df_full.iloc[start_loc : loc + 1]
                        # <<< Kiểm tra độ dài slice tối thiểu CHÍNH XÁC >>>
                        if not df_slice.empty and len(df_slice) >= min_len_req:
                            symbol_data_slice[tf] = df_slice
                        else: symbol_data_valid = False; break
                    except Exception: symbol_data_valid = False; break
                if symbol_data_valid and symbol_data_slice:
                    current_data_slice[symbol] = symbol_data_slice
                    available_symbols_in_slice.append(symbol)

            if not available_symbols_in_slice: continue

            # --- 2b. Thực hiện các tác vụ cho NẾN HIỆN TẠI ---

            # --- Tính toán nặng (Regime/Embedding) - Gọi hàm gốc (chúng sẽ dùng self.data mới nhất) ---
            heavy_tasks = []
            for symbol in available_symbols_in_slice: # Chỉ tính cho symbol có slice hợp lệ
                 heavy_tasks.append(self._calculate_and_add_hybrid_regimes(symbol))
                 heavy_tasks.append(self._update_latest_embeddings([symbol]))
            if heavy_tasks: await asyncio.gather(*heavy_tasks, return_exceptions=True) # Chạy song song

            # --- Kiểm tra Circuit Breaker (dùng dữ liệu slice) ---
            global_cb_active = False
            for symbol in available_symbols_in_slice:
                 df_primary_slice = current_data_slice[symbol].get(tf_primary)
                 if df_primary_slice is not None and not df_primary_slice.empty:
                      if await self.check_circuit_breaker(df_primary_slice): # <<< THÊM await
                        global_cb_active = True; break
            if global_cb_active: logging.warning(f"Backtest: CB ACTIVE at {current_timestamp}. Skipping."); continue

            # --- Xử lý Logic Vào Lệnh Mới ---
            entry_tasks = []
            for symbol in available_symbols_in_slice:
                if symbol not in self.open_positions:
                    # <<< Gọi hàm gốc, truyền slice >>>
                    entry_tasks.append(self._process_symbol_trading_logic(symbol, current_data_slice[symbol]))
            if entry_tasks: await asyncio.gather(*entry_tasks, return_exceptions=True)

            # --- Quản lý Vị thế Đang Mở ---
            # <<< Gọi hàm gốc, truyền slice >>>
            await self._manage_open_positions(current_data_slice)

        logging.info("Backtest Loop finished.")

    async def _rollback_to_safe_state(self):
        func_name = "_rollback_to_safe_state"
        acquired_locks = [] # Theo dõi các lock đã acquire
        rollback_state_local = None # Lưu trữ state đọc được để dùng sau

        try:
            # === Bước 1: Acquire cb_lock để đọc và xóa rollback_state ===
            logging.debug(f"{func_name}: Acquiring cb_lock...")
            await asyncio.wait_for(self.cb_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            acquired_locks.append(self.cb_lock)
            logging.debug(f"{func_name}: cb_lock acquired.")

            # Đọc và xóa state rollback bên trong cb_lock
            if self.rollback_state:
                logging.warning(f"{func_name}: Found rollback state. Proceeding with rollback.")
                # <<< Sao chép state ra biến cục bộ TRƯỚC KHI xóa >>>
                rollback_state_local = copy.deepcopy(self.rollback_state) # Cần deepcopy vì có dict lồng nhau
                self.rollback_state = None # Xóa ngay để tránh rollback lại lần nữa nếu lỗi ở bước sau
            else:
                logging.warning(f"{func_name}: Rollback requested but no rollback state available.")
                # Không cần làm gì thêm, giải phóng cb_lock và thoát
                return

            # === Bước 2: Acquire các lock còn lại theo đúng thứ tự ===
            logging.debug(f"{func_name}: Acquiring remaining locks...")
            acquired_balance = await asyncio.wait_for(self.balance_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_balance: raise asyncio.TimeoutError("Timeout acquiring balance_lock")
            acquired_locks.append(self.balance_lock)

            acquired_positions = await asyncio.wait_for(self.positions_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_positions: raise asyncio.TimeoutError("Timeout acquiring positions_lock")
            acquired_locks.append(self.positions_lock)

            acquired_history = await asyncio.wait_for(self.trade_history_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
            if not acquired_history: raise asyncio.TimeoutError("Timeout acquiring trade_history_lock")
            acquired_locks.append(self.trade_history_lock)
            logging.debug(f"{func_name}: All necessary locks acquired.")

            # === Bước 3: Thực hiện khôi phục state (bên trong tất cả lock) ===
            # Sử dụng dữ liệu từ rollback_state_local đã sao chép
            self.balance = rollback_state_local["balance"]
            self.equity_peak = rollback_state_local["equity_peak"]
            # Khôi phục history cẩn thận
            if len(self.trade_history) > rollback_state_local["trade_history_len"]:
                 self.trade_history = self.trade_history[:rollback_state_local["trade_history_len"]]
            self.exposure_per_symbol = rollback_state_local["exposure_per_symbol"].copy() # Đã copy khi lưu
            self.open_positions = rollback_state_local["open_positions"] # Đã deepcopy khi lưu

            # <<< Cập nhật state CB cuối cùng (vẫn trong cb_lock) >>>
            self.circuit_breaker_triggered = True # Giữ trạng thái triggered
            self.last_trade_time = None # Reset thời gian để cooldown lại

            logging.info(f"{func_name}: State successfully rolled back. Balance={self.balance:.2f}. CB remains triggered.")

        except asyncio.TimeoutError as te:
            logging.error(f"{func_name}: Timeout acquiring lock during rollback: {te}. Rollback may be incomplete!")
        except KeyError as ke:
             logging.error(f"{func_name}: Missing key in rollback state data: {ke}. Rollback failed.", exc_info=True)
        except Exception as roll_e:
             logging.error(f"{func_name}: Unexpected error during state rollback: {roll_e}", exc_info=True)
             logging.error(traceback.format_exc())
        finally:
            # === Đảm bảo giải phóng TẤT CẢ các lock đã acquire theo thứ tự NGƯỢC LẠI ===
            logging.debug(f"{func_name}: Releasing rollback locks...")
            for lock in reversed(acquired_locks):
                try:
                    if lock.locked():
                        lock.release()
                except RuntimeError as rel_e:
                    logging.error(f"{func_name}: Error releasing lock {lock}: {rel_e}")
            logging.debug(f"{func_name}: Rollback locks released.")


    def plot_performance(self):
        if not self.trade_history:
            logging.warning("No trade history to plot performance.")
            return

        try:
             equity_curve = [CONFIG.get("initial_balance", 10000)]
             timestamps = [self.trade_history[0]["timestamp"]] # Lấy timestamp đầu tiên

             for trade in self.trade_history:
                  equity_curve.append(trade["balance_after"]) # Dùng balance_after từ log trade
                  timestamps.append(trade["timestamp"])

             # Chuyển timestamps sang định dạng phù hợp cho plot
             plot_timestamps = pd.to_datetime(timestamps)

             plt.figure(figsize=(14, 7))
             plt.plot(plot_timestamps, equity_curve, label="Equity Curve", color='blue')
             plt.xlabel("Time")
             plt.ylabel("Equity (USDT)")
             plt.title("Trading Bot Performance")
             plt.legend()
             plt.grid(True)
             plt.xticks(rotation=45)
             plt.tight_layout() # Điều chỉnh layout
             plt.savefig("equity_curve.png")
             plt.close() # Đóng plot để giải phóng bộ nhớ
             logging.info("Performance plot saved as equity_curve.png")
        except Exception as plot_e:
             logging.error(f"Error plotting performance: {plot_e}", exc_info=True)


# --- Main function and execution block ---
async def main():
    bot = None # Khởi tạo là None
    try:
        bot = EnhancedTradingBot()
        # await bot.initialize() # Initialize được gọi trong run
        logging.info("Starting bot run loop...")
        await bot.run() # run đã bao gồm initialize và vòng lặp chính
        logging.info("Bot run loop finished.")
        if bot: bot.plot_performance() # Vẽ đồ thị nếu bot tồn tại
    except ValueError as config_err: # Bắt lỗi cấu hình cụ thể
         logging.critical(f"Configuration Error: {config_err}")
    except ccxt.AuthenticationError:
         logging.critical("CCXT Authentication Error. Please check API keys.")
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}", exc_info=True)
        logging.critical(traceback.format_exc())
    finally:
        if bot and not bot._cleanup_complete:
            logging.info("Ensuring cleanup is called from main...")
            await bot._cleanup()
        logging.info("Main execution finished.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received in __main__. Exiting.")