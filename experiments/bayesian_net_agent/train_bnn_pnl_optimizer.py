import torch
import torch.nn as nn
import torch.nn.functional as F # Cần F cho dropout
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from typing import Optional, Tuple, List, Dict, Any
import time
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats
from scipy.signal import find_peaks

# Giả sử TA-Lib đã được cài đặt
try:
    import talib
except ImportError:
    logging.warning("TA-Lib not found. Some proxy features might not be calculated.")
    talib = None

# Import các thành phần cần thiết từ api.py
try:
    from api import CONFIG 
    # Định nghĩa lại BNN nâng cao ở đây để dễ tùy chỉnh cho Optuna
    class EnhancedBayesianNeuralNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int = 1, dropout_rate: float = 0.0):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_hidden_layers = max(1, num_hidden_layers) # Đảm bảo ít nhất 1 lớp ẩn
            self.dropout_rate = dropout_rate

            layers = []
            in_features = input_dim
            # Thêm các lớp ẩn
            for _ in range(self.num_hidden_layers):
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.ReLU())
                if self.dropout_rate > 0:
                     layers.append(nn.Dropout(self.dropout_rate))
                in_features = hidden_dim

            # Lớp output
            layers.append(nn.Linear(in_features, output_dim))

            self.net = nn.Sequential(*layers)

        def forward(self, x):
             # Dropout chỉ áp dụng trong quá trình huấn luyện SVI thông qua random module
             # Forward chuẩn không cần áp dụng dropout trực tiếp ở đây
             return self.net(x)

        # --- Model và Guide cho Pyro (Quan trọng: cần cập nhật để xử lý nhiều lớp) ---
        def model(self, x, y=None):
             # Tạo prior distributions cho TẤT CẢ các trọng số và bias
             priors = {}
             last_shape = (x.shape[-1],) # Shape của input

             module_idx = 0
             for module in self.net:
                  if isinstance(module, nn.Linear):
                       weight_shape = module.weight.shape
                       bias_shape = module.bias.shape
                       # Sử dụng Normal priors đơn giản
                       priors[f'net.{module_idx}.weight'] = dist.Normal(torch.zeros(weight_shape), torch.ones(weight_shape)).to_event(2)
                       priors[f'net.{module_idx}.bias'] = dist.Normal(torch.zeros(bias_shape), torch.ones(bias_shape)).to_event(1)
                  module_idx += 1

             # Tạo module ngẫu nhiên với priors
             lifted_module = pyro.random_module("module", self, priors)
             lifted_reg_model = lifted_module()

             # Chạy dữ liệu qua model với trọng số ngẫu nhiên từ prior
             prediction_mean = lifted_reg_model(x) # Lấy output từ forward

             # Assume a likelihood (ví dụ Normal với std dev cố định hoặc học được)
             # Giữ std dev cố định cho đơn giản
             sigma = pyro.param("likelihood_sigma", torch.tensor(0.1), constraint=dist.constraints.positive) # Có thể học sigma này

             with pyro.plate("data", x.shape[0]):
                  # Lấy mẫu từ likelihood
                  pyro.sample("obs", dist.Normal(prediction_mean, sigma).to_event(1), obs=y)

             return prediction_mean # Trả về mean để dùng trong Predictive

        def guide(self, x, y=None):
             # Guide cần định nghĩa các tham số cho phân phối q (xấp xỉ hậu nghiệm)
             # Ví dụ: Dùng Normal mean-field (mỗi tham số có mean và std riêng)
             guide_params = {}
             module_idx = 0
             for module in self.net:
                  if isinstance(module, nn.Linear):
                       weight_shape = module.weight.shape
                       bias_shape = module.bias.shape

                       # Mean và std cho weights
                       w_mu = pyro.param(f'guide_W_mu_{module_idx}', torch.randn(weight_shape) * 0.1)
                       w_sigma = pyro.param(f'guide_W_sigma_{module_idx}', torch.randn(weight_shape) * 0.1) # Dùng softplus để đảm bảo dương
                       guide_params[f'net.{module_idx}.weight'] = dist.Normal(w_mu, F.softplus(w_sigma)).to_event(2)

                       # Mean và std cho bias
                       b_mu = pyro.param(f'guide_b_mu_{module_idx}', torch.randn(bias_shape) * 0.1)
                       b_sigma = pyro.param(f'guide_b_sigma_{module_idx}', torch.randn(bias_shape) * 0.1)
                       guide_params[f'net.{module_idx}.bias'] = dist.Normal(b_mu, F.softplus(b_sigma)).to_event(1)
                  module_idx += 1

             # (Optional) Có thể thêm guide cho likelihood_sigma nếu học nó
             # sigma_mu = pyro.param(...)
             # sigma_sigma = pyro.param(...)
             # pyro.sample("likelihood_sigma", dist.LogNormal(sigma_mu, F.softplus(sigma_sigma)))

             # Tạo module ngẫu nhiên với guide distributions
             lifted_module = pyro.random_module("module", self, guide_params)
             return lifted_module()

except ImportError as e:
    logging.error(f"Could not import from api.py: {e}.")
    logging.error("Make sure api.py is in the same directory or accessible via PYTHONPATH.")
    exit()
except AttributeError as ae:
     logging.error(f"Could not import 'detect_swing_points' from api.py: {ae}. Duration proxy calculation will fail.")
     def detect_swing_points(df: pd.DataFrame, lookback: int = 5, prominence_factor: float = 0.001) -> pd.DataFrame:
         df['swing_high'] = False
         df['swing_low'] = False
         logging.warning("Using dummy detect_swing_points function.")
         return df

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR
MODEL_SAVE_DIR = SCRIPT_DIR / "trained_models_bnn_v3" # Thư mục mới
MODEL_SAVE_DIR.mkdir(exist_ok=True)
BEST_GUIDE_PARAMS_PATH = MODEL_SAVE_DIR / "bnn_pnl_guide_best.pth"
FINAL_GUIDE_PARAMS_PATH = MODEL_SAVE_DIR / "bnn_pnl_guide_final_optuna.pth"
SCALER_SAVE_PATH = MODEL_SAVE_DIR / "bnn_pnl_scaler.pkl"
OPTUNA_DB_PATH = f"sqlite:///{MODEL_SAVE_DIR / 'bnn_optuna_study_v3.db'}" # DB mới

# Data Parameters
SYMBOLS_TO_TRAIN = CONFIG.get("symbols", ["BTC/USDT", "ETH/USDT"])
TIMEFRANE = "15m"
# >>> THÊM FEATURES KỸ THUẬT <<<
INPUT_FEATURES = [
    "entry_price_norm", "atr_norm", "volatility",
    "correlation_proxy", "liquidity_proxy", "duration_proxy",
    # Chỉ báo mới
    "RSI",
    "MACD_hist", # Dùng MACD Histogram (MACD - Signal)
    "BB_width",  # Độ rộng Bollinger Bands chuẩn hóa
    "ADX"
]
INPUT_DIM = len(INPUT_FEATURES)
TARGET_FEATURE = "future_return_pct"
FUTURE_RETURN_PERIOD = 10

# Training Hyperparameters (Defaults/Ranges for Optuna)
DEFAULT_EPOCHS = 100
DEFAULT_SVI_STEPS_PER_EPOCH = 300
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.002
VALIDATION_SPLIT_RATIO = 0.20
EARLY_STOPPING_PATIENCE = 10
NUM_PREDICTION_SAMPLES = 100
OPTUNA_N_TRIALS = 50 # >>> TĂNG SỐ TRIALS <<<

# --- Helper Functions ---
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

def calculate_time_since_last_swing(df: pd.DataFrame) -> pd.Series:
    # ... (giữ nguyên)
    if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
        return pd.Series(0, index=df.index)
    is_swing = df['swing_high'] | df['swing_low']
    if not is_swing.any(): return pd.Series(np.arange(len(df)), index=df.index)
    counter = (~is_swing).cumsum()
    time_since = counter - counter[is_swing].ffill().fillna(0)
    return time_since.fillna(0)

# --- Data Loading and Preprocessing ---
def load_and_prepare_bnn_data(symbol: str) -> Optional[pd.DataFrame]:
    symbol_safe = symbol.replace('/', '_').replace(':', '')
    file_path = DATA_DIR / f"{symbol_safe}_data.pkl"
    if not file_path.exists(): return None
    try:
        try: data_dict = joblib.load(file_path)
        except:
            import pickle
            with open(file_path, 'rb') as f: data_dict = pickle.load(f)
        if TIMEFRANE not in data_dict or not isinstance(data_dict[TIMEFRANE], pd.DataFrame): return None
        df = data_dict[TIMEFRANE].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
             try: df.index = pd.to_datetime(df.index)
             except:
                  if 'timestamp' in df.columns:
                       df['timestamp'] = pd.to_datetime(df['timestamp'])
                       df = df.set_index('timestamp')
                  else: return None
        df = df.sort_index()

        # >>> Tính toán thêm chỉ báo kỹ thuật nếu chưa có <<<
        base_cols = ['open','high', 'low', 'close', 'ATR', 'volatility', 'volume', 'RSI']
        # Kiểm tra và tính toán nếu thiếu
        if 'RSI' not in df.columns and talib:
             df['RSI'] = talib.RSI(df['close'])
        if ('MACD' not in df.columns or 'MACD_signal' not in df.columns) and talib:
             macd, macdsignal, _ = talib.MACD(df['close'])
             df['MACD'] = macd
             df['MACD_signal'] = macdsignal
        if 'MACD_hist' not in df.columns and 'MACD' in df.columns and 'MACD_signal' in df.columns:
             df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        if ('BB_upper' not in df.columns or 'BB_lower' not in df.columns or 'BB_middle' not in df.columns) and talib:
             upper, middle, lower = talib.BBANDS(df['close'])
             df['BB_upper'] = upper
             df['BB_middle'] = middle
             df['BB_lower'] = lower
        if 'BB_width' not in df.columns and 'BB_upper' in df.columns and 'BB_lower' in df.columns and 'BB_middle' in df.columns:
             df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'].replace(0, np.nan)
        if 'ADX' not in df.columns and talib:
             df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])

        # Kiểm tra lại các cột cần thiết sau khi tính toán
        required_cols = base_cols + ['MACD_hist', 'BB_width', 'ADX'] # Các cột cần cho features
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns after calculation for {symbol}: {missing_cols}")
            return None

        # Xử lý NaN
        df = df.ffill().bfill()
        if df.isnull().sum().sum() > 0: df.dropna(inplace=True)
        if len(df) < 50: return None

        # --- Tính toán Features Proxy (như cũ + các chỉ báo mới) ---
        rolling_mean_close = df['close'].rolling(20, min_periods=5).mean().bfill().ffill()
        df['entry_price_norm'] = (df['close'] - rolling_mean_close) / rolling_mean_close.replace(0, np.nan)
        df['atr_norm'] = df['ATR'] / df['close'].replace(0, np.nan)
        df['return'] = df['close'].pct_change()
        df['correlation_proxy'] = df['return'].rolling(20, min_periods=5).corr(df['return'].shift(1))
        rolling_mean_vol = df['volume'].rolling(20, min_periods=5).mean()
        rolling_std_vol = df['volume'].rolling(20, min_periods=5).std().replace(0, np.nan)
        df['liquidity_proxy'] = (df['volume'] - rolling_mean_vol) / rolling_std_vol
        try:
            df = detect_swing_points(df, lookback=7, prominence_factor=0.0015)
            df['duration_proxy'] = calculate_time_since_last_swing(df)
        except Exception as swing_err:
            logging.error(f"Error calculating swing points or duration for {symbol}: {swing_err}")
            df['duration_proxy'] = 0.0

        # Target Proxy
        df[TARGET_FEATURE] = df['close'].pct_change(FUTURE_RETURN_PERIOD).shift(-FUTURE_RETURN_PERIOD)

        # Final Processing
        # >>> Đảm bảo INPUT_FEATURES khớp với các cột đã tính <<<
        available_features = [f for f in INPUT_FEATURES if f in df.columns]
        if len(available_features) != len(INPUT_FEATURES):
             missing_input_features = set(INPUT_FEATURES) - set(available_features)
             logging.error(f"FATAL: Missing final input features for {symbol}: {missing_input_features}")
             return None

        final_cols = available_features + [TARGET_FEATURE]
        df_final = df[final_cols].copy()
        initial_rows = len(df_final)
        df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_final) < 100: return None

        return df_final

    except Exception as e:
        logging.error(f"Error loading or preparing BNN data for {symbol}: {e}", exc_info=True)
        return None

# --- BNN Prediction Function --- (Giữ nguyên)
def predict_bnn(model: nn.Module, guide: Any, X_tensor: torch.Tensor, num_samples: int = NUM_PREDICTION_SAMPLES) -> Tuple[torch.Tensor, torch.Tensor]:
    # Cast model sang kiểu đúng nếu cần (dùng EnhancedBayesianNeuralNetwork)
    if not hasattr(model, 'model'): # Kiểm tra nếu là instance của class mới
         logging.error("Predict_bnn received an unexpected model type.")
         return torch.zeros_like(X_tensor[:, :1]), torch.ones_like(X_tensor[:, :1])

    predictive = Predictive(model.model, guide=guide, num_samples=num_samples, return_sites=("_RETURN",)) # Chỉ cần _RETURN
    with torch.no_grad():
        preds = predictive(X_tensor, None)
    y_pred_samples = preds['_RETURN'] # Shape: (num_samples, num_data_points, output_dim)
    if y_pred_samples.shape[-1] != 1: # Đảm bảo output_dim là 1
         logging.warning(f"Prediction samples have unexpected last dimension: {y_pred_samples.shape}. Squeezing.")
         y_pred_samples = y_pred_samples.squeeze(-1) # Bỏ chiều cuối nếu thừa
         if y_pred_samples.ndim == 2: # Nếu còn 2 chiều (samples, data) -> thêm chiều output
              y_pred_samples = y_pred_samples.unsqueeze(-1)

    y_mean = y_pred_samples.mean(dim=0)
    y_std = y_pred_samples.std(dim=0)
    return y_mean, y_std


# --- Uncertainty Calibration Check --- (Giữ nguyên)
def check_uncertainty_calibration(y_true_np: np.ndarray, y_pred_mean_np: np.ndarray, y_pred_std_np: np.ndarray):
    # ... (giữ nguyên)
    logging.info("--- Uncertainty Calibration Check ---")
    for std_multiple in [1, 2]: # Giảm bớt kiểm tra
        lower_bound = y_pred_mean_np - std_multiple * y_pred_std_np
        upper_bound = y_pred_mean_np + std_multiple * y_pred_std_np
        within_bounds = ((y_true_np >= lower_bound) & (y_true_np <= upper_bound)).mean()
        theoretical_coverage = {1: 0.6827, 2: 0.9545}
        logging.info(f"Coverage within +/- {std_multiple} std dev: {within_bounds:.4f} (Theoretical: {theoretical_coverage[std_multiple]:.4f})")
    logging.info("------------------------------------")


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, input_dim) -> float:
    # 1. Đề xuất Siêu tham số
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True) # Khoảng rộng hơn
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    # >>> THÊM <<<
    num_layers = trial.suggest_int("num_layers", 1, 3) # Số lớp ẩn
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4) # Tỷ lệ dropout
    # ---
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    svi_steps = trial.suggest_int("svi_steps", 150, 400, step=50)

    epochs = DEFAULT_EPOCHS

    # logging.info(f"\n--- Optuna Trial {trial.number} ---") # Giảm log
    # logging.info(f"Params: lr={lr:.6f}, layers={num_layers}, hidden={hidden_dim}, dropout={dropout_rate:.2f}, batch={batch_size}, svi_steps={svi_steps}")

    # 2. Khởi tạo Model, Guide, Optimizer, SVI
    pyro.clear_param_store()
    # >>> SỬ DỤNG Enhanced BNN <<<
    model = EnhancedBayesianNeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_hidden_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(DEVICE)
    # ---
    guide = model.guide # Sử dụng guide được định nghĩa trong EnhancedBNN
    optimizer = Adam({"lr": lr})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, 'min', patience=5, factor=0.5, verbose=False)
    svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())

    # 3. Training Loop với Early Stopping
    best_trial_val_loss = float('inf')
    epochs_no_improve = 0
    best_guide_state_for_trial = None

    try: # Bọc vòng lặp train trong try-except
        for epoch in range(epochs):
            model.train()
            # Guide không có hàm train/eval trừ khi nó là nn.Module phức tạp
            # guide.train()
            epoch_loss = 0.0
            for step in range(svi_steps):
                indices = torch.randperm(X_train_tensor.size(0), device=DEVICE)[:batch_size]
                batch_X = X_train_tensor[indices]
                batch_y = y_train_tensor[indices]
                if batch_y.shape[-1] != 1: batch_y = batch_y.view(-1, 1)

                loss = svi.step(batch_X, batch_y)
                if not torch.isfinite(loss): # Kiểm tra NaN/Inf trong loss
                    logging.warning(f"Trial {trial.number}: Non-finite loss detected ({loss}). Pruning trial.")
                    raise optuna.exceptions.TrialPruned()
                epoch_loss += loss.item() # Lấy giá trị Python

            avg_epoch_loss = epoch_loss / (svi_steps * batch_size)

            # Validation
            model.eval()
            # guide.eval()
            val_loss = svi.evaluate_loss(X_val_tensor, y_val_tensor) / X_val_tensor.size(0)
            if not torch.isfinite(val_loss): # Kiểm tra NaN/Inf trong val_loss
                 logging.warning(f"Trial {trial.number}: Non-finite validation loss detected ({val_loss}). Pruning trial.")
                 raise optuna.exceptions.TrialPruned()
            val_loss_py = val_loss.item()

            # Update LR Scheduler
            scheduler.step(val_loss_py)

            # Early Stopping Logic
            if val_loss_py < best_trial_val_loss:
                best_trial_val_loss = val_loss_py
                epochs_no_improve = 0
                best_guide_state_for_trial = pyro.get_param_store().get_state()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                # logging.info(f"Trial {trial.number}: Early stopping at epoch {epoch+1}. Best Val Loss: {best_trial_val_loss:.6f}")
                break

            # Optuna Pruning
            trial.report(val_loss_py, epoch)
            if trial.should_prune():
                 # logging.info(f"Trial {trial.number}: Pruned by Optuna at epoch {epoch+1}.")
                 raise optuna.exceptions.TrialPruned()

    except RuntimeError as e:
        # Bắt các lỗi runtime thường gặp trong PyTorch/Pyro
        logging.warning(f"Trial {trial.number}: Runtime Error during training: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    # Lưu guide tốt nhất của trial này vào user_attrs để callback lấy
    if best_guide_state_for_trial:
         trial.set_user_attr("best_guide_state", best_guide_state_for_trial)
         # logging.info(f"Trial {trial.number} finished. Best Val Loss: {best_trial_val_loss:.6f}")
         return best_trial_val_loss
    else:
         logging.warning(f"Trial {trial.number}: No best guide state saved. Returning infinity.")
         return float('inf')


# --- Main Execution ---
if __name__ == "__main__":
    main_start_time = time.time()

    # --- 1. Load and Prepare Data ---
    all_df_list = []
    logging.info("Loading and preparing data...")
    for symbol in SYMBOLS_TO_TRAIN:
        df_symbol = load_and_prepare_bnn_data(symbol)
        if df_symbol is not None:
             all_df_list.append(df_symbol)
    if not all_df_list: exit()
    full_df = pd.concat(all_df_list, ignore_index=True)
    if isinstance(full_df.index, pd.DatetimeIndex): full_df = full_df.sort_index()
    logging.info(f"Total data points: {len(full_df)}")

    # --- 2. Time Series Split ---
    n_total = len(full_df)
    split_idx = int(n_total * (1 - VALIDATION_SPLIT_RATIO))
    if split_idx < 50 or n_total - split_idx < 50: # Cần đủ dữ liệu ở cả 2 phần
         logging.error(f"Not enough data for time series split (Train: {split_idx}, Val: {n_total - split_idx}). Need at least 50 in each.")
         exit()
    train_df = full_df.iloc[:split_idx]
    val_df = full_df.iloc[split_idx:]

    # >>> Lấy INPUT_DIM từ dữ liệu sau khi chuẩn bị <<<
    # Điều này đảm bảo INPUT_DIM chính xác ngay cả khi một số feature bị thiếu/lỗi
    temp_X_train = train_df[INPUT_FEATURES].values
    ACTUAL_INPUT_DIM = temp_X_train.shape[1]
    logging.info(f"Actual INPUT_DIM based on prepared data: {ACTUAL_INPUT_DIM}")

    X_train_raw = train_df[INPUT_FEATURES].values
    y_train = train_df[TARGET_FEATURE].values.reshape(-1, 1)
    X_val_raw = val_df[INPUT_FEATURES].values
    y_val = val_df[TARGET_FEATURE].values.reshape(-1, 1)

    if np.any(~np.isfinite(X_train_raw)) or np.any(~np.isfinite(y_train)) or \
       np.any(~np.isfinite(X_val_raw)) or np.any(~np.isfinite(y_val)):
        logging.error("NaN or Inf detected after split. Aborting.")
        exit()

    # --- 3. Scale Data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    try: joblib.dump(scaler, SCALER_SAVE_PATH); logging.info(f"Scaler saved.")
    except Exception as e: logging.error(f"Failed to save scaler: {e}")

    # --- 4. Convert to Tensors ---
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    # --- 5. Optuna Hyperparameter Optimization ---
    logging.info(f"Starting Optuna ({OPTUNA_N_TRIALS} trials)... DB: {OPTUNA_DB_PATH}")
    study = optuna.create_study(
        direction="minimize", study_name="bnn_pnl_opt_v3",
        storage=OPTUNA_DB_PATH, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=5) # Cấu hình pruner chặt hơn
    )

    # Callback để lưu guide tốt nhất
    best_optuna_val_loss = float('inf')
    saved_best_guide_state = None # Lưu trạng thái vào bộ nhớ thay vì file ngay lập tức

    def save_best_guide_callback(study, trial):
        global best_optuna_val_loss, saved_best_guide_state
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            if trial.value < best_optuna_val_loss:
                best_optuna_val_loss = trial.value
                # Lấy state guide từ user attribute
                if "best_guide_state" in trial.user_attrs:
                    saved_best_guide_state = trial.user_attrs["best_guide_state"]
                    logging.info(f"Callback: Updated best guide state in memory from Trial {trial.number} (Val Loss: {trial.value:.6f})")

    try:
        study.optimize(
            lambda trial: objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, ACTUAL_INPUT_DIM), # Truyền ACTUAL_INPUT_DIM
            n_trials=OPTUNA_N_TRIALS,
            callbacks=[save_best_guide_callback],
            gc_after_trial=True # Dọn dẹp bộ nhớ sau mỗi trial
        )
    except Exception as optuna_err:
         logging.error(f"Optuna optimization failed: {optuna_err}", exc_info=True)

    # --- 6. Load Best Model and Evaluate ---
    if not study.best_trial:
        logging.error("Optuna did not find any successful trials. Cannot proceed.")
        exit()

    logging.info("\n--- Optuna Optimization Summary ---")
    logging.info(f"Best Trial Number: {study.best_trial.number}")
    logging.info(f"Best Value (Validation Loss): {study.best_value:.6f}")
    logging.info("Best Parameters:")
    for key, value in study.best_params.items(): logging.info(f"  {key}: {value}")
    logging.info("---------------------------------")


    if saved_best_guide_state is None:
        logging.error("Could not retrieve the best guide state from Optuna study. Cannot perform final evaluation or save the best guide.")
    else:
        # Lưu guide tốt nhất cuối cùng từ bộ nhớ vào file
        try:
             pyro.clear_param_store()
             pyro.get_param_store().set_state(saved_best_guide_state)
             pyro.get_param_store().save(FINAL_GUIDE_PARAMS_PATH)
             logging.info(f"Saved final optimized guide params from best trial to {FINAL_GUIDE_PARAMS_PATH}")
        except Exception as save_err:
             logging.error(f"Failed to save the final best guide state: {save_err}")
             exit() # Không thể tiếp tục nếu không lưu được guide

        # Tạo lại model với cấu trúc tốt nhất
        best_params = study.best_params
        final_model = EnhancedBayesianNeuralNetwork(
            input_dim=ACTUAL_INPUT_DIM, # Sử dụng dim thực tế
            hidden_dim=best_params.get("hidden_dim", 64),
            output_dim=1,
            num_hidden_layers=best_params.get("num_layers", 1),
            dropout_rate=best_params.get("dropout_rate", 0.0)
        ).to(DEVICE)
        final_guide = final_model.guide # Lấy guide tương ứng

        # Tải lại tham số guide tốt nhất vào store (để chắc chắn)
        pyro.clear_param_store()
        pyro.get_param_store().load(FINAL_GUIDE_PARAMS_PATH)

        # --- 7. Final Evaluation ---
        final_model.eval()
        # final_guide.eval() # Guide không cần eval trừ khi là nn.Module

        logging.info("Evaluating prediction accuracy on validation set using FINAL OPTIMIZED guide...")
        try:
            y_pred_mean, y_pred_std = predict_bnn(final_model, final_guide, X_val_tensor, num_samples=NUM_PREDICTION_SAMPLES)
            y_pred_mean_np = y_pred_mean.cpu().numpy()
            y_pred_std_np = y_pred_std.cpu().numpy()
            y_val_np = y_val # y_val đã reshape

            # Tính Metrics
            mse = mean_squared_error(y_val_np, y_pred_mean_np)
            mae = mean_absolute_error(y_val_np, y_pred_mean_np)
            r2 = r2_score(y_val_np, y_pred_mean_np)
            valid_preds = np.isfinite(y_val_np.flatten()) & np.isfinite(y_pred_mean_np.flatten()) # Chỉ tính corr trên giá trị hữu hạn
            correlation, p_value = scipy.stats.pearsonr(y_val_np.flatten()[valid_preds], y_pred_mean_np.flatten()[valid_preds]) if valid_preds.sum() > 1 else (np.nan, np.nan)
            avg_pred_std = np.mean(y_pred_std_np[np.isfinite(y_pred_std_np)]) if np.any(np.isfinite(y_pred_std_np)) else np.nan

            logging.info("--- FINAL BNN Prediction Performance (Validation Set) ---")
            logging.info(f"Mean Squared Error (MSE):      {mse:.6f}")
            logging.info(f"Mean Absolute Error (MAE):       {mae:.6f}")
            logging.info(f"R-squared (R²):                {r2:.4f}")
            logging.info(f"Pearson Correlation:           {correlation:.4f} (p-value: {p_value:.3e})")
            logging.info(f"Average Predictive Std Dev:    {avg_pred_std:.6f}")
            logging.info("---------------------------------------------------------")

            # Kiểm tra hiệu chuẩn Uncertainty
            check_uncertainty_calibration(y_val_np, y_pred_mean_np, y_pred_std_np)

            # Vẽ đồ thị cuối cùng
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                # Vẽ ít điểm hơn để tránh quá tải đồ thị
                plot_indices = np.random.choice(len(y_val_np), min(len(y_val_np), 2000), replace=False)
                plt.scatter(y_val_np[plot_indices], y_pred_mean_np[plot_indices], alpha=0.3, label="Predictions")
                # Chỉ vẽ error bar cho một phần nhỏ để nhìn rõ
                error_indices = np.random.choice(plot_indices, min(len(plot_indices), 200), replace=False)
                plt.errorbar(y_val_np[error_indices], y_pred_mean_np[error_indices], yerr=y_pred_std_np[error_indices], fmt='none', alpha=0.2, ecolor='gray', label='Predictive Std Dev (Sample)')
                plt.plot([y_val_np.min(), y_val_np.max()], [y_val_np.min(), y_val_np.max()], '--', color='red', label="Ideal Fit")
                plt.xlabel("Actual Future Return (%)")
                plt.ylabel("Predicted Future Return (%)")
                plt.title(f"FINAL BNN (Optuna Best Trial {study.best_trial.number}): Actual vs. Predicted")
                plt.legend()
                plt.grid(True)
                plot_save_path = MODEL_SAVE_DIR / "bnn_actual_vs_predicted_final.png"
                plt.savefig(plot_save_path)
                plt.close()
                logging.info(f"Final Actual vs Predicted plot saved to {plot_save_path}")
            except ImportError: pass
            except Exception as plot_err: logging.error(f"Error generating final plot: {plot_err}")

        except Exception as final_eval_err:
             logging.error(f"Error during final evaluation: {final_eval_err}", exc_info=True)

    main_total_time = time.time() - main_start_time
    logging.info(f"Total script execution time: {main_total_time:.2f} seconds")