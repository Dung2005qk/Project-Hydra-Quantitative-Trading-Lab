import torch
from typing import Optional, Tuple, List, Dict, Any
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import optuna # Import Optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error # Thêm MSE để tính RMSE
import sqlite3
import shutil # Thêm thư viện để copy file

# Import model và CONFIG từ api.py
class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.encoder = nn.Linear(input_dim, d_model)
        # Sử dụng batch_first=True để quản lý shape dễ dàng hơn
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        # Bỏ logging ở init để tránh log thừa khi load lại model nhiều lần
        # logging.info(f"Initializing TransformerForecaster with input_dim={input_dim}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dropout={dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Xử lý dữ liệu đầu vào và trả về dự đoán."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Kích thước chiều cuối của tensor đầu vào ({x.shape[-1]}) không khớp với input_dim của mô hình ({self.input_dim})")
        # Di chuyển input tensor đến cùng device với model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        x = F.relu(self.encoder(x))  # Thêm activation để cải thiện tính phi tuyến
        x = self.transformer(x)
        return self.decoder(x)

    # --- ĐỔI TÊN PHƯƠNG THỨC TỪ predict THÀNH predict_volatility ---
    @torch.no_grad()
    def predict_volatility(self, lookback_data_tensor: torch.Tensor, forward_periods: int) -> np.ndarray:

        self.eval()  # Chuyển sang chế độ đánh giá

        device = next(self.parameters()).device
        if lookback_data_tensor.device != device:
             lookback_data_tensor = lookback_data_tensor.to(device)

        # --- Kiểm tra shape tensor đầu vào ---
        # Hàm này giờ nhận tensor đã được chuẩn bị bởi IntelligentStopSystem
        if not (lookback_data_tensor.ndim == 3 and lookback_data_tensor.shape[0] == 1 and lookback_data_tensor.shape[2] == self.input_dim):
             logging.error(f"TransformerForecaster predict_volatility: Invalid input tensor shape: {lookback_data_tensor.shape}. Expected (1, seq_len, {self.input_dim})")
             return np.full(forward_periods, np.nan)
        if lookback_data_tensor.shape[1] == 0:
             logging.error(f"TransformerForecaster predict_volatility: Input tensor has zero sequence length.")
             return np.full(forward_periods, np.nan)

        # Dự đoán và xử lý đầu ra
        try:
            # Gọi hàm forward để lấy kết quả dự đoán
            pred = self.forward(lookback_data_tensor) # Shape: (1, seq_len, 1)

            # Kiểm tra shape đầu ra
            if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[2] != 1 or pred.shape[1] == 0:
                logging.error(f"Shape đầu ra không mong đợi từ forward: {pred.shape}. Kỳ vọng (1, seq_len, 1)")
                return np.full(forward_periods, np.nan)

            # Lấy giá trị dự đoán cuối cùng (cho bước tiếp theo)
            last_pred_value = pred[0, -1, 0].item()

            # Trả về dự báo bằng cách lặp lại giá trị cuối cùng
            return np.array([last_pred_value] * forward_periods)

        except Exception as e:
            logging.error(f"Lỗi trong quá trình dự đoán volatility: {e}", exc_info=True)
            return np.full(forward_periods, np.nan)


# --- Configuration ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("train_transformer_forecaster_full_v3.log", mode='w'), # Log riêng v3
        logging.StreamHandler()
    ]
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# Paths
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# Trỏ DATA_DIR đến thư mục dataset "databot" trên Kaggle
DATASET_NAME = "databot" # <<< THAY TÊN DATASET CHỨA DỮ LIỆU GIÁ >>>
DATA_DIR = KAGGLE_INPUT_DIR / DATASET_NAME
logging.info(f"Data directory set to: {DATA_DIR}")

# *** START: Cấu hình Optuna Persistence ***
OPTUNA_DATASET_NAME = "optunadata" # <<< THAY TÊN DATASET CHỨA DB OPTUNA >>>
OPTUNA_DATASET_DIR = KAGGLE_INPUT_DIR / OPTUNA_DATASET_NAME
OPTUNA_DB_FILENAME = "optuna_study_transformer_resumable.db" # Tên file DB thống nhất
OPTUNA_DB_PATH_WORKING = KAGGLE_WORKING_DIR / OPTUNA_DB_FILENAME # Đường dẫn DB trong thư mục working (để ghi)
OPTUNA_DB_PATH_INPUT = OPTUNA_DATASET_DIR / OPTUNA_DB_FILENAME # Đường dẫn DB trong dataset input (để đọc/copy)
OPTUNA_STUDY_NAME = "transformer_volatility_v2_resumable" # Tên study cố định để load lại
STORAGE_NAME = f"sqlite:///{OPTUNA_DB_PATH_WORKING}" # Storage URI trỏ đến file trong working dir
# *** END: Cấu hình Optuna Persistence ***

# Lưu kết quả vào thư mục output của Kaggle
MODEL_SAVE_DIR = KAGGLE_WORKING_DIR / "trained_models_iss_kaggle" # Tạo thư mục con trong working
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Model save directory set to: {MODEL_SAVE_DIR}")

FINAL_MODEL_SAVE_PATH = MODEL_SAVE_DIR / "iss_transformer_forecaster_final_kaggle.pth"
INPUT_SCALER_SAVE_PATH = MODEL_SAVE_DIR / "iss_transformer_input_scaler_kaggle.pkl"
TARGET_SCALER_SAVE_PATH = MODEL_SAVE_DIR / "iss_transformer_target_scaler_kaggle.pkl"

# Data Parameters
SYMBOLS_TO_TRAIN = ["BTC/USDT", "ETH/USDT"]
TIMEFRANE = "15m"
INPUT_FEATURES = ["close", "RSI", "ATR", "volatility", "volume"]
TARGET_FEATURE = "volatility"
ALL_FEATURES_TO_LOAD = list(set(INPUT_FEATURES + [TARGET_FEATURE]))
PREDICT_STEPS = 1
MIN_SEQUENCES_FOR_TRAINING = 100

# Optuna & Training Parameters
# N_TRIALS giờ là mục tiêu số trials HOÀN THÀNH
N_TRIALS = 75 # Số lần thử Optuna HOÀN THÀNH cần đạt được
EPOCHS_PER_TRIAL = 50 # Giảm số epochs cho mỗi trial để tiết kiệm thời gian WFV
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
assert abs(TRAIN_SPLIT + VALIDATION_SPLIT + TEST_SPLIT - 1.0) < 1e-9
EARLY_STOPPING_PATIENCE = 15 # Giảm patience một chút

# WFV Parameters
WFV_INITIAL_TRAIN_RATIO = 0.5 # Tỷ lệ dữ liệu ban đầu cho cửa sổ train WFV
WFV_TEST_RATIO = 0.1       # Tỷ lệ dữ liệu cho cửa sổ test WFV
WFV_STEP_RATIO = WFV_TEST_RATIO # Bước trượt bằng cửa sổ test
WFV_EPOCHS_PER_FOLD = 25     # Số epochs huấn luyện cho mỗi fold WFV
WFV_ACCEPTABLE_MAE_THRESHOLD_FACTOR = 1.5 # Ngưỡng chấp nhận WFV MAE (so với Val MAE)
WFV_ACCEPTABLE_MAE_STD_FACTOR = 0.3     # Ngưỡng chấp nhận độ lệch chuẩn WFV MAE

# Model Parameters
TRANSFORMER_INPUT_DIM = len(INPUT_FEATURES)

# --- Helper Functions ---
def load_and_prepare_data(symbol: str, required_cols: List[str]) -> Optional[pd.DataFrame]:
    """Tải dữ liệu, kiểm tra cột, xử lý NaN."""
    symbol_safe = symbol.replace('/', '_').replace(':', '')
    file_path = DATA_DIR / f"{symbol_safe}_data.pkl"
    if not file_path.exists():
        logging.warning(f"Data file not found for {symbol} at {file_path}")
        return None
    try:
        # Thử joblib trước, nếu lỗi thì thử pickle
        try:
            data_dict = joblib.load(file_path)
        except:
            logging.warning(f"Failed loading {symbol} with joblib, trying pickle...")
            import pickle
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)

        if TIMEFRANE not in data_dict or not isinstance(data_dict.get(TIMEFRANE), pd.DataFrame):
            logging.warning(f"Timeframe '{TIMEFRANE}' not found or not a DataFrame for {symbol}.")
            return None

        # Kiểm tra xem các cột cần thiết có tồn tại không
        missing_cols = [col for col in required_cols if col not in data_dict[TIMEFRANE].columns]
        if missing_cols:
            logging.warning(f"Missing required columns for {symbol} in timeframe {TIMEFRANE}: {missing_cols}")
            # Có thể quyết định trả về None hoặc cố gắng tiếp tục với các cột có sẵn
            # return None # Hoặc comment dòng này để thử tiếp tục
            pass # Cho phép tiếp tục nếu chỉ thiếu vài cột không quan trọng

        # Chỉ lấy các cột thực sự có trong DataFrame và thuộc required_cols
        available_cols = [col for col in required_cols if col in data_dict[TIMEFRANE].columns]
        if not available_cols:
             logging.warning(f"No required columns available for {symbol} in timeframe {TIMEFRANE}.")
             return None

        df = data_dict[TIMEFRANE][available_cols].copy()

        # Kiểm tra và xử lý NaN hiệu quả hơn
        if df.isnull().values.any():
            logging.debug(f"NaNs detected in {symbol}. Applying ffill/bfill.")
            df = df.ffill().bfill()
        if df.isnull().values.any(): # Kiểm tra lại sau fill
            logging.warning(f"NaNs remain after ffill/bfill in {symbol}. Filling with 0.")
            df.fillna(0, inplace=True)
        # Kiểm tra cuối cùng, nếu vẫn còn NaN thì không dùng được
        if df.isnull().values.any():
            logging.error(f"Could not resolve all NaNs for {symbol}. Skipping.")
            return None

        # Kiểm tra kiểu dữ liệu số
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logging.warning(f"Column '{col}' in {symbol} is not numeric ({df[col].dtype}). Attempting conversion.")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Chuyển đổi, lỗi thành NaN
                    # Xử lý lại NaN nếu có sau chuyển đổi
                    if df[col].isnull().any():
                        logging.warning(f"NaNs introduced after converting '{col}' for {symbol}. Filling with 0.")
                        df[col].fillna(0, inplace=True)
                except Exception as e_conv:
                    logging.error(f"Failed to convert column '{col}' to numeric for {symbol}: {e_conv}. Skipping symbol.")
                    return None
        return df
    except FileNotFoundError:
        logging.warning(f"Data file not found for {symbol} at {file_path}")
        return None
    except EOFError:
         logging.error(f"EOFError encountered while loading {symbol} from {file_path}. File might be corrupted or empty.")
         return None
    except Exception as e:
        logging.error(f"Error loading/preparing data for {symbol}: {e}", exc_info=False) # Giảm log lỗi
        return None

def create_sequences(feature_data: np.ndarray,
                     target_data: np.ndarray,
                     seq_length: int,
                     predict_steps: int
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Tạo sequences X và y từ dữ liệu đã scale."""
    X, y = [], []
    n_samples = len(feature_data)
    # Kiểm tra đầu vào cơ bản
    if n_samples == 0 or seq_length <= 0 or predict_steps <= 0:
        logging.error(f"Invalid input to create_sequences: n_samples={n_samples}, seq_length={seq_length}, predict_steps={predict_steps}")
        return None, None
    if n_samples != len(target_data):
        logging.error(f"Feature data length ({n_samples}) != Target data length ({len(target_data)})")
        return None, None

    # Tính toán chỉ số cuối cùng để bắt đầu sequence
    # Sequence cuối cùng bắt đầu tại 'last_valid_start_index'
    # Nó sẽ lấy features từ [last_valid_start_index : last_valid_start_index + seq_length]
    # và target tại [last_valid_start_index + seq_length + predict_steps - 1]
    # Do đó, chỉ số target cao nhất cần truy cập là n_samples - 1
    # last_valid_start_index + seq_length + predict_steps - 1 <= n_samples - 1
    # last_valid_start_index <= n_samples - seq_length - predict_steps
    last_valid_start_index = n_samples - seq_length - predict_steps
    # last_start_index = n_samples - seq_length - predict_steps # Thay đổi logic chỉ số ở đây

    if last_valid_start_index < 0:
        logging.warning(f"Not enough data ({n_samples}) to create sequences with seq_length={seq_length} and predict_steps={predict_steps}")
        return None, None # Không đủ dữ liệu để tạo ít nhất một sequence

    for i in range(last_valid_start_index + 1): # Duyệt qua tất cả các điểm bắt đầu hợp lệ
        feature_seq_end = i + seq_length
        target_index = i + seq_length + predict_steps - 1 # Chỉ số của target tương ứng

        # Kiểm tra chỉ số một lần nữa (dù không thực sự cần nếu tính toán ở trên đúng)
        # if feature_seq_end > n_samples or target_index >= n_samples:
        #     logging.warning(f"Index out of bounds during sequence creation at step {i}. This shouldn't happen.")
        #     continue # Bỏ qua vòng lặp này nếu có lỗi logic chỉ số

        X.append(feature_data[i:feature_seq_end, :])
        y.append(target_data[target_index]) # Target là một giá trị vô hướng tại thời điểm tương lai

    if not X: # Nếu không có sequence nào được tạo (ví dụ: do lỗi index)
        logging.warning("No sequences were created.")
        return None, None

    try:
        X_np = np.array(X)
        y_np = np.array(y).reshape(-1, 1) # Đảm bảo y là cột vector
    except Exception as e:
        logging.error(f"Error converting sequences to numpy arrays: {e}")
        return None, None

    # Kiểm tra cuối cùng về số lượng mẫu khớp nhau
    if X_np.shape[0] != y_np.shape[0]:
        logging.error(f"Mismatch in number of sequences created: X={X_np.shape[0]}, y={y_np.shape[0]}")
        return None, None
    if X_np.shape[0] == 0:
         logging.warning("Created sequence arrays are empty.")
         return None, None

    return X_np, y_np


def train_eval_model(trial: Optional[optuna.trial.Trial], # Chấp nhận None khi không phải Optuna trial
                      model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      target_scaler: StandardScaler,
                      epochs: int,
                      patience: int,
                      learning_rate: float,
                      fold_num: Optional[int] = None # Thêm fold_num để log cho WFV
                     ) -> Tuple[float, float]:
    """Huấn luyện và đánh giá model, trả về (best_val_mae, last_val_mae)."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.HuberLoss(delta=1.0) # Sử dụng HuberLoss
    best_val_mae = float('inf')
    last_val_mae = float('inf') # MAE của epoch cuối cùng của trial/fold
    no_improve_epochs = 0
    # Xác định tiền tố log dựa trên việc có trial hay fold_num
    if trial:
        log_prefix = f"Trial {trial.number}"
    elif fold_num is not None:
        log_prefix = f"WFV Fold {fold_num}"
    else:
        log_prefix = "Final Train" # Hoặc một tiền tố mặc định khác

    for epoch in range(epochs):
        model.train()
        train_loss_scaled = 0.0; processed_batches = 0
        batch_count = 0
        for batch_X, batch_y_scaled in train_loader:
            batch_count += 1
            batch_X, batch_y_scaled = batch_X.to(DEVICE), batch_y_scaled.to(DEVICE)
            # Kiểm tra batch size > 0 và shape hợp lệ
            if batch_X.shape[0] <= 0 or batch_y_scaled.shape[0] <= 0 or batch_X.shape[0] != batch_y_scaled.shape[0]:
                # logging.warning(f"{log_prefix} - Epoch {epoch+1} - Skipping invalid batch {batch_count}/{len(train_loader)} with X_shape={batch_X.shape}, y_shape={batch_y_scaled.shape}")
                continue
            # Kiểm tra dim cuối của X
            if batch_X.ndim != 3 or batch_X.shape[2] != model.input_dim:
                 logging.warning(f"{log_prefix} - Epoch {epoch+1} - Skipping batch {batch_count} with invalid input dimensions: {batch_X.shape}. Expected (*, seq_len, {model.input_dim})")
                 continue


            try:
                optimizer.zero_grad()
                outputs_scaled = model(batch_X) # Shape: (batch, seq_len, 1)

                # Kiểm tra output shape trước khi tính loss
                # Cần lấy output ở bước cuối cùng của sequence
                if outputs_scaled is None or outputs_scaled.shape[0] != batch_X.shape[0] or outputs_scaled.shape[1] == 0 or outputs_scaled.shape[2] != 1:
                     logging.warning(f"{log_prefix} - Epoch {epoch+1} - Skipping batch {batch_count} due to unexpected model output shape: {outputs_scaled.shape if outputs_scaled is not None else 'None'}. Input shape was: {batch_X.shape}")
                     continue

                # Lấy output ở bước thời gian cuối cùng để so sánh với target
                last_step_output_scaled = outputs_scaled[:, -1, :] # Shape: (batch, 1)

                # Target cũng phải có shape (batch, 1)
                if batch_y_scaled.shape != last_step_output_scaled.shape:
                     # logging.warning(f"{log_prefix} - Epoch {epoch+1} - Reshaping target from {batch_y_scaled.shape} to {last_step_output_scaled.shape} for loss calculation in batch {batch_count}.")
                     try:
                          # Cố gắng reshape target nếu cần, ví dụ từ (batch,) sang (batch, 1)
                          batch_y_scaled = batch_y_scaled.view(last_step_output_scaled.shape)
                     except Exception as reshape_err:
                          logging.error(f"{log_prefix} - Epoch {epoch+1} - Failed to reshape target for batch {batch_count}. Target shape: {batch_y_scaled.shape}, Output shape: {last_step_output_scaled.shape}. Error: {reshape_err}")
                          continue # Bỏ qua batch này nếu không reshape được


                loss_scaled = criterion(last_step_output_scaled, batch_y_scaled)
                loss_scaled.backward(); optimizer.step()
                train_loss_scaled += loss_scaled.item(); processed_batches += 1

            except RuntimeError as e:
                 if "shape mismatch" in str(e):
                      logging.error(f"{log_prefix} - Epoch {epoch+1} - RuntimeError (Shape Mismatch) in batch {batch_count}: {e}. Input: {batch_X.shape}, Target: {batch_y_scaled.shape if 'batch_y_scaled' in locals() else 'N/A'}, Output: {outputs_scaled.shape if 'outputs_scaled' in locals() else 'N/A'}", exc_info=False)
                 else:
                      logging.error(f"{log_prefix} - Epoch {epoch+1} - Generic RuntimeError in batch {batch_count}: {e}. Input: {batch_X.shape}", exc_info=True)
                 # Có thể quyết định bỏ qua batch lỗi hoặc dừng hẳn
                 continue # Bỏ qua batch lỗi
            except Exception as e:
                 logging.error(f"{log_prefix} - Epoch {epoch+1} - Unexpected error in training batch {batch_count}: {e}. Input: {batch_X.shape}", exc_info=True)
                 continue # Bỏ qua batch lỗi

        if processed_batches == 0:
            logging.warning(f"{log_prefix} - Epoch {epoch+1} - No batches processed successfully in training.")
            # Có thể coi đây là lỗi nghiêm trọng và dừng sớm trial/fold nếu muốn
            # return float('inf'), float('inf') # Hoặc xử lý khác
            continue # Chuyển sang epoch tiếp theo

        train_loss_scaled /= processed_batches

        # Validation
        model.eval()
        val_loss_scaled = 0.0; all_y_true_original = []; all_y_pred_original = []
        processed_val_batches = 0
        with torch.no_grad():
            for batch_X, batch_y_scaled in val_loader:
                batch_X, batch_y_scaled = batch_X.to(DEVICE), batch_y_scaled.to(DEVICE)
                if batch_X.shape[0] <= 0 or batch_y_scaled.shape[0] <= 0 or batch_X.shape[0] != batch_y_scaled.shape[0]:
                     continue
                if batch_X.ndim != 3 or batch_X.shape[2] != model.input_dim:
                     logging.warning(f"{log_prefix} - Val Epoch {epoch+1} - Skipping batch with invalid input dimensions: {batch_X.shape}.")
                     continue

                try:
                    outputs_scaled = model(batch_X) # (batch, seq_len, 1)

                    if outputs_scaled is None or outputs_scaled.shape[0] != batch_X.shape[0] or outputs_scaled.shape[1] == 0 or outputs_scaled.shape[2] != 1:
                        logging.warning(f"{log_prefix} - Val Epoch {epoch+1} - Skipping batch due to unexpected model output shape: {outputs_scaled.shape if outputs_scaled is not None else 'None'}")
                        continue

                    last_step_output_scaled = outputs_scaled[:, -1, :] # (batch, 1)

                    # Đảm bảo target có shape (batch, 1)
                    if batch_y_scaled.shape != last_step_output_scaled.shape:
                         # logging.warning(f"{log_prefix} - Val Epoch {epoch+1} - Reshaping target from {batch_y_scaled.shape} to {last_step_output_scaled.shape} for loss.")
                          try: batch_y_scaled = batch_y_scaled.view(last_step_output_scaled.shape)
                          except: continue # Bỏ qua nếu reshape lỗi

                    loss_scaled = criterion(last_step_output_scaled, batch_y_scaled)
                    val_loss_scaled += loss_scaled.item()

                    # Inverse transform để tính MAE gốc
                    y_pred_scaled_last = last_step_output_scaled.cpu().numpy()
                    y_true_scaled = batch_y_scaled.cpu().numpy()

                    # Kiểm tra shape trước inverse_transform
                    if y_pred_scaled_last.ndim == 1: y_pred_scaled_last = y_pred_scaled_last.reshape(-1, 1)
                    if y_true_scaled.ndim == 1: y_true_scaled = y_true_scaled.reshape(-1, 1)

                    # Chỉ inverse transform nếu shape là (N, 1)
                    if y_pred_scaled_last.shape[1] == 1 and y_true_scaled.shape[1] == 1:
                        y_pred_original = target_scaler.inverse_transform(y_pred_scaled_last)
                        y_true_original = target_scaler.inverse_transform(y_true_scaled)
                        all_y_true_original.extend(y_true_original.flatten())
                        all_y_pred_original.extend(y_pred_original.flatten())
                        processed_val_batches += 1
                    else:
                         logging.warning(f"{log_prefix} - Val Epoch {epoch+1} - Skipping MAE calculation for batch due to incorrect shapes for inverse_transform. Pred shape: {y_pred_scaled_last.shape}, True shape: {y_true_scaled.shape}")


                except Exception as e:
                     logging.error(f"{log_prefix} - Val Epoch {epoch+1} - Error during validation batch: {e}. Input shape: {batch_X.shape}", exc_info=False)
                     continue # Bỏ qua batch lỗi

        if processed_val_batches == 0:
            logging.warning(f"{log_prefix} - Epoch {epoch+1} - No batches processed successfully in validation.")
            current_epoch_val_mae = float('inf') # Gán giá trị xấu nếu không có batch nào hợp lệ
        else:
            val_loss_scaled /= processed_val_batches
            # Tính MAE chỉ nếu có dữ liệu
            if all_y_true_original and all_y_pred_original:
                 try:
                      current_epoch_val_mae = mean_absolute_error(all_y_true_original, all_y_pred_original)
                 except ValueError as mae_err:
                      logging.error(f"{log_prefix} - Epoch {epoch+1} - Error calculating MAE: {mae_err}. True count: {len(all_y_true_original)}, Pred count: {len(all_y_pred_original)}")
                      current_epoch_val_mae = float('inf') # Lỗi tính toán -> kết quả xấu
            else:
                 current_epoch_val_mae = float('inf') # Không có dữ liệu -> kết quả xấu

        last_val_mae = current_epoch_val_mae # Cập nhật MAE epoch cuối

        # Logging và Early Stopping
        log_msg = (f"{log_prefix} - Epoch {epoch+1}/{epochs} - "
                   f"TrainLoss: {train_loss_scaled:.5f} | ValLoss: {val_loss_scaled:.5f} | ValMAE: {current_epoch_val_mae:.6f}")

        # Xử lý trường hợp MAE là inf hoặc nan
        is_new_best = False
        if np.isfinite(current_epoch_val_mae) and current_epoch_val_mae < best_val_mae:
            log_msg += " (New best!)"
            best_val_mae = current_epoch_val_mae
            no_improve_epochs = 0
            is_new_best = True
        elif np.isfinite(best_val_mae): # Chỉ tăng bộ đếm nếu best_val_mae đã hữu hạn
             no_improve_epochs += 1
        # Nếu best_val_mae vẫn là inf, không làm gì với no_improve_epochs

        logging.info(log_msg)

        # Logic Early Stopping (chỉ kích hoạt nếu best_val_mae không còn là inf)
        if np.isfinite(best_val_mae) and no_improve_epochs >= patience:
            logging.info(f"{log_prefix} - Early stopping at epoch {epoch+1} due to no improvement in ValMAE for {patience} epochs.")
            break

        # Báo cáo cho Optuna (chỉ khi đang chạy Optuna trial và MAE hữu hạn)
        if trial and np.isfinite(current_epoch_val_mae):
             try:
                  trial.report(current_epoch_val_mae, epoch)
                  if trial.should_prune():
                       logging.info(f"{log_prefix} - Pruned by Optuna at epoch {epoch+1}.")
                       raise optuna.exceptions.TrialPruned()
             except Exception as report_err:
                  # Bắt lỗi nếu report hoặc prune có vấn đề, nhưng vẫn tiếp tục epoch
                  logging.warning(f"{log_prefix} - Epoch {epoch+1} - Error during Optuna report/prune: {report_err}")


    # Đảm bảo trả về giá trị hữu hạn, nếu không tìm thấy best MAE thì trả về inf
    if not np.isfinite(best_val_mae):
         logging.warning(f"{log_prefix} - Finished training but best_val_mae is not finite ({best_val_mae}).")
    if not np.isfinite(last_val_mae):
        logging.warning(f"{log_prefix} - Finished training and last_val_mae is not finite ({last_val_mae}).")

    return best_val_mae, last_val_mae # Trả về MAE tốt nhất và MAE cuối cùng

# --- Main Execution ---
if __name__ == "__main__":
    # *** START: Sao chép Optuna DB từ Input vào Working Directory ***
    if OPTUNA_DB_PATH_INPUT.exists():
        try:
            logging.info(f"Found Optuna DB in input dataset: {OPTUNA_DB_PATH_INPUT}")
            shutil.copyfile(OPTUNA_DB_PATH_INPUT, OPTUNA_DB_PATH_WORKING)
            logging.info(f"Copied Optuna DB to working directory: {OPTUNA_DB_PATH_WORKING}")
        except Exception as e:
            logging.error(f"Failed to copy Optuna DB from {OPTUNA_DB_PATH_INPUT} to {OPTUNA_DB_PATH_WORKING}: {e}", exc_info=True)
            # Quyết định xem có nên dừng script hay không nếu không copy được
            # exit() # Hoặc chỉ cảnh báo và tiếp tục tạo DB mới
            logging.warning("Proceeding without the existing database file.")
    else:
        logging.info(f"Optuna DB not found in input dataset path: {OPTUNA_DB_PATH_INPUT}. A new DB will be created if needed.")
    # *** END: Sao chép Optuna DB ***


    # --- Bước 1: Load Data ---
    loaded_data_list: List[Tuple[str, pd.DataFrame]] = []
    for symbol in SYMBOLS_TO_TRAIN:
        df = load_and_prepare_data(symbol, ALL_FEATURES_TO_LOAD)
        if df is not None and not df.empty:
             # Kiểm tra lại sau khi load xem có đủ dữ liệu không
             if len(df) > MIN_SEQUENCES_FOR_TRAINING : # Cần đủ dài để tạo sequence
                 loaded_data_list.append((symbol, df))
             else:
                  logging.warning(f"Skipping {symbol} due to insufficient data after loading and cleaning ({len(df)} rows).")
        else:
             logging.warning(f"No valid data loaded for symbol: {symbol}")

    if not loaded_data_list:
        logging.error("No valid data loaded for any symbol. Cannot proceed.")
        exit()
    logging.info(f"Loaded data for {len(loaded_data_list)} symbols.")

    # --- Bước 2: Fit Scalers ---
    all_df_list = [df for _, df in loaded_data_list]
    # Kiểm tra lại lần nữa trước khi concat
    if not all_df_list:
        logging.error("all_df_list is empty after filtering symbols. Cannot fit scalers.")
        exit()

    try:
        # Chỉ concat các cột thực sự cần cho input và target
        input_data_frames = [df[INPUT_FEATURES] for df in all_df_list if all(col in df.columns for col in INPUT_FEATURES)]
        target_data_frames = [df[[TARGET_FEATURE]] for df in all_df_list if TARGET_FEATURE in df.columns]

        if not input_data_frames or not target_data_frames:
             logging.error("Could not extract input or target features from loaded dataframes.")
             exit()

        input_data = pd.concat(input_data_frames, ignore_index=True)
        target_data = pd.concat(target_data_frames, ignore_index=True)

        # Kiểm tra NaN trước khi fit scaler
        if input_data.isnull().values.any():
            logging.warning("NaNs found in combined input data before scaling. Filling with 0.")
            input_data.fillna(0, inplace=True)
        if target_data.isnull().values.any():
            logging.warning("NaNs found in combined target data before scaling. Filling with 0.")
            target_data.fillna(0, inplace=True)


        input_scaler = StandardScaler().fit(input_data.values)
        # Target scaler cần fit trên dữ liệu có shape (n_samples, 1)
        target_scaler = StandardScaler().fit(target_data.values.reshape(-1, 1))

        joblib.dump(input_scaler, INPUT_SCALER_SAVE_PATH); joblib.dump(target_scaler, TARGET_SCALER_SAVE_PATH)
        logging.info(f"Scalers fitted and saved. Input features: {input_scaler.n_features_in_}, Target samples: {target_scaler.n_samples_seen_}")
    except ValueError as ve:
         logging.error(f"ValueError during scaler fitting: {ve}. Input data shape: {input_data.shape if 'input_data' in locals() else 'N/A'}, Target data shape: {target_data.shape if 'target_data' in locals() else 'N/A'}", exc_info=True)
         exit()
    except Exception as e:
         logging.error(f"Scaler fit/save error: {e}", exc_info=True)
         exit()


    # --- Bước 3: Scale Data ---
    try:
        # Scale lại dữ liệu đã được kiểm tra NaN
        all_scaled_features = input_scaler.transform(input_data.values)
        all_scaled_target = target_scaler.transform(target_data.values.reshape(-1, 1)).flatten() # Flatten thành 1D array
        logging.info(f"Data scaled. Features shape: {all_scaled_features.shape}, Target shape: {all_scaled_target.shape}")
    except Exception as e:
         logging.error(f"Error scaling data: {e}", exc_info=True)
         exit()

    # --- Bước 4: Chia Train/Val/Test ---
    n_total = len(all_scaled_features)
    if n_total == 0:
         logging.error("No data available after scaling to perform train/val/test split.")
         exit()

    test_size = int(n_total * TEST_SPLIT)
    val_size = int(n_total * VALIDATION_SPLIT)
    train_size = n_total - val_size - test_size

    # Đảm bảo các tập không rỗng và train_size đủ lớn
    min_train_size = max(100, MIN_SEQUENCES_FOR_TRAINING + 50) # Cần đủ lớn để tạo sequences
    if train_size < min_train_size or val_size <= 0 or test_size <= 0:
        logging.error(f"Data split error or insufficient training data. Total: {n_total}, Train: {train_size} (min required: {min_train_size}), Val: {val_size}, Test: {test_size}")
        exit()

    # Chia theo THỜI GIAN (quan trọng cho time series)
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, n_total)
    logging.info(f"Data split (time-based): Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    train_features, train_target = all_scaled_features[train_indices], all_scaled_target[train_indices]
    val_features, val_target = all_scaled_features[val_indices], all_scaled_target[val_indices]
    test_features, test_target = all_scaled_features[test_indices], all_scaled_target[test_indices]


    # --- Bước 5: Optuna Hyperparameter Search (với Persistence) ---
    logging.info(f"\n--- Stage 1: Optuna Hyperparameter Search ({N_TRIALS} completed trials target) ---")
    logging.info(f"Using Optuna study '{OPTUNA_STUDY_NAME}' with storage: {STORAGE_NAME}")

    try:
        # Tải study hiện có hoặc tạo mới nếu không tồn tại trong file DB
        study = optuna.create_study(
            study_name=OPTUNA_STUDY_NAME,
            storage=STORAGE_NAME,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), # Thêm warmup steps
            load_if_exists=True  # Quan trọng: Tải nếu study đã tồn tại trong DB
        )
        logging.info(f"Successfully loaded or created Optuna study '{OPTUNA_STUDY_NAME}'.")

    except Exception as e:
        logging.error(f"Failed to create or load Optuna study '{OPTUNA_STUDY_NAME}' from {STORAGE_NAME}: {e}", exc_info=True)
        exit()

    # Hàm Objective cho Optuna (Giữ nguyên như trước)
    def objective_optuna_wrapper(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        seq_length_step = 30; seq_length_min = 90; seq_length_max = 240
        step_values = list(range(seq_length_min, seq_length_max + 1, seq_length_step))
        specific_values = [168, 192]; possible_seq_lengths = sorted(list(set(step_values + specific_values)))
        possible_seq_lengths = [l for l in possible_seq_lengths if seq_length_min <= l <= seq_length_max]
        # Thêm fallback nếu list rỗng (dù khó xảy ra)
        if not possible_seq_lengths: possible_seq_lengths = [120]
        seq_length = trial.suggest_categorical("sequence_length", possible_seq_lengths)

        d_model = trial.suggest_categorical("d_model", [128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        head_dim_min = 16
        if d_model == 128:
            # Đặt tên tham số riêng cho trường hợp d_model=128
            possible_nheads_128 = [h for h in [4, 8] if (128 / h) >= head_dim_min]
            if not possible_nheads_128: nhead = 8 # Fallback nếu list rỗng
            else: nhead = trial.suggest_categorical("nhead_128", possible_nheads_128) # Tên riêng
        elif d_model == 256:
            # Đặt tên tham số riêng cho trường hợp d_model=256
            possible_nheads_256 = [h for h in [8, 16] if (256 / h) >= head_dim_min]
            if not possible_nheads_256: nhead = 8 # Fallback
            else: nhead = trial.suggest_categorical("nhead_256", possible_nheads_256) # Tên riêng

        dropout_rate = trial.suggest_float("dropout", 0.1, 0.3)

        logging.debug(f"Trial {trial.number}: Params=LR:{lr:.1E}, BS:{batch_size}, SL:{seq_length}, DM:{d_model}, NH:{nhead}, NL:{num_layers}, DO:{dropout_rate:.2f}")

        # Tạo sequence trong objective để đảm bảo dùng đúng seq_length của trial
        X_train_t, y_train_t = create_sequences(train_features, train_target, seq_length, PREDICT_STEPS)
        X_val_t, y_val_t = create_sequences(val_features, val_target, seq_length, PREDICT_STEPS)

        # Kiểm tra dữ liệu sequence
        if X_train_t is None or y_train_t is None or X_val_t is None or y_val_t is None:
             logging.warning(f"Trial {trial.number}: Sequence creation failed for seq_length={seq_length}. Pruning.")
             raise optuna.exceptions.TrialPruned("Sequence creation failed")
        if len(X_train_t) < MIN_SEQUENCES_FOR_TRAINING or len(X_val_t) < 10:
             logging.warning(f"Trial {trial.number}: Not enough sequences after creation (Train: {len(X_train_t)}, Val: {len(X_val_t)}). Pruning.")
             raise optuna.exceptions.TrialPruned("Not enough sequences after creation")


        try:
            train_ds_t = TensorDataset(torch.tensor(X_train_t, dtype=torch.float32), torch.tensor(y_train_t, dtype=torch.float32))
            val_ds_t = TensorDataset(torch.tensor(X_val_t, dtype=torch.float32), torch.tensor(y_val_t, dtype=torch.float32))
            # Giảm workers để tránh lỗi treo, đặc biệt trong Kaggle
            train_dl_t = DataLoader(train_ds_t, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False) # Tắt pin_memory
            val_dl_t = DataLoader(val_ds_t, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

            # Khởi tạo model bên trong trial
            model_t = TransformerForecaster(TRANSFORMER_INPUT_DIM, d_model, nhead, num_layers, dropout=dropout_rate).to(DEVICE)
            logging.info(f"Trial {trial.number}: Initialized model with input_dim={TRANSFORMER_INPUT_DIM}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dropout={dropout_rate}")


            # Gọi hàm huấn luyện/đánh giá
            best_trial_mae, _ = train_eval_model(
                trial=trial, # Truyền trial object vào
                model=model_t,
                train_loader=train_dl_t,
                val_loader=val_dl_t,
                target_scaler=target_scaler,
                epochs=EPOCHS_PER_TRIAL,
                patience=EARLY_STOPPING_PATIENCE,
                learning_rate=lr,
                fold_num=None # Không phải WFV fold
            )

            # Kiểm tra giá trị trả về trước khi return cho Optuna
            if not np.isfinite(best_trial_mae):
                 logging.warning(f"Trial {trial.number} finished with non-finite best MAE ({best_trial_mae}). Reporting 'inf'.")
                 return float('inf') # Báo cáo giá trị xấu nếu huấn luyện lỗi
            else:
                return best_trial_mae # Trả về MAE tốt nhất của trial

        except optuna.exceptions.TrialPruned as e:
            logging.info(f"Trial {trial.number} was pruned: {e}")
            raise # Re-raise prune để Optuna xử lý
        except Exception as e:
            logging.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
            # Có thể chọn prune trial này hoặc trả về giá trị rất xấu
            # raise optuna.exceptions.TrialPruned(f"Runtime error: {e}")
            return float('inf') # Trả về giá trị xấu để Optuna không chọn params này

    # *** START: Vòng lặp chạy Optuna cho đến khi đủ N_TRIALS HOÀN THÀNH ***
    trials_run_this_session = 0
    max_consecutive_failures = 10 # Dừng nếu lỗi liên tục quá nhiều lần
    consecutive_failures = 0

    while True:
        # Lấy danh sách các trial đã chạy từ DB
        all_trials = study.get_trials(deepcopy=False) # deepcopy=False để hiệu quả hơn
        # Chỉ đếm những trial đã hoàn thành thành công
        completed_trials_count = sum(1 for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE)

        logging.info(f"Optuna status: {completed_trials_count}/{N_TRIALS} completed trials so far.")

        if completed_trials_count >= N_TRIALS:
            logging.info(f"Target number of completed trials ({N_TRIALS}) reached. Stopping Optuna optimization.")
            break

        if consecutive_failures >= max_consecutive_failures:
             logging.error(f"Stopping Optuna optimization due to {max_consecutive_failures} consecutive trial failures.")
             break

        remaining_target = N_TRIALS - completed_trials_count
        logging.info(f"Attempting to run the next trial (aiming for {remaining_target} more completed trials)...")

        try:
            # Chạy MỘT trial mới
            study.optimize(objective_optuna_wrapper, n_trials=1, # Chỉ chạy 1 trial mỗi lần gọi optimize
                           timeout=None, # Không giới hạn thời gian cho mỗi trial (nếu cần, đặt ở đây)
                           gc_after_trial=True) # Dọn dẹp bộ nhớ sau mỗi trial
            trials_run_this_session += 1
            # Reset bộ đếm lỗi nếu trial chạy (không nhất thiết phải thành công)
            consecutive_failures = 0
        except optuna.exceptions.TrialPruned:
             logging.info(f"A trial was pruned in this session.")
             # Không tăng consecutive_failures vì prune là hành vi dự kiến
        except Exception as e:
            logging.error(f"An unexpected error occurred during study.optimize: {e}", exc_info=True)
            consecutive_failures += 1
            # Có thể thêm delay nhỏ ở đây nếu lỗi liên quan đến tài nguyên
            # import time
            # time.sleep(5)

    # *** END: Vòng lặp chạy Optuna ***

    # Kiểm tra lại sau vòng lặp xem có đủ trial hoàn thành không
    final_completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if len(final_completed_trials) < N_TRIALS:
         logging.warning(f"Optuna optimization finished, but only {len(final_completed_trials)} trials completed successfully (target was {N_TRIALS}).")
         # Quyết định xem có nên dừng hay tiếp tục với best trial hiện có
         if not study.best_trial:
              logging.error("No best trial found after Optuna search. Cannot proceed.")
              exit()
         logging.warning("Proceeding with the best trial found so far.")
    else:
         logging.info(f"Optuna optimization completed successfully with {len(final_completed_trials)} completed trials.")


    # Lấy kết quả tốt nhất từ study (luôn tồn tại nếu có ít nhất 1 trial complete)
    if study.best_trial is None:
         # Điều này chỉ xảy ra nếu không có trial nào COMPLETE và vòng lặp bị dừng sớm
         logging.error("Optuna finished without any successful trial. Cannot proceed.")
         exit()

    best_params = study.best_trial.params
    best_val_mae_optuna = study.best_trial.value
    logging.info("\n--- Optuna Optimization Finished ---")
    logging.info(f"Best Trial Number: {study.best_trial.number}")
    logging.info(f"Best Val MAE (Optuna Stage): {best_val_mae_optuna:.6f}")
    logging.info(f"Best Parameters: {best_params}")

    # --- Bước 6: Walk-Forward Validation ---
    logging.info("\n--- Stage 2: Walk-Forward Validation ---")
    # Lấy lại tham số tốt nhất từ Optuna, xử lý key nhead có thể khác nhau
    wfv_lr = best_params["lr"]
    wfv_batch_size = best_params["batch_size"]
    wfv_seq_length = best_params["sequence_length"]
    wfv_d_model = best_params["d_model"]
    # Lấy nhead một cách an toàn hơn, tìm key bắt đầu bằng "nhead_"
    if wfv_d_model == 128:
        # Kiểm tra xem key có tồn tại không phòng trường hợp trial lỗi/prune
        if "nhead_128" in best_params:
             wfv_nhead = best_params["nhead_128"]
        else:
             logging.warning("Best trial had d_model=128 but 'nhead_128' key missing in params. Using fallback nhead=8.")
             wfv_nhead = 8 # Fallback
    elif wfv_d_model == 256:
        if "nhead_256" in best_params:
            wfv_nhead = best_params["nhead_256"]
        else:
             logging.warning("Best trial had d_model=256 but 'nhead_256' key missing in params. Using fallback nhead=8.")
             wfv_nhead = 8 # Fallback
    else:
        # Trường hợp d_model không phải 128 hoặc 256 (không nên xảy ra với suggest_categorical hiện tại)
        logging.error(f"Unexpected d_model value ({wfv_d_model}) in best_params. Cannot determine nhead. Using fallback 8.")
        wfv_nhead = 8

    wfv_num_layers = best_params["num_layers"]
    # Lấy dropout nếu có trong best_params, nếu không thì dùng giá trị mặc định hoặc 0.1
    wfv_dropout = best_params.get("dropout", 0.1) # Sử dụng .get với giá trị mặc định


    # Dữ liệu cho WFV là tập Train + Val
    wfv_features = np.concatenate((train_features, val_features), axis=0)
    wfv_target = np.concatenate((train_target, val_target), axis=0)
    n_wfv = len(wfv_features)

    # Tính toán kích thước cửa sổ WFV
    wfv_initial_train_size = int(n_wfv * WFV_INITIAL_TRAIN_RATIO)
    wfv_test_size = int(n_wfv * WFV_TEST_RATIO)
    wfv_step = int(n_wfv * WFV_STEP_RATIO)
    if wfv_step < 1: wfv_step = 1 # Đảm bảo step > 0

    # Kiểm tra xem có đủ dữ liệu cho ít nhất một fold WFV không
    min_data_for_wfv = wfv_initial_train_size + wfv_test_size
    if n_wfv < min_data_for_wfv:
        logging.error(f"Not enough data ({n_wfv}) for the first Walk-Forward Validation fold. Requires at least {min_data_for_wfv} data points (Initial Train: {wfv_initial_train_size}, Test: {wfv_test_size}). Aborting WFV.")
        # Quyết định: Có thể bỏ qua WFV và đi thẳng đến training cuối cùng, hoặc dừng hẳn
        # Ở đây ta sẽ dừng, vì WFV là bước quan trọng
        exit()
    else:
         logging.info(f"Starting WFV. Total data: {n_wfv}, Initial Train: {wfv_initial_train_size}, Test: {wfv_test_size}, Step: {wfv_step}")


    wfv_fold_maes = []
    num_wfv_folds = 0
    # Điểm bắt đầu của cửa sổ test đầu tiên là wfv_initial_train_size
    # Điểm kết thúc của dữ liệu là n_wfv
    # Cửa sổ test cuối cùng kết thúc tại chỉ số <= n_wfv
    # => test_idx_start + wfv_test_size <= n_wfv
    # => test_idx_start <= n_wfv - wfv_test_size
    for test_idx_start in range(wfv_initial_train_size, n_wfv - wfv_test_size + 1, wfv_step):
        num_wfv_folds += 1
        train_idx_end = test_idx_start
        # Cửa sổ training có thể là mở rộng (từ 0 đến train_idx_end) hoặc trượt (từ test_idx_start - wfv_initial_train_size đến train_idx_end)
        # Ở đây dùng cửa sổ mở rộng (expanding window)
        train_idx_start = 0
        test_idx_end = test_idx_start + wfv_test_size

        logging.info(f"--- WFV Fold {num_wfv_folds} ---")
        logging.info(f"Train indices: [{train_idx_start}, {train_idx_end}) - Size: {train_idx_end - train_idx_start}")
        logging.info(f"Test indices:  [{test_idx_start}, {test_idx_end}) - Size: {test_idx_end - test_idx_start}")


        fold_train_feat, fold_train_targ = wfv_features[train_idx_start:train_idx_end], wfv_target[train_idx_start:train_idx_end]
        fold_test_feat, fold_test_targ = wfv_features[test_idx_start:test_idx_end], wfv_target[test_idx_start:test_idx_end]

        # Kiểm tra kích thước dữ liệu fold
        if len(fold_train_feat) == 0 or len(fold_test_feat) == 0:
             logging.warning(f"WFV Fold {num_wfv_folds}: Skipping due to empty train or test data slice.")
             continue

        X_train_f, y_train_f = create_sequences(fold_train_feat, fold_train_targ, wfv_seq_length, PREDICT_STEPS)
        X_test_f, y_test_f = create_sequences(fold_test_feat, fold_test_targ, wfv_seq_length, PREDICT_STEPS)

        # Kiểm tra sequence fold, yêu cầu tối thiểu nới lỏng hơn cho WFV
        min_train_seq_wfv = max(10, MIN_SEQUENCES_FOR_TRAINING // 4)
        min_test_seq_wfv = 5
        if X_train_f is None or y_train_f is None or X_test_f is None or y_test_f is None:
             logging.warning(f"WFV Fold {num_wfv_folds}: Skipping due to sequence creation failure.")
             continue
        if len(X_train_f) < min_train_seq_wfv or len(X_test_f) < min_test_seq_wfv:
             logging.warning(f"WFV Fold {num_wfv_folds}: Skipping due to insufficient sequences (Train: {len(X_train_f)}, Test: {len(X_test_f)}). Min required: Train {min_train_seq_wfv}, Test {min_test_seq_wfv}.")
             continue

        try:
            train_ds_f = TensorDataset(torch.tensor(X_train_f, dtype=torch.float32), torch.tensor(y_train_f, dtype=torch.float32))
            test_ds_f = TensorDataset(torch.tensor(X_test_f, dtype=torch.float32), torch.tensor(y_test_f, dtype=torch.float32))
            train_dl_f = DataLoader(train_ds_f, batch_size=wfv_batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
            test_dl_f = DataLoader(test_ds_f, batch_size=wfv_batch_size, shuffle=False, num_workers=0, pin_memory=False)

            # Khởi tạo model mới cho mỗi fold WFV
            model_f = TransformerForecaster(TRANSFORMER_INPUT_DIM, wfv_d_model, wfv_nhead, wfv_num_layers, dropout=wfv_dropout).to(DEVICE)
            logging.info(f"WFV Fold {num_wfv_folds}: Initialized model with best params.")

            # Huấn luyện nhanh cho fold (không dùng trial object, không early stopping trong WFV)
            # Sử dụng hàm train_eval_model nhưng không tận dụng phần early stopping/best model của nó
            # Chỉ chạy đủ WFV_EPOCHS_PER_FOLD
            # Hoặc đơn giản hóa vòng lặp training ở đây:
            optimizer_f = optim.Adam(model_f.parameters(), lr=wfv_lr)
            criterion_f = nn.HuberLoss(delta=1.0)
            logging.info(f"WFV Fold {num_wfv_folds}: Starting training for {WFV_EPOCHS_PER_FOLD} epochs...")
            for epoch in range(WFV_EPOCHS_PER_FOLD):
                 model_f.train()
                 epoch_train_loss = 0.0
                 batches_processed = 0
                 for batch_X, batch_y in train_dl_f:
                      batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                      if batch_X.shape[0] <= 0: continue
                      optimizer_f.zero_grad()
                      outputs = model_f(batch_X)
                      if outputs is not None and outputs.shape[1] > 0:
                          last_step_output = outputs[:, -1, :]
                          # Đảm bảo target shape khớp
                          if batch_y.shape != last_step_output.shape:
                               try: batch_y = batch_y.view(last_step_output.shape)
                               except: continue # Bỏ qua batch nếu reshape lỗi
                          loss = criterion_f(last_step_output, batch_y)
                          loss.backward(); optimizer_f.step()
                          epoch_train_loss += loss.item()
                          batches_processed += 1
                 # Log loss huấn luyện của fold sau mỗi epoch (tùy chọn)
                 # if batches_processed > 0:
                 #      logging.debug(f"WFV Fold {num_wfv_folds} Epoch {epoch+1}/{WFV_EPOCHS_PER_FOLD} Train Loss: {epoch_train_loss/batches_processed:.6f}")

            logging.info(f"WFV Fold {num_wfv_folds}: Training complete.")

            # Đánh giá fold trên tập test của fold đó
            model_f.eval()
            fold_preds_original, fold_true_original = [], []
            with torch.no_grad():
                for batch_X, batch_y_scaled in test_dl_f:
                     batch_X, batch_y_scaled = batch_X.to(DEVICE), batch_y_scaled.to(DEVICE)
                     if batch_X.shape[0] <= 0: continue
                     try:
                         outputs_scaled = model_f(batch_X)
                         if outputs_scaled is not None and outputs_scaled.shape[1] > 0:
                              pred_s = outputs_scaled[:, -1, :].cpu().numpy() # (batch, 1)
                              true_s = batch_y_scaled.cpu().numpy() # (batch, 1) or need reshape

                              # Đảm bảo shape là (N, 1) trước inverse_transform
                              if pred_s.ndim == 1: pred_s = pred_s.reshape(-1, 1)
                              if true_s.ndim == 1: true_s = true_s.reshape(-1, 1)

                              if pred_s.shape[1] == 1 and true_s.shape[1] == 1:
                                  fold_preds_original.extend(target_scaler.inverse_transform(pred_s).flatten())
                                  fold_true_original.extend(target_scaler.inverse_transform(true_s).flatten())
                              else:
                                   logging.warning(f"WFV Fold {num_wfv_folds} Eval: Skipping batch due to incorrect shapes for inverse_transform. Pred: {pred_s.shape}, True: {true_s.shape}")

                     except Exception as eval_e:
                          logging.error(f"WFV Fold {num_wfv_folds}: Error during evaluation batch: {eval_e}", exc_info=False)


            if fold_true_original and fold_preds_original:
                fold_mae = mean_absolute_error(fold_true_original, fold_preds_original)
                logging.info(f"WFV Fold {num_wfv_folds} Test MAE (Original Scale): {fold_mae:.6f}")
                wfv_fold_maes.append(fold_mae)
            else:
                logging.warning(f"WFV Fold {num_wfv_folds}: No valid predictions generated for MAE calculation.")
                # Có thể thêm một giá trị MAE rất lớn vào list để phạt fold này
                # wfv_fold_maes.append(float('inf'))

        except Exception as e:
            logging.error(f"WFV Fold {num_wfv_folds} encountered an error: {e}", exc_info=True)
            # Có thể thêm giá trị MAE xấu nếu fold lỗi
            # wfv_fold_maes.append(float('inf'))

    # --- Bước 7: Quyết định dựa trên WFV ---
    if not wfv_fold_maes: # Nếu không có fold nào hoàn thành hoặc tất cả đều lỗi
         logging.error("Walk-Forward Validation failed to produce any results. Cannot proceed to final training.")
         # Có thể backup bằng cách chỉ dựa vào validation MAE ban đầu nếu muốn
         # logging.warning("Skipping WFV check and proceeding based on initial validation MAE.")
         # Hoặc dừng hẳn
         exit()


    avg_wfv_mae = np.mean([mae for mae in wfv_fold_maes if np.isfinite(mae)]) # Tính trung bình MAE hữu hạn
    std_wfv_mae = np.std([mae for mae in wfv_fold_maes if np.isfinite(mae)])   # Tính std dev MAE hữu hạn
    num_valid_folds = len([mae for mae in wfv_fold_maes if np.isfinite(mae)])

    logging.info(f"\n--- WFV Results ---")
    logging.info(f"Number of Folds Executed: {num_wfv_folds}")
    logging.info(f"Number of Folds with Valid MAE: {num_valid_folds}")
    if num_valid_folds > 0 :
        logging.info(f"Average MAE (Valid Folds): {avg_wfv_mae:.6f}")
        logging.info(f"Std Dev MAE (Valid Folds): {std_wfv_mae:.6f}")
    else:
        logging.error("No valid MAE scores obtained from WFV folds.")
        exit() # Dừng nếu không có fold nào thành công

    # Kiểm tra ngưỡng chấp nhận WFV
    # Cần best_val_mae_optuna phải hữu hạn để so sánh
    if not np.isfinite(best_val_mae_optuna):
         logging.error("Best validation MAE from Optuna is not finite. Cannot perform WFV check. Aborting.")
         exit()

    acceptable_mae = best_val_mae_optuna * WFV_ACCEPTABLE_MAE_THRESHOLD_FACTOR
    # Ngưỡng std dev có thể dựa trên MAE trung bình của WFV hoặc MAE tốt nhất của Optuna
    # Dựa trên MAE trung bình WFV có vẻ hợp lý hơn để đánh giá sự ổn định của WFV
    acceptable_std = avg_wfv_mae * WFV_ACCEPTABLE_MAE_STD_FACTOR

    logging.info(f"WFV Acceptance Thresholds: Max Avg MAE = {acceptable_mae:.6f}, Max Std Dev = {acceptable_std:.6f}")

    # Thực hiện kiểm tra
    wfv_passed = True
    if avg_wfv_mae > acceptable_mae:
        logging.error(f"WFV performance NOT acceptable: Average MAE ({avg_wfv_mae:.6f}) > Threshold ({acceptable_mae:.6f})")
        wfv_passed = False
    if std_wfv_mae > acceptable_std:
        logging.error(f"WFV performance NOT acceptable: Std Dev ({std_wfv_mae:.6f}) > Threshold ({acceptable_std:.6f})")
        wfv_passed = False

    if not wfv_passed:
        logging.error("Aborting final training due to unacceptable WFV performance.")
        exit()
    else:
        logging.info("WFV performance is acceptable. Proceeding to final training.")


    # --- Bước 8: Huấn luyện Model Cuối Cùng ---
    logging.info("\n--- Stage 3: Training Final Model on Train+Val ---")
    # Dùng lại tham số WFV (là best_params)
    # Tạo sequence từ dữ liệu wfv (Train+Val)
    X_train_val_final, y_train_val_final = create_sequences(wfv_features, wfv_target, wfv_seq_length, PREDICT_STEPS)

    if X_train_val_final is None or y_train_val_final is None or len(X_train_val_final) == 0:
        logging.error("Final sequence creation failed or resulted in empty data. Cannot train final model.")
        exit()
    logging.info(f"Created final training sequences. Shape X: {X_train_val_final.shape}, Shape y: {y_train_val_final.shape}")


    final_dataset = TensorDataset(torch.tensor(X_train_val_final, dtype=torch.float32), torch.tensor(y_train_val_final, dtype=torch.float32))
    final_train_loader = DataLoader(final_dataset, batch_size=wfv_batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    # Số epochs cuối = gấp đôi số epochs trial hoặc nhiều hơn một chút, đảm bảo đủ lớn
    final_epochs = max(EPOCHS_PER_TRIAL * 2, 75) # Tăng số epochs cuối
    logging.info(f"Training final model for {final_epochs} epochs...")

    # Khởi tạo model cuối cùng với params tốt nhất
    final_model = TransformerForecaster(TRANSFORMER_INPUT_DIM, wfv_d_model, wfv_nhead, wfv_num_layers, dropout=wfv_dropout).to(DEVICE)
    final_optimizer = optim.Adam(final_model.parameters(), lr=wfv_lr)
    final_criterion = nn.HuberLoss(delta=1.0)

    for epoch in range(final_epochs):
        final_model.train(); epoch_loss = 0.0; batches = 0
        for batch_X, batch_y in final_train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            if batch_X.shape[0] <= 0: continue
            try:
                final_optimizer.zero_grad()
                outputs = final_model(batch_X)
                if outputs is not None and outputs.shape[1] > 0:
                    last_step_output = outputs[:, -1, :]
                    if batch_y.shape != last_step_output.shape:
                         batch_y = batch_y.view(last_step_output.shape)

                    loss = final_criterion(last_step_output, batch_y)
                    loss.backward(); final_optimizer.step()
                    epoch_loss += loss.item(); batches += 1
            except Exception as final_train_e:
                 logging.error(f"Final Train Epoch {epoch+1} - Error in batch: {final_train_e}", exc_info=False)
                 continue # Bỏ qua batch lỗi

        if batches > 0 and (epoch + 1) % 10 == 0: # Log mỗi 10 epochs
             logging.info(f"Final Train Epoch {epoch+1}/{final_epochs} - Avg Loss: {epoch_loss/batches:.6f}")
        elif batches == 0:
             logging.warning(f"Final Train Epoch {epoch+1} - No batches processed.")


    # Lưu model cuối cùng
    try:
        torch.save(final_model.state_dict(), FINAL_MODEL_SAVE_PATH)
        logging.info(f"Final model state dictionary saved to {FINAL_MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Failed to save the final model: {e}", exc_info=True)

    # --- Bước 9: Đánh giá trên Test Set ---
    logging.info("\n--- Stage 4: Evaluating Final Model on Test Set ---")
    # Tạo sequence cho test set
    X_test_seq, y_test_seq = create_sequences(test_features, test_target, wfv_seq_length, PREDICT_STEPS)

    if X_test_seq is None or y_test_seq is None or len(X_test_seq) == 0:
        logging.error("Cannot create test sequences or test sequences are empty. Skipping final evaluation.")
    else:
        logging.info(f"Created test sequences. Shape X: {X_test_seq.shape}, Shape y: {y_test_seq.shape}")
        try:
            test_ds = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32))
            # Sử dụng batch size lớn hơn cho evaluation nếu có thể
            eval_batch_size = wfv_batch_size * 2
            test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=False)

            # Đảm bảo model ở chế độ eval
            final_model.eval()
            test_preds_original, test_true_original = [], []
            with torch.no_grad():
                for batch_X, batch_y_scaled in test_loader:
                     batch_X, batch_y_scaled = batch_X.to(DEVICE), batch_y_scaled.to(DEVICE)
                     if batch_X.shape[0] <= 0: continue
                     try:
                         outputs_scaled = final_model(batch_X)
                         if outputs_scaled is not None and outputs_scaled.shape[1] > 0:
                              pred_s = outputs_scaled[:, -1, :].cpu().numpy()
                              true_s = batch_y_scaled.cpu().numpy()

                              if pred_s.ndim == 1: pred_s = pred_s.reshape(-1, 1)
                              if true_s.ndim == 1: true_s = true_s.reshape(-1, 1)

                              if pred_s.shape[1] == 1 and true_s.shape[1] == 1:
                                  test_preds_original.extend(target_scaler.inverse_transform(pred_s).flatten())
                                  test_true_original.extend(target_scaler.inverse_transform(true_s).flatten())
                              else:
                                   logging.warning(f"Test Eval: Skipping batch due to incorrect shapes for inverse_transform. Pred: {pred_s.shape}, True: {true_s.shape}")

                     except Exception as test_eval_e:
                          logging.error(f"Test evaluation error in batch: {test_eval_e}", exc_info=False)

            # Tính toán và log kết quả cuối cùng
            if test_true_original and test_preds_original:
                test_mae = mean_absolute_error(test_true_original, test_preds_original)
                test_rmse = np.sqrt(mean_squared_error(test_true_original, test_preds_original))
                logging.info(f"--- Final Model Evaluation on **TEST SET** ---")
                logging.info(f"Test MAE (Original Scale): {test_mae:.6f}")
                logging.info(f"Test RMSE (Original Scale): {test_rmse:.6f}")
                logging.info(f"Number of Test Samples Evaluated: {len(test_true_original)}")
                logging.info(f"---------------------------------------------")
            else:
                logging.warning("No valid test predictions were generated. Cannot calculate final metrics.")
        except Exception as e:
            logging.error(f"An error occurred during final test set evaluation: {e}", exc_info=True)

    logging.info("\n--- Script Finished ---")