import torch
from typing import Optional, Tuple, List, Dict, Any
import torch.nn as nn
import torch.optim as optim
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

# Import model và CONFIG từ api.py
try:
    from api import TransformerForecaster, CONFIG # Giả sử nằm cùng thư mục hoặc trong PYTHONPATH
except ImportError as e:
    logging.error(f"Could not import from api.py: {e}. Make sure it's accessible.")
    # Fallback definitions
    class TransformerForecaster(nn.Module):
        def __init__(self, input_dim, d_model, nhead, num_layers):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
            logging.warning("Using dummy TransformerForecaster due to import error.")
        def forward(self, x): return self.fc(x)
    CONFIG = {"symbols": ["BTC/USDT"], "timeframes": ["15m"]}
    logging.warning("Using dummy CONFIG due to import error.")
except Exception as imp_e:
    logging.error(f"An unexpected error occurred during import: {imp_e}")
    exit()


# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("train_transformer_forecaster_full_v3.log", mode='w'), # Log riêng v3
        logging.StreamHandler()
    ]
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# Paths
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

DATA_DIR = SCRIPT_DIR
MODEL_SAVE_DIR = SCRIPT_DIR / "trained_models_iss"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_SAVE_PATH = MODEL_SAVE_DIR / "iss_transformer_forecaster_final_optuna_v3.pth" # Tên file cuối
INPUT_SCALER_SAVE_PATH = MODEL_SAVE_DIR / "iss_transformer_input_scaler.pkl"
TARGET_SCALER_SAVE_PATH = MODEL_SAVE_DIR / "iss_transformer_target_scaler.pkl"

# Data Parameters
SYMBOLS_TO_TRAIN = CONFIG.get("symbols", ["BTC/USDT", "ETH/USDT"])
TIMEFRANE = "15m"
INPUT_FEATURES = ["close", "RSI", "ATR", "volatility", "volume"]
TARGET_FEATURE = "volatility"
ALL_FEATURES_TO_LOAD = list(set(INPUT_FEATURES + [TARGET_FEATURE]))
PREDICT_STEPS = 1
MIN_SEQUENCES_FOR_TRAINING = 100

# Optuna & Training Parameters
N_TRIALS = 50 # Số lần thử Optuna
EPOCHS_PER_TRIAL = 30 # Giảm số epochs cho mỗi trial để tiết kiệm thời gian WFV
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
assert abs(TRAIN_SPLIT + VALIDATION_SPLIT + TEST_SPLIT - 1.0) < 1e-9
EARLY_STOPPING_PATIENCE = 7 # Giảm patience một chút

# WFV Parameters
WFV_INITIAL_TRAIN_RATIO = 0.5 # Tỷ lệ dữ liệu ban đầu cho cửa sổ train WFV
WFV_TEST_RATIO = 0.1       # Tỷ lệ dữ liệu cho cửa sổ test WFV
WFV_STEP_RATIO = WFV_TEST_RATIO # Bước trượt bằng cửa sổ test
WFV_EPOCHS_PER_FOLD = 15      # Số epochs huấn luyện cho mỗi fold WFV
WFV_ACCEPTABLE_MAE_THRESHOLD_FACTOR = 1.6 # Ngưỡng chấp nhận WFV MAE (so với Val MAE)
WFV_ACCEPTABLE_MAE_STD_FACTOR = 0.35     # Ngưỡng chấp nhận độ lệch chuẩn WFV MAE

# Model Parameters
TRANSFORMER_INPUT_DIM = len(INPUT_FEATURES)

# --- Helper Functions ---
def load_and_prepare_data(symbol: str, required_cols: List[str]) -> Optional[pd.DataFrame]:
    """Tải dữ liệu, kiểm tra cột, xử lý NaN."""
    symbol_safe = symbol.replace('/', '_').replace(':', '')
    file_path = DATA_DIR / f"{symbol_safe}_data.pkl"
    if not file_path.exists(): return None
    try:
        try: data_dict = joblib.load(file_path)
        except:
            import pickle
            with open(file_path, 'rb') as f: data_dict = pickle.load(f)
        if TIMEFRANE not in data_dict or not isinstance(data_dict[TIMEFRANE], pd.DataFrame): return None
        df = data_dict[TIMEFRANE][required_cols].copy() # Chỉ lấy cột cần thiết
        if df.isnull().values.any(): # Kiểm tra NaN trước khi fill
            df = df.ffill().bfill() # Fill trước
        if df.isnull().values.any(): # Kiểm tra lại
            df.fillna(0, inplace=True) # Fill 0 nếu vẫn còn
        if df.isnull().values.any(): return None # Vẫn lỗi -> bỏ qua
        return df
    except Exception as e:
        logging.error(f"Err load/prep {symbol}: {e}", exc_info=False) # Giảm log lỗi
        return None

def create_sequences(feature_data: np.ndarray,
                     target_data: np.ndarray,
                     seq_length: int,
                     predict_steps: int
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Tạo sequences X và y từ dữ liệu đã scale."""
    X, y = [], []
    n_samples = len(feature_data)
    if n_samples != len(target_data): return None, None
    last_start_index = n_samples - seq_length - predict_steps
    if last_start_index < 0: return None, None
    for i in range(last_start_index + 1):
        X.append(feature_data[i : i + seq_length, :])
        y.append(target_data[i + seq_length + predict_steps - 1])
    if not X: return None, None
    X_np = np.array(X); y_np = np.array(y).reshape(-1, 1)
    if X_np.shape[0] != y_np.shape[0]: return None, None
    return X_np, y_np

def train_eval_model(trial: optuna.trial.Trial, # Thêm trial
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
    log_prefix = f"Trial {trial.number}" if trial else f"WFV Fold {fold_num}"

    for epoch in range(epochs):
        model.train()
        train_loss_scaled = 0.0; processed_batches = 0
        for batch_X, batch_y_scaled in train_loader:
            batch_X, batch_y_scaled = batch_X.to(DEVICE), batch_y_scaled.to(DEVICE)
            if batch_X.shape[0] <= 1: continue
            try:
                optimizer.zero_grad()
                outputs_scaled = model(batch_X)
                if outputs_scaled.shape[1] > 0:
                    loss_scaled = criterion(outputs_scaled[:, -1, :], batch_y_scaled)
                    loss_scaled.backward(); optimizer.step()
                    train_loss_scaled += loss_scaled.item(); processed_batches += 1
                else: continue
            except Exception: continue # Bỏ qua batch lỗi nhẹ nhàng hơn
        if processed_batches == 0: continue
        train_loss_scaled /= processed_batches

        # Validation
        model.eval()
        val_loss_scaled = 0.0; all_y_true_original = []; all_y_pred_original = []
        processed_val_batches = 0
        with torch.no_grad():
            for batch_X, batch_y_scaled in val_loader:
                batch_X, batch_y_scaled = batch_X.to(DEVICE), batch_y_scaled.to(DEVICE)
                if batch_X.shape[0] <= 0: continue
                try:
                    outputs_scaled = model(batch_X)
                    if outputs_scaled.shape[1] > 0:
                        loss_scaled = criterion(outputs_scaled[:, -1, :], batch_y_scaled)
                        val_loss_scaled += loss_scaled.item()
                        y_pred_scaled_last = outputs_scaled[:, -1, :].cpu().numpy()
                        y_true_scaled = batch_y_scaled.cpu().numpy()
                        y_pred_original = target_scaler.inverse_transform(y_pred_scaled_last)
                        y_true_original = target_scaler.inverse_transform(y_true_scaled)
                        all_y_true_original.extend(y_true_original.flatten())
                        all_y_pred_original.extend(y_pred_original.flatten())
                        processed_val_batches += 1
                    else: continue
                except Exception: continue # Bỏ qua batch lỗi

        if processed_val_batches == 0: current_epoch_val_mae = float('inf')
        else:
            val_loss_scaled /= processed_val_batches
            current_epoch_val_mae = mean_absolute_error(all_y_true_original, all_y_pred_original) if all_y_true_original else float('inf')

        last_val_mae = current_epoch_val_mae # Cập nhật MAE epoch cuối

        # Logging và Early Stopping
        log_msg = (f"{log_prefix} - Epoch {epoch+1}/{epochs} - "
                   f"TrainLoss: {train_loss_scaled:.5f} | ValLoss: {val_loss_scaled:.5f} | ValMAE: {current_epoch_val_mae:.6f}")
        if current_epoch_val_mae < best_val_mae:
            log_msg += " (New best!)"
            best_val_mae = current_epoch_val_mae
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        logging.info(log_msg)

        if no_improve_epochs >= patience:
            logging.info(f"{log_prefix} - Early stopping at epoch {epoch+1}.")
            break

        # Báo cáo cho Optuna (chỉ khi đang chạy Optuna trial)
        if trial:
            trial.report(current_epoch_val_mae, epoch)
            if trial.should_prune():
                 logging.info(f"{log_prefix} - Pruned at epoch {epoch+1}.")
                 raise optuna.exceptions.TrialPruned()

    return best_val_mae, last_val_mae # Trả về MAE tốt nhất và MAE cuối cùng

# --- Main Execution ---
if __name__ == "__main__":
    # --- Bước 1: Load Data ---
    loaded_data_list: List[Tuple[str, pd.DataFrame]] = []
    for symbol in SYMBOLS_TO_TRAIN:
        df = load_and_prepare_data(symbol, ALL_FEATURES_TO_LOAD)
        if df is not None: loaded_data_list.append((symbol, df))
    if not loaded_data_list: logging.error("No valid data loaded."); exit()
    logging.info(f"Loaded data for {len(loaded_data_list)} symbols.")

    # --- Bước 2: Fit Scalers ---
    all_df_list = [df for _, df in loaded_data_list]
    try:
        input_data = pd.concat([df[INPUT_FEATURES] for df in all_df_list], ignore_index=True)
        target_data = pd.concat([df[[TARGET_FEATURE]] for df in all_df_list], ignore_index=True)
        input_scaler = StandardScaler().fit(input_data.values)
        target_scaler = StandardScaler().fit(target_data.values.reshape(-1, 1))
        joblib.dump(input_scaler, INPUT_SCALER_SAVE_PATH); joblib.dump(target_scaler, TARGET_SCALER_SAVE_PATH)
        logging.info(f"Scalers fitted and saved.")
    except Exception as e: logging.error(f"Scaler fit/save error: {e}"); exit()

    # --- Bước 3: Scale Data ---
    all_scaled_features = input_scaler.transform(input_data.values)
    all_scaled_target = target_scaler.transform(target_data.values.reshape(-1, 1)).flatten()
    logging.info(f"Data scaled. Features shape: {all_scaled_features.shape}, Target shape: {all_scaled_target.shape}")

    # --- Bước 4: Chia Train/Val/Test ---
    n_total = len(all_scaled_features)
    test_size = int(n_total * TEST_SPLIT)
    val_size = int(n_total * VALIDATION_SPLIT)
    train_size = n_total - val_size - test_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0: logging.error("Data split error."); exit()
    # Chia theo THỜI GIAN (quan trọng cho time series)
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, n_total)
    logging.info(f"Data split (time-based): Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    train_features, train_target = all_scaled_features[train_indices], all_scaled_target[train_indices]
    val_features, val_target = all_scaled_features[val_indices], all_scaled_target[val_indices]
    test_features, test_target = all_scaled_features[test_indices], all_scaled_target[test_indices]

    # --- Bước 5: Optuna Hyperparameter Search ---
    logging.info(f"\n--- Stage 1: Optuna Hyperparameter Search ({N_TRIALS} trials) ---")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    def objective_optuna_wrapper(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        seq_length = trial.suggest_int("sequence_length", 30, 120, step=10)
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        possible_nheads = [h for h in [2, 4, 8, 16] if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", possible_nheads) if possible_nheads else 1
        num_layers = trial.suggest_int("num_layers", 1, 4)

        logging.debug(f"Trial {trial.number}: Params=LR:{lr:.1E}, BS:{batch_size}, SL:{seq_length}, DM:{d_model}, NH:{nhead}, NL:{num_layers}")

        X_train_t, y_train_t = create_sequences(train_features, train_target, seq_length, PREDICT_STEPS)
        X_val_t, y_val_t = create_sequences(val_features, val_target, seq_length, PREDICT_STEPS)
        if X_train_t is None or X_val_t is None or len(X_train_t) < MIN_SEQUENCES_FOR_TRAINING or len(X_val_t) < 10:
             raise optuna.exceptions.TrialPruned("Not enough sequences")

        try:
            train_ds_t = TensorDataset(torch.tensor(X_train_t, dtype=torch.float32), torch.tensor(y_train_t, dtype=torch.float32))
            val_ds_t = TensorDataset(torch.tensor(X_val_t, dtype=torch.float32), torch.tensor(y_val_t, dtype=torch.float32))
            # Giảm workers để tránh lỗi treo
            train_dl_t = DataLoader(train_ds_t, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
            val_dl_t = DataLoader(val_ds_t, batch_size=batch_size, shuffle=False, num_workers=0)
            model_t = TransformerForecaster(TRANSFORMER_INPUT_DIM, d_model, nhead, num_layers).to(DEVICE)
            best_trial_mae, _ = train_eval_model(trial, model_t, train_dl_t, val_dl_t, target_scaler, EPOCHS_PER_TRIAL, EARLY_STOPPING_PATIENCE, lr)
            return best_trial_mae # Trả về MAE tốt nhất của trial
        except optuna.exceptions.TrialPruned: raise # Re-raise prune
        except Exception as e: logging.error(f"Trial {trial.number} Error: {e}"); return float('inf') # Lỗi khác

    try:
        study.optimize(objective_optuna_wrapper, n_trials=N_TRIALS, timeout=7200)
    except Exception as e: logging.error(f"Optuna failed: {e}"); exit()

    if study.best_trial is None: logging.error("Optuna found no valid trials."); exit()
    best_params = study.best_params
    best_val_mae_optuna = study.best_value
    logging.info("\n--- Optuna Optimization Finished ---")
    logging.info(f"Best Val MAE (Optuna Stage): {best_val_mae_optuna:.6f}")
    logging.info(f"Best Parameters: {best_params}")

    # --- Bước 6: Walk-Forward Validation ---
    logging.info("\n--- Stage 2: Walk-Forward Validation ---")
    # Lấy lại tham số tốt nhất từ Optuna
    wfv_lr = best_params["lr"]
    wfv_batch_size = best_params["batch_size"]
    wfv_seq_length = best_params["seq_length"]
    wfv_d_model = best_params["d_model"]
    wfv_nhead = best_params["nhead"]
    wfv_num_layers = best_params["num_layers"]

    # Dữ liệu cho WFV là tập Train + Val
    wfv_features = np.concatenate((train_features, val_features), axis=0)
    wfv_target = np.concatenate((train_target, val_target), axis=0)
    n_wfv = len(wfv_features)
    wfv_initial_train_size = int(n_wfv * WFV_INITIAL_TRAIN_RATIO)
    wfv_test_size = int(n_wfv * WFV_TEST_RATIO)
    wfv_step = int(n_wfv * WFV_STEP_RATIO)
    if wfv_step < 1: wfv_step = 1 # Đảm bảo step > 0

    if n_wfv < wfv_initial_train_size + wfv_test_size:
        logging.error("Not enough data for WFV first fold."); exit()

    wfv_fold_maes = []
    num_wfv_folds = 0
    for i in range(wfv_initial_train_size, n_wfv - wfv_test_size + 1, wfv_step):
        train_idx_start, train_idx_end = i - wfv_initial_train_size, i
        test_idx_start, test_idx_end = i, i + wfv_test_size
        if test_idx_end > n_wfv: break
        num_wfv_folds += 1

        fold_train_feat, fold_train_targ = wfv_features[train_idx_start:train_idx_end], wfv_target[train_idx_start:train_idx_end]
        fold_test_feat, fold_test_targ = wfv_features[test_idx_start:test_idx_end], wfv_target[test_idx_start:test_idx_end]

        X_train_f, y_train_f = create_sequences(fold_train_feat, fold_train_targ, wfv_seq_length, PREDICT_STEPS)
        X_test_f, y_test_f = create_sequences(fold_test_feat, fold_test_targ, wfv_seq_length, PREDICT_STEPS)

        if X_train_f is None or X_test_f is None or len(X_train_f) < MIN_SEQUENCES_FOR_TRAINING // 2 or len(X_test_f) < 5:
             logging.warning(f"WFV Fold {num_wfv_folds}: Skipping due to insufficient sequences.")
             continue

        try:
            train_ds_f = TensorDataset(torch.tensor(X_train_f, dtype=torch.float32), torch.tensor(y_train_f, dtype=torch.float32))
            test_ds_f = TensorDataset(torch.tensor(X_test_f, dtype=torch.float32), torch.tensor(y_test_f, dtype=torch.float32))
            train_dl_f = DataLoader(train_ds_f, batch_size=wfv_batch_size, shuffle=True, drop_last=True, num_workers=0)
            test_dl_f = DataLoader(test_ds_f, batch_size=wfv_batch_size, shuffle=False, num_workers=0)
            model_f = TransformerForecaster(TRANSFORMER_INPUT_DIM, wfv_d_model, wfv_nhead, wfv_num_layers).to(DEVICE)
            # Huấn luyện nhanh cho fold (không cần trial object) - Không có early stopping ở đây
            optimizer_f = optim.Adam(model_f.parameters(), lr=wfv_lr)
            criterion_f = nn.HuberLoss(delta=1.0)
            for epoch in range(WFV_EPOCHS_PER_FOLD):
                 model_f.train()
                 for batch_X, batch_y in train_dl_f:
                      batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                      optimizer_f.zero_grad()
                      outputs = model_f(batch_X)
                      if outputs.shape[1] > 0:
                          loss = criterion_f(outputs[:, -1, :], batch_y)
                          loss.backward(); optimizer_f.step()

            # Đánh giá fold
            model_f.eval()
            fold_preds, fold_true = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_dl_f:
                     batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                     outputs = model_f(batch_X)
                     if outputs.shape[1] > 0:
                          pred_s = outputs[:, -1, :].cpu().numpy()
                          true_s = batch_y.cpu().numpy()
                          fold_preds.extend(target_scaler.inverse_transform(pred_s).flatten())
                          fold_true.extend(target_scaler.inverse_transform(true_s).flatten())
            if fold_true:
                fold_mae = mean_absolute_error(fold_true, fold_preds)
                logging.info(f"WFV Fold {num_wfv_folds} MAE: {fold_mae:.6f}")
                wfv_fold_maes.append(fold_mae)
            else: logging.warning(f"WFV Fold {num_wfv_folds}: No predictions.")

        except Exception as e: logging.error(f"WFV Fold {num_wfv_folds} Error: {e}")

    # --- Bước 7: Quyết định dựa trên WFV ---
    if not wfv_fold_maes: logging.error("WFV failed."); exit()
    avg_wfv_mae = np.mean(wfv_fold_maes)
    std_wfv_mae = np.std(wfv_fold_maes)
    logging.info(f"\n--- WFV Results ---")
    logging.info(f"Avg MAE: {avg_wfv_mae:.6f}, Std Dev: {std_wfv_mae:.6f}, Folds: {num_wfv_folds}")

    acceptable_mae = best_val_mae_optuna * WFV_ACCEPTABLE_MAE_THRESHOLD_FACTOR
    acceptable_std = avg_wfv_mae * WFV_ACCEPTABLE_MAE_STD_FACTOR

    if avg_wfv_mae > acceptable_mae or std_wfv_mae > acceptable_std:
        logging.error(f"WFV performance NOT acceptable (AvgMAE > {acceptable_mae:.6f} or StdDev > {acceptable_std:.6f}). Aborting final training.")
        exit()
    else:
        logging.info("WFV performance acceptable. Proceeding to final training.")

    # --- Bước 8: Huấn luyện Model Cuối Cùng ---
    logging.info("\n--- Stage 3: Training Final Model on Train+Val ---")
    # Dùng lại tham số WFV (là best_params)
    X_train_val_final, y_train_val_final = create_sequences(wfv_features, wfv_target, wfv_seq_length, PREDICT_STEPS)
    if X_train_val_final is None: logging.error("Final sequence creation failed."); exit()
    final_dataset = TensorDataset(torch.tensor(X_train_val_final, dtype=torch.float32), torch.tensor(y_train_val_final, dtype=torch.float32))
    final_train_loader = DataLoader(final_dataset, batch_size=wfv_batch_size, shuffle=True, drop_last=True, num_workers=0)
    # Số epochs cuối = gấp đôi số epochs trial hoặc nhiều hơn một chút
    final_epochs = EPOCHS_PER_TRIAL * 2 + 10
    logging.info(f"Training final model for {final_epochs} epochs...")
    final_model = TransformerForecaster(TRANSFORMER_INPUT_DIM, wfv_d_model, wfv_nhead, wfv_num_layers).to(DEVICE)
    final_optimizer = optim.Adam(final_model.parameters(), lr=wfv_lr)
    final_criterion = nn.HuberLoss(delta=1.0)
    for epoch in range(final_epochs):
        final_model.train(); epoch_loss = 0.0; batches = 0
        for batch_X, batch_y in final_train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            final_optimizer.zero_grad()
            outputs = final_model(batch_X)
            if outputs.shape[1] > 0:
                loss = final_criterion(outputs[:, -1, :], batch_y)
                loss.backward(); final_optimizer.step()
                epoch_loss += loss.item(); batches += 1
        if batches > 0 and (epoch + 1) % 10 == 0:
             logging.info(f"Final Train Epoch {epoch+1}/{final_epochs} - Avg Loss: {epoch_loss/batches:.6f}")

    torch.save(final_model.state_dict(), FINAL_MODEL_SAVE_PATH)
    logging.info(f"Final model saved to {FINAL_MODEL_SAVE_PATH}")

    # --- Bước 9: Đánh giá trên Test Set ---
    logging.info("\n--- Stage 4: Evaluating Final Model on Test Set ---")
    X_test_seq, y_test_seq = create_sequences(test_features, test_target, wfv_seq_length, PREDICT_STEPS)
    if X_test_seq is None: logging.error("Cannot create test sequences.")
    else:
        try:
            test_ds = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32))
            test_loader = DataLoader(test_ds, batch_size=wfv_batch_size, shuffle=False, num_workers=0)
            final_model.eval()
            test_preds, test_true = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                     batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                     outputs = final_model(batch_X)
                     if outputs.shape[1] > 0:
                          pred_s = outputs[:, -1, :].cpu().numpy()
                          true_s = batch_y.cpu().numpy()
                          test_preds.extend(target_scaler.inverse_transform(pred_s).flatten())
                          test_true.extend(target_scaler.inverse_transform(true_s).flatten())
            if test_true:
                test_mae = mean_absolute_error(test_true, test_preds)
                test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
                logging.info(f"--- Final Model Evaluation on **TEST SET** ---")
                logging.info(f"Test MAE (Original Scale): {test_mae:.6f}")
                logging.info(f"Test RMSE (Original Scale): {test_rmse:.6f}")
                logging.info(f"Number of Test Samples Used: {len(test_true)}")
                logging.info(f"---------------------------------------------")
            else: logging.warning("No test predictions.")
        except Exception as e: logging.error(f"Test evaluation error: {e}")

    logging.info("\n--- Script Finished ---")