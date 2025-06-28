# %% [code]
# --- Core Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # random_split is not used directly here
import numpy as np
import joblib
import os
import sys # For logging handler
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report # accuracy_score is not used directly
from typing import Optional
import time
import optuna # Import Optuna
import gc # Import garbage collector
import shutil # Import for file copying

# %% [code]
# --- Kaggle Path Configuration ---
PROCESSED_DATA_DATASET_SLUG = "processed-lob-data" # Dataset với X, y
# !!! THAY THẾ BẰNG SLUG DATASET CHỨA FILE .db VÀ .pkl CỦA BẠN !!!
OPTUNA_STATE_DATASET_SLUG = "trained-models-lob-optuna" # <<<<<<<<<<<<<<<<<<< THAY ĐỔI Ở ĐÂY

KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working") # Output directory

PROCESSED_DATA_DIR = KAGGLE_INPUT_DIR / PROCESSED_DATA_DATASET_SLUG
OPTUNA_STATE_INPUT_DIR = KAGGLE_INPUT_DIR / OPTUNA_STATE_DATASET_SLUG

# --- Model Definition (Giữ nguyên phiên bản đã sửa) ---
INPUT_DIM_EXPECTED = 20
D_MODEL_ORIG = 64
NHEAD_ORIG = 4
NUM_LAYERS_ORIG = 2

class LOBTransformer(nn.Module):
    def __init__(self, input_dim=INPUT_DIM_EXPECTED, d_model=D_MODEL_ORIG, nhead=NHEAD_ORIG, num_layers=NUM_LAYERS_ORIG, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim; self.d_model = d_model; self.nhead = nhead; self.num_layers = num_layers
        self.encoder = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, d_model)
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(src)
        encoded_for_transformer = encoded.unsqueeze(1)
        transformer_output_raw = self.transformer(encoded_for_transformer)
        transformer_features = transformer_output_raw.squeeze(1)
        lob_features_output = self.decoder(transformer_features)
        return lob_features_output

# %% [code]
# --- Configuration ---
LOG_FILE_PATH = KAGGLE_WORKING_DIR / "train_lob_transformer_optuna_continue.log" # Đổi tên log file nếu muốn

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
logging.basicConfig(
    level=logging.DEBUG, # Keep DEBUG to see epoch progress
    format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[ logging.FileHandler(LOG_FILE_PATH, encoding='utf-8', mode='w'), logging.StreamHandler(sys.stdout) ]
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")
logging.info(f"Input Data Directory: {PROCESSED_DATA_DIR}")
logging.info(f"Optuna State Input Directory: {OPTUNA_STATE_INPUT_DIR}")
logging.info(f"Output (Working) Directory: {KAGGLE_WORKING_DIR}")
logging.info("--- Using LOBTransformer defined directly in the notebook (Training Version) ---")

# --- Paths ---
# Output paths in working directory (nơi file mới sẽ được ghi/cập nhật)
MODEL_SAVE_DIR = KAGGLE_WORKING_DIR / "trained_models_lob_optuna"
MODEL_SAVE_DIR.mkdir(exist_ok=True)
FINAL_MODEL_SAVE_PATH = MODEL_SAVE_DIR / "lob_transformer_backbone_optuna_final.pth"
WORKING_SCALER_PATH = MODEL_SAVE_DIR / "lob_input_scaler_optuna.pkl" # Scaler sẽ được load vào đây (hoặc ghi đè)
WORKING_DB_PATH_OBJ = MODEL_SAVE_DIR / "lob_optuna_study.db"        # DB sẽ được copy vào đây và cập nhật
OPTUNA_DB_STORAGE_URI = f"sqlite:///{WORKING_DB_PATH_OBJ}"          # Optuna sử dụng URI này

# Input paths from datasets
INPUT_X_PATH = PROCESSED_DATA_DIR / "combined_lob_training_X.npy"
INPUT_Y_PATH = PROCESSED_DATA_DIR / "combined_lob_training_y.npy"
# --- Đường dẫn đến file trạng thái đã lưu trong Dataset đầu vào ---
# Giả định file nằm trực tiếp hoặc trong thư mục con giống lúc lưu
INPUT_SCALER_PATH = OPTUNA_STATE_INPUT_DIR / "lob_input_scaler_optuna.pkl"
# Kiểm tra xem file có nằm trong thư mục con không
if not INPUT_SCALER_PATH.is_file():
    INPUT_SCALER_PATH = OPTUNA_STATE_INPUT_DIR / "trained_models_lob_optuna" / "lob_input_scaler_optuna.pkl"

INPUT_DB_PATH = OPTUNA_STATE_INPUT_DIR / "lob_optuna_study.db"
if not INPUT_DB_PATH.is_file():
     INPUT_DB_PATH = OPTUNA_STATE_INPUT_DIR / "trained_models_lob_optuna" / "lob_optuna_study.db"


# --- Training Hyperparameters ---
EPOCHS_OPTUNA_TRIAL = 35 # Giữ nguyên số epoch cho mỗi trial mới
EPOCHS_FINAL_TRAIN = 150
DEFAULT_BATCH_SIZE = 512
VALIDATION_SPLIT_RATIO = 0.15
EARLY_STOPPING_PATIENCE_FINAL = 15
NUM_CLASSES = 3
OPTUNA_N_TRIALS = 50 # Tổng số trials mong muốn cuối cùng
NUM_WORKERS = 0

# %% [code]
# --- Prediction Head ---
class PredictionHead(nn.Module):
     def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.0):
         super().__init__(); self.dropout = nn.Dropout(dropout_rate); self.fc = nn.Linear(input_dim, num_classes)
     def forward(self, backbone_output: torch.Tensor) -> torch.Tensor:
         return self.fc(self.dropout(backbone_output))

# %% [code]
# --- Optuna Objective Function (Giữ nguyên như phiên bản cuối cùng trước đó) ---
train_dataset_obj_global: Optional[TensorDataset] = None
val_dataset_obj_global: Optional[TensorDataset] = None

def objective(trial: optuna.trial.Trial) -> float:

    global train_dataset_obj_global, val_dataset_obj_global
    if train_dataset_obj_global is None or val_dataset_obj_global is None:
        logging.error("Global datasets not initialized!")
        raise RuntimeError("Datasets not available for Optuna trial.")
    lob_model_trial, prediction_head_trial, optimizer, current_train_loader, current_val_loader = None, None, None, None, None
    combined_params = None
    try:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8, 16]) # << DANH SÁCH CỐ ĐỊNH

        # >>> KIỂM TRA SỰ TƯƠNG THÍCH SAU KHI SUGGEST <<<
        if d_model % nhead != 0:
            # Nếu nhead không phải là ước số của d_model, bỏ qua trial này
            logging.info(f"Trial {trial.number}: Pruning because nhead={nhead} is not a divisor of d_model={d_model}.")
            raise optuna.exceptions.TrialPruned(f"Incompatible nhead={nhead} for d_model={d_model}")
        num_layers = trial.suggest_int("num_layers", 1, 4)
        batch_size_trial = trial.suggest_categorical("batch_size", [256, 512, 1024])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        backbone_dropout = trial.suggest_float("backbone_dropout", 0.0, 0.3)
        head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5)
        logging.debug(f"Trial {trial.number}: Creating DataLoaders (batch_size={batch_size_trial}, num_workers={NUM_WORKERS})...")
        current_train_loader = DataLoader(train_dataset_obj_global, batch_size=batch_size_trial, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        current_val_loader = DataLoader(val_dataset_obj_global, batch_size=batch_size_trial, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        logging.debug(f"Trial {trial.number}: DataLoaders created.")
        logging.debug(f"Trial {trial.number}: Initializing models...")
        lob_model_trial = LOBTransformer(input_dim=INPUT_DIM_EXPECTED, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=backbone_dropout).to(DEVICE)
        prediction_head_trial = PredictionHead(input_dim=d_model, num_classes=NUM_CLASSES, dropout_rate=head_dropout).to(DEVICE)
        logging.debug(f"Trial {trial.number}: Models on {DEVICE}.")
        combined_params = list(lob_model_trial.parameters()) + list(prediction_head_trial.parameters())
        optimizer = optim.AdamW(combined_params, lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_trial_val_loss = float('inf')
        logging.debug(f"Trial {trial.number}: Starting training loop for {EPOCHS_OPTUNA_TRIAL} epochs.")
        for epoch in range(EPOCHS_OPTUNA_TRIAL):
            epoch_start_time_trial = time.time()
            lob_model_trial.train(); prediction_head_trial.train()
            train_loss = 0.0; processed_batches_train = 0
            for batch_X, batch_y in current_train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                lob_features_output = lob_model_trial(batch_X)
                predictions = prediction_head_trial(lob_features_output)
                loss = criterion(predictions, batch_y)
                if not torch.isfinite(loss):
                    logging.warning(f"Trial {trial.number} E{epoch+1}: Non-finite train loss ({loss.item()}). Pruning.")
                    raise optuna.exceptions.TrialPruned("Non-finite training loss")
                loss.backward(); optimizer.step()
                train_loss += loss.item(); processed_batches_train += 1
            avg_train_loss = train_loss / processed_batches_train if processed_batches_train > 0 else 0.0
            lob_model_trial.eval(); prediction_head_trial.eval()
            val_loss = 0.0; processed_batches_val = 0
            with torch.no_grad():
                for batch_X, batch_y in current_val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    lob_features_output = lob_model_trial(batch_X)
                    predictions = prediction_head_trial(lob_features_output)
                    loss = criterion(predictions, batch_y)
                    if not torch.isfinite(loss):
                        logging.warning(f"Trial {trial.number} E{epoch+1}: Non-finite valid loss ({loss.item()}). Pruning.")
                        raise optuna.exceptions.TrialPruned("Non-finite validation loss")
                    val_loss += loss.item(); processed_batches_val +=1
            avg_val_loss = val_loss / processed_batches_val if processed_batches_val > 0 else float('inf')
            epoch_duration_trial = time.time() - epoch_start_time_trial
            logging.debug(f"Trial {trial.number} Epoch {epoch+1}/{EPOCHS_OPTUNA_TRIAL} [{epoch_duration_trial:.2f}s] -> T Loss: {avg_train_loss:.6f}, V Loss: {avg_val_loss:.6f}") # Gọn hơn
            best_trial_val_loss = min(best_trial_val_loss, avg_val_loss)
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                logging.info(f"Trial {trial.number}: Pruned at epoch {epoch+1} (val_loss={avg_val_loss:.6f}).")
                raise optuna.exceptions.TrialPruned()
        logging.debug(f"Trial {trial.number} finished. Best Val Loss in trial: {best_trial_val_loss:.6f}")
        return best_trial_val_loss
    except optuna.exceptions.TrialPruned as e_pruned: raise e_pruned
    except Exception as e: logging.error(f"Error in Optuna trial {trial.number}: {e}", exc_info=True); return float('inf')
    finally:
        del lob_model_trial, prediction_head_trial, optimizer, current_train_loader, current_val_loader, combined_params
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect(); logging.debug(f"Trial {trial.number}: Cleanup finished.")
    # === KẾT THÚC CODE OBJECTIVE ===

# %% [code]
# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    main_start_time = time.time()
    logging.info("=== Starting LOB Transformer Training (Continue Session) ===")

    # --- 1. Load Data (Giữ nguyên) ---
    logging.info(f"Loading LOB data from: {PROCESSED_DATA_DIR}")
    if not INPUT_X_PATH.is_file() or not INPUT_Y_PATH.is_file(): raise FileNotFoundError(f"Input data not found in {PROCESSED_DATA_DIR}")
    X_all = np.load(INPUT_X_PATH); y_all = np.load(INPUT_Y_PATH)
    logging.info(f"Loaded X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
    # Validation (Giữ nguyên)
    if X_all.ndim != 2 or X_all.shape[1] != INPUT_DIM_EXPECTED: raise ValueError("Incorrect X shape.")
    if y_all.ndim == 2 and y_all.shape[1] == 1: y_all = y_all.flatten()
    if y_all.ndim != 1: raise ValueError("Incorrect y shape.")
    if X_all.shape[0] != y_all.shape[0]: raise ValueError("Mismatched samples.")

    # --- 2. Data Splitting (Giữ nguyên) ---
    n_samples = X_all.shape[0]; val_size = int(n_samples * VALIDATION_SPLIT_RATIO); train_size = n_samples - val_size
    if train_size <= 50 or val_size <= 50: raise ValueError("Dataset too small.")
    indices = np.arange(n_samples); np.random.seed(42); np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    X_train, X_val = X_all[train_indices], X_all[val_indices]
    y_train, y_val = y_all[train_indices], y_all[val_indices]
    logging.info(f"Data split: Train={train_size}, Validation={val_size}")
    del X_all, y_all, indices, train_indices, val_indices; gc.collect()

    # --- 3. Load Scaler and Scale Data --- <<<<<<<<<<<<<<< THAY ĐỔI Ở ĐÂY
    logging.info(f"Attempting to load scaler from input dataset: {INPUT_SCALER_PATH}")
    if not INPUT_SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler file not found in input dataset: {INPUT_SCALER_PATH}. "
                              f"Ensure dataset '{OPTUNA_STATE_DATASET_SLUG}' is added and file exists.")
    try:
        scaler = joblib.load(INPUT_SCALER_PATH)
        logging.info("Scaler loaded successfully from input dataset.")
        # Apply the loaded scaler
        logging.info("Applying loaded scaler to train and validation data...")
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        logging.info("Data scaled using loaded scaler.")
        # Không cần lưu lại scaler ở đây nữa, vì nó đã có trong input
        # joblib.dump(scaler, WORKING_SCALER_PATH) # Bỏ qua hoặc comment out
    except Exception as e:
        logging.error(f"Error loading or applying scaler from {INPUT_SCALER_PATH}: {e}", exc_info=True); raise
    del X_train, X_val; gc.collect() # Xóa dữ liệu chưa scale

    # --- 4. Create Datasets (Giữ nguyên) ---
    logging.info("Creating TensorDatasets...")
    try:
        train_dataset_obj_global = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset_obj_global = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        logging.info(f"TensorDatasets created: Train={len(train_dataset_obj_global)}, Val={len(val_dataset_obj_global)}")
        del X_train_scaled, X_val_scaled, y_val; gc.collect()
    except Exception as e:
        logging.error(f"Error creating TensorDatasets: {e}", exc_info=True); raise

    # --- Pre-Optuna: Copy existing DB --- <<<<<<<<<<<<<<< THAY ĐỔI Ở ĐÂY
    logging.info(f"Checking for existing Optuna DB at input path: {INPUT_DB_PATH}")
    study_loaded_from_input = False
    if INPUT_DB_PATH.is_file():
        logging.info("Existing Optuna DB found in input dataset. Copying to working directory...")
        try:
            # Đảm bảo thư mục đích tồn tại trước khi copy
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(INPUT_DB_PATH, WORKING_DB_PATH_OBJ)
            logging.info(f"Successfully copied DB to: {WORKING_DB_PATH_OBJ}")
            study_loaded_from_input = True
        except Exception as e:
            logging.error(f"Failed to copy Optuna DB: {e}. Optuna will create a new study if {WORKING_DB_PATH_OBJ} doesn't exist.", exc_info=True)
            study_loaded_from_input = False
    else:
        logging.warning(f"Input DB not found at {INPUT_DB_PATH}. "
                      f"Ensure dataset '{OPTUNA_STATE_DATASET_SLUG}' is added and file exists. "
                      f"Optuna will create a new study if {WORKING_DB_PATH_OBJ} doesn't exist.")
        study_loaded_from_input = False

    # --- 5. Run Optuna Optimization (Tiếp tục hoặc Bắt đầu) --- <<<<<<<<<<<<<<< THAY ĐỔI Ở ĐÂY
    logging.info(f"Starting/Loading Optuna study... DB URI: {OPTUNA_DB_STORAGE_URI}")
    study = optuna.create_study(
        direction="minimize",
        study_name="lob_transformer_kaggle_opt", # Giữ study name nhất quán
        storage=OPTUNA_DB_STORAGE_URI, # Luôn dùng đường dẫn working
        load_if_exists=True # Load file đã copy hoặc tạo mới
    )

    # Log trạng thái
    current_trials_count = len(study.trials)
    if study_loaded_from_input and current_trials_count > 0:
         logging.info(f"Successfully loaded existing study with {current_trials_count} trials from copied DB.")
    elif not study_loaded_from_input and current_trials_count == 0 :
         logging.info("Created a new Optuna study (Input DB not found or copy failed).")
    # Các trường hợp khác ít xảy ra hơn nhưng để kiểm tra
    elif not study_loaded_from_input and current_trials_count > 0:
         logging.warning(f"DB not copied, but loaded existing study with {current_trials_count} trials from working dir (maybe previous run?).")
    elif study_loaded_from_input and current_trials_count == 0:
         logging.warning("DB copied, but loaded study has no trials? Check input DB file.")


    try:
        # Tính toán số trials còn lại cần chạy
        # Đếm số trials đã hoàn thành (có thể có trials đang chạy/lỗi)
        completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        target_total_trials = OPTUNA_N_TRIALS
        trials_to_run = target_total_trials - completed_trials_count

        if trials_to_run > 0:
             logging.info(f"Study has {completed_trials_count} COMPLETED trials. Running {trials_to_run} more trial(s) to reach target of {target_total_trials}.")
             # Chỉ chạy số trials còn thiếu
             study.optimize(objective, n_trials=trials_to_run, timeout=None, gc_after_trial=True)
             logging.info(f"Finished running additional trials. Total completed trials now: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        else:
             logging.info(f"Study already has {completed_trials_count} completed trials. Target of {target_total_trials} reached or exceeded. Skipping optimization phase.")

    except KeyboardInterrupt: logging.warning("Optuna optimization interrupted.")
    except Exception as optuna_err: logging.error(f"Optuna optimization failed: {optuna_err}", exc_info=True)

    # --- 6. Get Best Parameters and Train Final Model (Giữ nguyên) ---
    logging.info("Retrieving best parameters from Optuna study...")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
         logging.error(f"Optuna study '{study.study_name}' finished without any completed trials. Cannot proceed.")
         raise RuntimeError("No successful Optuna trials completed.")

    # Lấy best_trial và gán best_params ngay lập tức
    best_trial = study.best_trial
    best_params = best_trial.params # Gán giá trị cho best_params

    logging.info("\n--- Optuna Optimization Finished (Results) ---")
    logging.info(f"Best Trial Number: {best_trial.number}")
    logging.info(f"Best Value (Validation Loss): {best_trial.value:.6f}")
    logging.info("Best Parameters Found:")
    for key, value in best_params.items(): # Log các tham số tìm được
        logging.info(f"  {key}: {value}")
    logging.info("------------------------------------")

    # --- Re-initialize with Best Parameters for Final Training ---
    logging.info("Initializing FINAL model and head with best hyperparameters...")

    # Lấy các tham số kiến trúc từ best_params, có giá trị mặc định dự phòng
    final_d_model = best_params.get("d_model", D_MODEL_ORIG)

    # Đảm bảo nhead hợp lệ cho final_d_model (tái sử dụng logic)
    final_nhead_options = [h for h in [2, 4, 8, 16] if final_d_model % h == 0] # Nên giới hạn nhead hợp lý hơn: [2, 4, 8]
    if not final_nhead_options:
        logging.warning(f"Best d_model {final_d_model} has no standard divisors for nhead. Using nhead=1.")
        final_nhead = 1
    else:
        final_nhead = best_params.get("nhead", NHEAD_ORIG)
        if final_nhead not in final_nhead_options:
            logging.warning(f"Best nhead {final_nhead} from Optuna is not compatible with best d_model {final_d_model}. Selecting first valid option: {final_nhead_options[0]}")
            final_nhead = final_nhead_options[0]

    final_num_layers = best_params.get("num_layers", NUM_LAYERS_ORIG)

    # Lấy các tham số tối ưu hóa và dropout
    final_lr = best_params.get("lr", 1e-4) # Cung cấp giá trị mặc định hợp lý
    final_weight_decay = best_params.get("weight_decay", 1e-5) # Cung cấp giá trị mặc định hợp lý
    # Lấy dropout rates từ best_params, nếu không có (do lỗi suggest) thì dùng mặc định 0.1
    final_backbone_dropout = best_params.get("backbone_dropout", 0.1)
    final_head_dropout = best_params.get("head_dropout", 0.1)
    final_batch_size = best_params.get("batch_size", DEFAULT_BATCH_SIZE)

    # Khởi tạo final LOB Transformer backbone model với ĐẦY ĐỦ tham số tốt nhất
    logging.info(f"Initializing LOBTransformer with: d_model={final_d_model}, nhead={final_nhead}, num_layers={final_num_layers}, dropout={final_backbone_dropout:.4f}")
    final_model = LOBTransformer(
        input_dim=INPUT_DIM_EXPECTED, # Kích thước input cố định
        d_model=final_d_model,
        nhead=final_nhead,
        num_layers=final_num_layers,
        dropout=final_backbone_dropout # Truyền dropout tốt nhất
    ).to(DEVICE)

    # Khởi tạo final prediction head với ĐẦY ĐỦ tham số tốt nhất
    logging.info(f"Initializing PredictionHead with: input_dim={final_d_model}, num_classes={NUM_CLASSES}, dropout_rate={final_head_dropout:.4f}")
    final_prediction_head = PredictionHead(
        input_dim=final_d_model, # Input của head là output của backbone (d_model)
        num_classes=NUM_CLASSES, # Số lớp cố định
        dropout_rate=final_head_dropout # Truyền dropout tốt nhất
    ).to(DEVICE)

    # --- Phần còn lại: Optimizer, Scheduler, DataLoaders, Final Training Loop ---
    logging.info("Setting up final optimizer, scheduler, and dataloaders...")
    final_combined_params = list(final_model.parameters()) + list(final_prediction_head.parameters())
    final_optimizer = optim.AdamW(
        final_combined_params,
        lr=final_lr, # Dùng LR tốt nhất
        weight_decay=final_weight_decay # Dùng WD tốt nhất
    )
    final_criterion = nn.CrossEntropyLoss()
    final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        final_optimizer, mode='min', factor=0.5, patience=5
    )
    logging.info("Calculating class weights...")
    try:
        # Lấy các lớp duy nhất và số lượng mẫu của mỗi lớp trong tập train
        classes = np.unique(y_train) # Phải là [0, 1, 2]
        if not np.array_equal(classes, [0, 1, 2]):
            logging.warning(f"Unexpected classes found in y_train: {classes}. Assuming classes are 0, 1, 2.")
            classes = np.array([0, 1, 2]) # Đảm bảo có đủ 3 lớp

        # Sử dụng compute_class_weight của sklearn
        # 'balanced' tự động tính trọng số tỷ lệ nghịch với tần suất lớp
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

        # Chuyển thành tensor và đưa lên device
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        logging.info(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")

        # Sử dụng trọng số này trong hàm loss
        final_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        logging.info("Using CrossEntropyLoss with calculated class weights.")

    except Exception as e_cw:
        logging.error(f"Failed to calculate or apply class weights: {e_cw}. Using standard CrossEntropyLoss.", exc_info=True)
        # Fallback về hàm loss không trọng số nếu có lỗi
        final_criterion = nn.CrossEntropyLoss()

    logging.info(f"Creating final DataLoaders (batch_size={final_batch_size}, num_workers={NUM_WORKERS})")
    # Đảm bảo train_dataset_obj_global và val_dataset_obj_global đã được tạo trước đó
    final_train_loader = DataLoader(train_dataset_obj_global, batch_size=final_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    final_val_loader = DataLoader(val_dataset_obj_global, batch_size=final_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- 7. Final Training Loop ---
    logging.info(f"Starting FINAL training loop for up to {EPOCHS_FINAL_TRAIN} epochs with best parameters...")
    final_best_val_loss = float('inf')
    final_epochs_no_improve = 0
    final_training_start_time = time.time()

    for epoch in range(EPOCHS_FINAL_TRAIN):
        epoch_start_time = time.time()
        # --- Training Phase ---
        final_model.train()
        final_prediction_head.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        processed_batches_final_train = 0

        for batch_X, batch_y in final_train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            final_optimizer.zero_grad()

            # --- Forward Pass ---
            encoded = final_model.encoder(batch_X)
            encoded_for_transformer = encoded.unsqueeze(1)
            transformer_output_raw = final_model.transformer(encoded_for_transformer)
            transformer_features = transformer_output_raw.squeeze(1)
            lob_features_output = final_model.decoder(transformer_features)
            predictions = final_prediction_head(lob_features_output)
            # --- End Forward Pass ---

            loss = final_criterion(predictions, batch_y)
            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(final_combined_params, max_norm=1.0)
            final_optimizer.step()

            train_loss += loss.item()
            _, predicted_labels = torch.max(predictions.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted_labels == batch_y).sum().item()
            processed_batches_final_train += 1

        avg_train_loss = train_loss / processed_batches_final_train if processed_batches_final_train > 0 else 0.0
        train_acc = 100 * correct_train / total_train if total_train > 0 else 0.0

        # --- Validation Phase ---
        final_model.eval()
        final_prediction_head.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds_final = []
        all_val_targets_final = []
        processed_batches_final_val = 0

        with torch.no_grad():
            for batch_X, batch_y in final_val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                # --- Forward Pass ---
                encoded = final_model.encoder(batch_X)
                encoded_for_transformer = encoded.unsqueeze(1)
                transformer_output_raw = final_model.transformer(encoded_for_transformer)
                transformer_features = transformer_output_raw.squeeze(1)
                lob_features_output = final_model.decoder(transformer_features)
                predictions = final_prediction_head(lob_features_output)
                # --- End Forward Pass ---

                loss = final_criterion(predictions, batch_y)
                val_loss += loss.item()
                _, predicted_labels = torch.max(predictions.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted_labels == batch_y).sum().item()
                # Store predictions and targets for classification report
                all_val_preds_final.extend(predicted_labels.cpu().numpy())
                all_val_targets_final.extend(batch_y.cpu().numpy())
                processed_batches_final_val += 1

        avg_val_loss = val_loss / processed_batches_final_val if processed_batches_final_val > 0 else float('inf')
        val_acc = 100 * correct_val / total_val if total_val > 0 else 0.0
        epoch_time = time.time() - epoch_start_time

        logging.info(f"FINAL Train Epoch {epoch+1}/{EPOCHS_FINAL_TRAIN} [{epoch_time:.2f}s] - Train Loss: {avg_train_loss:.6f}, Acc: {train_acc:.2f}% "
                    f"| Val Loss: {avg_val_loss:.6f}, Acc: {val_acc:.2f}%")

        # --- Early Stopping & Save Best BACKBONE Model ---
        if avg_val_loss < final_best_val_loss:
            final_best_val_loss = avg_val_loss
            # *** SAVE ONLY THE BACKBONE (LOBTransformer) STATE DICT ***
            torch.save(final_model.state_dict(), FINAL_MODEL_SAVE_PATH)
            logging.info(f"✅ Best FINAL Model (Backbone) Saved - Epoch {epoch+1}, Val Loss: {final_best_val_loss:.6f} to {FINAL_MODEL_SAVE_PATH}")
            final_epochs_no_improve = 0
            # Print classification report for the best epoch
            try:
                # Ensure target names match your classes (0: Down, 1: Up, 2: Stationary - adjust if needed)
                report = classification_report(all_val_targets_final, all_val_preds_final, target_names=['Down', 'Up', 'Stationary'], zero_division=0)
                logging.info(f"FINAL Classification Report (Best Epoch {epoch+1}):\n{report}")
            except Exception as report_err:
                logging.warning(f"Could not generate final classification report for epoch {epoch+1}: {report_err}")
        else:
            final_epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {final_epochs_no_improve} epoch(s). Patience: {EARLY_STOPPING_PATIENCE_FINAL}.")
            if final_epochs_no_improve >= EARLY_STOPPING_PATIENCE_FINAL:
                logging.info(f"--- FINAL training early stopping triggered after {epoch+1} epochs. ---")
                break # Stop training

        # Step the scheduler based on validation loss
        final_scheduler.step(avg_val_loss)

    # --- End of Final Training ---
    final_training_time = time.time() - final_training_start_time
    logging.info(f"--- FINAL Training Finished in {final_training_time:.2f} seconds ---")
    logging.info(f"Best final validation loss achieved: {final_best_val_loss:.6f}")
    logging.info(f"FINAL Trained LOB Transformer backbone saved to: {FINAL_MODEL_SAVE_PATH}")
    logging.info(f"Input Scaler saved to: {SCALER_SAVE_PATH}")
    logging.info(f"Optuna study saved in: {OPTUNA_DB_PATH}")

    overall_total_time = time.time() - main_start_time
    logging.info(f"=== Overall Script Execution Time: {overall_total_time:.2f} seconds ===")
    print(f"=== Overall Script Execution Time: {overall_total_time:.2f} seconds ===")
    print(f"Log file saved to: {LOG_FILE_PATH}")
    print(f"Best model backbone saved to: {FINAL_MODEL_SAVE_PATH}")
    print(f"Scaler saved to: {SCALER_SAVE_PATH}")
    print(f"Optuna DB saved to: {OPTUNA_DB_PATH.replace('sqlite:///', str(MODEL_SAVE_DIR)+'/')}")