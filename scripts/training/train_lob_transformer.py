import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import joblib
import os
import sys # For sys.path manipulation
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from typing import Optional, Tuple, List, Dict, Any
import time
import optuna # Import Optuna
import gc

PROCESSED_DATA_DATASET_SLUG = "processed-lob-data"

KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working") # Output directory

PROCESSED_DATA_DIR = KAGGLE_INPUT_DIR / PROCESSED_DATA_DATASET_SLUG
INPUT_DIM_EXPECTED = 20 # Default/Expected input dim for LOB data
D_MODEL_ORIG = 64       # Keep original defaults if needed later
NHEAD_ORIG = 4
NUM_LAYERS_ORIG = 2

class LOBTransformer(nn.Module):
    """
    LOB Transformer Backbone - Phiên bản được tinh chỉnh cho training script.
    - Loại bỏ scaler_path, device, transform method không dùng trong training.
    - Thêm phương thức forward chuẩn cho nn.Module.
    - Thêm tham số dropout vào TransformerEncoderLayer.
    """
    # Thêm dropout vào signature với giá trị mặc định
    def __init__(self, input_dim=INPUT_DIM_EXPECTED, d_model=D_MODEL_ORIG, nhead=NHEAD_ORIG, num_layers=NUM_LAYERS_ORIG, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        # self.device không cần lưu trữ ở đây, script sẽ xử lý bên ngoài

        self.encoder = nn.Linear(input_dim, d_model)

        # Khởi tạo TransformerEncoderLayer với dropout
        # batch_first=True nghĩa là input/output có dạng (Batch, Sequence, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout, # Áp dụng dropout ở đây
            batch_first=True,
            # activation='relu', # Có thể chỉ định hàm kích hoạt nếu muốn
            # norm_first=False   # Hành vi mặc định của Pytorch
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder có thể là một lớp Linear đơn giản hoặc phức tạp hơn tùy thiết kế
        self.decoder = nn.Linear(d_model, d_model)

        # Không cần load scaler bên trong model khi training
        # self.scaler = None
        # logging.info("--- LOBTransformer (Training Version) Initialized ---") # Có thể log nếu muốn

        # Không cần gọi self.to(device) ở đây, script sẽ làm bên ngoài

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass chuẩn cho nn.Module, nhận vào batch tensor đã scale.
        Input `src` shape: (batch_size, input_dim)
        Output shape: (batch_size, d_model)
        """
        # 1. Encode features to d_model dimension
        # src shape: (B, input_dim) -> encoded shape: (B, d_model)
        encoded = self.encoder(src)

        # 2. Prepare for TransformerEncoder: Add sequence dimension (length 1)
        # encoded shape: (B, d_model) -> encoded_for_transformer shape: (B, 1, d_model)
        encoded_for_transformer = encoded.unsqueeze(1)

        # 3. Pass through Transformer Encoder layers
        # Input/Output shape for transformer: (B, 1, d_model)
        transformer_output_raw = self.transformer(encoded_for_transformer)

        # 4. Remove the sequence dimension
        # transformer_output_raw shape: (B, 1, d_model) -> transformer_features shape: (B, d_model)
        transformer_features = transformer_output_raw.squeeze(1)

        # 5. Pass through the final decoder layer
        # transformer_features shape: (B, d_model) -> lob_features_output shape: (B, d_model)
        lob_features_output = self.decoder(transformer_features)

        return lob_features_output
LOG_FILE_PATH = KAGGLE_WORKING_DIR / "train_lob_transformer_optuna.log"

# Reset logger handlers if running the cell multiple times
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout) # Log to notebook output as well
    ]
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")
logging.info(f"Input Data Directory: {PROCESSED_DATA_DIR}") # Still correct
logging.info(f"Output (Working) Directory: {KAGGLE_WORKING_DIR}") # Still correct

# --- Paths ---
MODEL_SAVE_DIR = KAGGLE_WORKING_DIR / "trained_models_lob_optuna" # Output subdir
MODEL_SAVE_DIR.mkdir(exist_ok=True)

INPUT_X_PATH = PROCESSED_DATA_DIR / "combined_lob_training_X.npy" # Still correct
INPUT_Y_PATH = PROCESSED_DATA_DIR / "combined_lob_training_y.npy" # Still correct

# Output files (will be saved in /kaggle/working/)
FINAL_MODEL_SAVE_PATH = MODEL_SAVE_DIR / "lob_transformer_backbone_optuna_final.pth"
SCALER_SAVE_PATH = MODEL_SAVE_DIR / "lob_input_scaler_optuna.pkl"
OPTUNA_DB_PATH = f"sqlite:///{MODEL_SAVE_DIR / 'lob_optuna_study.db'}" # Use Path object for consistency

# --- Training Hyperparameters ---
# These remain unchanged by the LOBTransformer definition location
EPOCHS_OPTUNA_TRIAL = 35
EPOCHS_FINAL_TRAIN = 150
DEFAULT_BATCH_SIZE = 512
VALIDATION_SPLIT_RATIO = 0.15
EARLY_STOPPING_PATIENCE_FINAL = 15
NUM_CLASSES = 3
OPTUNA_N_TRIALS = 50
NUM_WORKERS = 0

# --- Prediction Head (Temporary for Training) ---
class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, transformer_output: torch.Tensor) -> torch.Tensor:
        # Assuming transformer_output is the final feature representation (e.g., from the decoder)
        x = self.dropout(transformer_output)
        return self.fc(x)

# ==============================================================================
# Optuna Objective Function
# ==============================================================================

# Declare datasets globally within the main execution scope so objective can access them
# These will be assigned values later in the main script part.
train_dataset_obj_global: Optional[TensorDataset] = None
val_dataset_obj_global: Optional[TensorDataset] = None

def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function: trains and evaluates a set of hyperparameters."""
    global train_dataset_obj_global, val_dataset_obj_global # Access the global datasets

    if train_dataset_obj_global is None or val_dataset_obj_global is None:
        logging.error("Global datasets not initialized before calling objective function!")
        raise RuntimeError("Datasets not available for Optuna trial.")

    try:
        # --- 1. Suggest Hyperparameters ---
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        # nhead must be a divisor of d_model
        possible_nheads = [h for h in [2, 4, 8, 16] if d_model % h == 0]
        if not possible_nheads:
             # This case shouldn't happen with the suggested d_model values, but handle defensively
             logging.warning(f"No valid nhead found for d_model={d_model}. Pruning trial.")
             raise optuna.exceptions.TrialPruned(f"No valid nhead for d_model={d_model}")
        nhead = trial.suggest_categorical("nhead", possible_nheads)

        num_layers = trial.suggest_int("num_layers", 1, 4)
        batch_size_trial = trial.suggest_categorical("batch_size", [256, 512, 1024])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True) # Adjusted range slightly
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4) # Dropout for PredictionHead


        current_train_loader = DataLoader(train_dataset_obj_global, batch_size=batch_size_trial, shuffle=True, num_workers=2, pin_memory=True)
        current_val_loader = DataLoader(val_dataset_obj_global, batch_size=batch_size_trial, shuffle=False, num_workers=2, pin_memory=True)
        logging.debug(f"Trial {trial.number}: Using batch size {batch_size_trial}")


        # --- 2. Initialize Model, Head, Optimizer, Loss ---
        # Ensure the imported or dummy LOBTransformer is used correctly
        lob_model_trial = LOBTransformer(
            input_dim=INPUT_DIM_EXPECTED,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
            # dropout in TransformerEncoderLayer might need adjustment in LOBTransformer class itself
        ).to(DEVICE)

        prediction_head_trial = PredictionHead(
            input_dim=d_model, # Input is d_model from LOBTransformer's decoder output
            num_classes=NUM_CLASSES,
            dropout_rate=dropout_rate # Use suggested dropout rate
        ).to(DEVICE)

        # Combine parameters from both backbone and head
        combined_params = list(lob_model_trial.parameters()) + list(prediction_head_trial.parameters())
        optimizer = optim.AdamW(combined_params, lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        logging.info(f"Trial {trial.number}: Initializing DataLoaders (num_workers={NUM_WORKERS})...")
        current_train_loader = DataLoader(train_dataset_obj_global, batch_size=batch_size_trial, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        current_val_loader = DataLoader(val_dataset_obj_global, batch_size=batch_size_trial, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        logging.info(f"Trial {trial.number}: DataLoaders initialized.")

        logging.info(f"Trial {trial.number}: Getting train loader iterator...")
        try:
            train_iterator = iter(current_train_loader)
            logging.info(f"Trial {trial.number}: Got train loader iterator.")
        except Exception as e_iter:
            logging.error(f"Trial {trial.number}: FAILED to get train iterator: {e_iter}", exc_info=True)
            raise e_iter

        logging.info(f"Trial {trial.number}: Attempting to get first training batch...")
        try:
            first_batch_X, first_batch_y = next(train_iterator)
            logging.info(f"Trial {trial.number}: Successfully got first batch. Shape: {first_batch_X.shape}")
        except StopIteration:
            logging.error(f"Trial {trial.number}: DataLoader is unexpectedly empty!")
            raise RuntimeError("DataLoader empty")
        except Exception as e_next:
            logging.error(f"Trial {trial.number}: FAILED to get first batch: {e_next}", exc_info=True)
            raise e_next # Dừng trial

        logging.info(f"Trial {trial.number}: Moving first batch to {DEVICE}...")
        try:
            first_batch_X = first_batch_X.to(DEVICE)
            first_batch_y = first_batch_y.to(DEVICE)
            logging.info(f"Trial {trial.number}: First batch on device. Ready to start epoch loop.")
        except Exception as e_device:
            logging.error(f"Trial {trial.number}: FAILED to move first batch to device: {e_device}", exc_info=True)
            raise e_device

        # Xóa iterator và batch đã lấy để vòng lặp epoch bắt đầu lại từ đầu
        del train_iterator, first_batch_X, first_batch_y
        gc.collect()

        # --- 3. Training & Validation Loop (for the trial) ---
        best_trial_val_loss = float('inf')
        logging.info(f"Trial {trial.number}: Entering main training loop...")

        for epoch in range(EPOCHS_OPTUNA_TRIAL):
            epoch_start_time_trial = time.time()
            # Training phase
            lob_model_trial.train()
            prediction_head_trial.train()
            train_loss = 0.0
            processed_batches_train = 0
            for batch_X, batch_y in current_train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()

                # --- Forward Pass ---
                # Assuming LOBTransformer has encoder -> transformer -> decoder structure
                encoded = lob_model_trial.encoder(batch_X) # (batch, d_model)
                # TransformerEncoder expects (batch, seq_len, features) or (seq_len, batch, features)
                # Assuming input is treated as a sequence of length 1 for the transformer part
                encoded_for_transformer = encoded.unsqueeze(1) # (batch, 1, d_model) if batch_first=True
                transformer_output_raw = lob_model_trial.transformer(encoded_for_transformer) # (batch, 1, d_model)
                transformer_features = transformer_output_raw.squeeze(1) # (batch, d_model)
                lob_features_output = lob_model_trial.decoder(transformer_features) # (batch, d_model) - Input to head
                # --- End Forward Pass ---

                predictions = prediction_head_trial(lob_features_output) # (batch, num_classes)
                loss = criterion(predictions, batch_y)

                if not torch.isfinite(loss):
                    logging.warning(f"Trial {trial.number} Epoch {epoch+1}: Non-finite training loss ({loss.item()}). Pruning.")
                    raise optuna.exceptions.TrialPruned("Non-finite training loss")

                loss.backward()
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(combined_params, max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                processed_batches_train += 1
            # Avoid division by zero if loader is empty
            avg_train_loss = train_loss / processed_batches_train if processed_batches_train > 0 else 0.0

            # Validation phase
            lob_model_trial.eval()
            prediction_head_trial.eval()
            val_loss = 0.0
            processed_batches_val = 0
            with torch.no_grad():
                for batch_X, batch_y in current_val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                    # --- Forward Pass (same as training) ---
                    encoded = lob_model_trial.encoder(batch_X)
                    encoded_for_transformer = encoded.unsqueeze(1)
                    transformer_output_raw = lob_model_trial.transformer(encoded_for_transformer)
                    transformer_features = transformer_output_raw.squeeze(1)
                    lob_features_output = lob_model_trial.decoder(transformer_features)
                    # --- End Forward Pass ---

                    predictions = prediction_head_trial(lob_features_output)
                    loss = criterion(predictions, batch_y)

                    if not torch.isfinite(loss):
                        logging.warning(f"Trial {trial.number} Epoch {epoch+1}: Non-finite validation loss ({loss.item()}). Pruning.")
                        raise optuna.exceptions.TrialPruned("Non-finite validation loss")

                    val_loss += loss.item()
                    processed_batches_val +=1
            # Avoid division by zero if loader is empty
            avg_val_loss = val_loss / processed_batches_val if processed_batches_val > 0 else float('inf')

            epoch_duration_trial = time.time() - epoch_start_time_trial
            logging.debug(f"Trial {trial.number} Epoch {epoch+1}/{EPOCHS_OPTUNA_TRIAL} [{epoch_duration_trial:.2f}s] -> Val Loss: {avg_val_loss:.6f}")


            # Store the best validation loss encountered in this trial
            best_trial_val_loss = min(best_trial_val_loss, avg_val_loss)

            # --- Optuna Pruning ---
            trial.report(avg_val_loss, epoch) # Report the validation loss for this epoch
            if trial.should_prune():
                logging.info(f"Trial {trial.number}: Pruned at epoch {epoch+1} (val_loss={avg_val_loss:.6f}).")
                # Clean up before pruning
                del lob_model_trial, prediction_head_trial, optimizer, combined_params, current_train_loader, current_val_loader
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        # --- Trial Cleanup ---
        del lob_model_trial, prediction_head_trial, optimizer, combined_params, current_train_loader, current_val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.debug(f"Trial {trial.number} completed. Best Val Loss: {best_trial_val_loss:.6f}")
        # Return the best validation loss achieved during this trial
        # Optuna aims to minimize this value
        return best_trial_val_loss

    except optuna.exceptions.TrialPruned as e_pruned:
        # Ensure cleanup happens even if pruned mid-epoch
        if 'lob_model_trial' in locals(): del lob_model_trial
        if 'prediction_head_trial' in locals(): del prediction_head_trial
        if 'optimizer' in locals(): del optimizer
        if 'combined_params' in locals(): del combined_params
        if 'current_train_loader' in locals(): del current_train_loader
        if 'current_val_loader' in locals(): del current_val_loader
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise e_pruned # Re-raise the exception so Optuna knows it was pruned
    except Exception as e:
        logging.error(f"Error in Optuna trial {trial.number}: {e}", exc_info=True)
        # --- Ensure Cleanup on Error ---
        if 'lob_model_trial' in locals(): del lob_model_trial
        if 'prediction_head_trial' in locals(): del prediction_head_trial
        if 'optimizer' in locals(): del optimizer
        if 'combined_params' in locals(): del combined_params
        if 'current_train_loader' in locals(): del current_train_loader
        if 'current_val_loader' in locals(): del current_val_loader
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Return a high value to indicate failure, or re-raise if appropriate
        return float('inf')


# ==============================================================================
# Main Execution Block (Run Sequentially in Notebook)
# ==============================================================================

main_start_time = time.time()
logging.info("=== Starting LOB Transformer Training with Optuna Optimization (Kaggle Notebook) ===")

# --- 1. Load Prepared Data ---
logging.info(f"Loading prepared data from: {PROCESSED_DATA_DIR}")
if not INPUT_X_PATH.is_file() or not INPUT_Y_PATH.is_file():
    logging.error(f"Input data files not found! Checked paths:")
    logging.error(f"X: {INPUT_X_PATH}")
    logging.error(f"Y: {INPUT_Y_PATH}")
    logging.error(f"Please ensure dataset '{PROCESSED_DATA_DATASET_SLUG}' is attached and contains the .npy files.")
    # Stop execution in a notebook context if files are missing
    raise FileNotFoundError("Input data files not found in Kaggle dataset.")
else:
    try:
        X_all = np.load(INPUT_X_PATH)
        y_all = np.load(INPUT_Y_PATH)
        logging.info(f"Loaded X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
    except Exception as e:
        logging.error(f"Error loading .npy files: {e}", exc_info=True)
        raise # Re-raise the exception

# Basic data validation
if X_all.ndim != 2 or X_all.shape[1] != INPUT_DIM_EXPECTED:
    logging.error(f"X data shape error! Expected 2D array with {INPUT_DIM_EXPECTED} features, got {X_all.shape}.")
    raise ValueError("Incorrect X data shape.")
# Ensure y is 1D
if y_all.ndim == 2 and y_all.shape[1] == 1:
    y_all = y_all.flatten()
if y_all.ndim != 1:
    logging.error(f"y data shape error! Expected 1D array, got {y_all.ndim} dimensions.")
    raise ValueError("Incorrect y data shape.")
if X_all.shape[0] != y_all.shape[0]:
    logging.error(f"X and y have mismatched sample counts: {X_all.shape[0]} != {y_all.shape[0]}.")
    raise ValueError("Mismatched X and y samples.")

logging.info(f"Data loaded successfully. X: {X_all.shape}, y: {y_all.shape}")
logging.info(f"Number of samples: {X_all.shape[0]}")
logging.info(f"Feature dimension: {X_all.shape[1]}")

# --- 2. Data Splitting ---
n_samples = X_all.shape[0]
val_size = int(n_samples * VALIDATION_SPLIT_RATIO)
train_size = n_samples - val_size

if train_size <= 50 or val_size <= 50: # Basic sanity check
    logging.error(f"Data size too small for splitting. Train: {train_size}, Val: {val_size}")
    raise ValueError("Dataset too small.")

# Reproducible split
indices = np.arange(n_samples)
np.random.seed(42)
np.random.shuffle(indices)

train_indices, val_indices = indices[:train_size], indices[train_size:]
X_train, X_val = X_all[train_indices], X_all[val_indices]
y_train, y_val = y_all[train_indices], y_all[val_indices]

logging.info(f"Data split into Train: {X_train.shape}, {y_train.shape} and Validation: {X_val.shape}, {y_val.shape}")
# Clear large arrays if memory is tight (optional)
# del X_all, y_all
# import gc
# gc.collect()

# --- 3. Scale Input Features (Fit ONCE on Train data) ---
logging.info("Fitting StandardScaler on training data...")
scaler = StandardScaler()
try:
    X_train_scaled = scaler.fit_transform(X_train)
    # Scale validation data using the *same* scaler fitted on train data
    X_val_scaled = scaler.transform(X_val)
    # Save the scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logging.info(f"Input Scaler fitted and saved to {SCALER_SAVE_PATH}")
except Exception as e:
    logging.error(f"Error fitting/saving scaler: {e}", exc_info=True)
    raise # Re-raise the exception

# Clear original unscaled data if memory is tight
# del X_train, X_val
# gc.collect()

# --- 4. Create Datasets (Tensors) ---
# These datasets will be used by the objective function
logging.info("Creating TensorDatasets...")
try:
    # Assign to the global variables declared earlier
    train_dataset_obj_global = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset_obj_global = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    logging.info(f"TensorDatasets created. Train size: {len(train_dataset_obj_global)}, Val size: {len(val_dataset_obj_global)}")
except Exception as e:
    logging.error(f"Error creating TensorDatasets: {e}", exc_info=True)
    raise

# --- 5. Run Optuna Optimization ---
logging.info(f"Starting Optuna study ({OPTUNA_N_TRIALS} trials)... DB: {OPTUNA_DB_PATH}")
# Ensure the directory for the DB exists
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

study = optuna.create_study(
    direction="minimize", # We want to minimize validation loss
    study_name="lob_transformer_kaggle_opt",
    storage=OPTUNA_DB_PATH, # Path to the SQLite database file
    load_if_exists=True, # Load existing study results if available
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=10) # Prune unpromising trials early
)

# Log existing trials if study was loaded
if study.trials:
    logging.info(f"Loaded existing Optuna study with {len(study.trials)} previous trials.")

try:
    # Pass the objective function (which now uses global datasets)
    # Use gc_after_trial=True to potentially help with memory management on Kaggle
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=None, gc_after_trial=True) # No timeout unless needed
except KeyboardInterrupt:
    logging.warning("Optuna optimization interrupted by user.")
except Exception as optuna_err:
    logging.error(f"Optuna optimization failed: {optuna_err}", exc_info=True)
    # Still try to proceed if some trials completed

# --- 6. Get Best Parameters and Train Final Model ---
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if not completed_trials:
     logging.error("Optuna did not complete any trials successfully. Cannot proceed to final training.")
     # Stop execution
     raise RuntimeError("No successful Optuna trials completed.")

# Find the best trial among completed ones
best_trial = study.best_trial # Optuna automatically finds the best based on 'direction'
logging.info("\n--- Optuna Optimization Finished ---")
logging.info(f"Study Name: {study.study_name}")
logging.info(f"Number of finished trials: {len(study.trials)}")
logging.info(f"Best Trial Number: {best_trial.number}")
logging.info(f"Best Value (Validation Loss): {best_trial.value:.6f}") # best_trial.value is the return value of the objective
logging.info("Best Parameters:")
best_params = best_trial.params
for key, value in best_params.items():
    logging.info(f"  {key}: {value}")
logging.info("------------------------------------")

# --- Re-initialize with Best Parameters for Final Training ---
logging.info("Initializing FINAL model with best hyperparameters...")

final_d_model = best_params.get("d_model", D_MODEL_ORIG) # Use found or original default

# Ensure nhead is valid for the final d_model
final_nhead_options = [h for h in [2, 4, 8, 16] if final_d_model % h == 0]
if not final_nhead_options: # Fallback needed
    logging.warning(f"Best d_model {final_d_model} has no standard divisors for nhead. Using nhead=1.")
    final_nhead = 1
else:
    final_nhead = best_params.get("nhead", NHEAD_ORIG)
    # Check if the best_params['nhead'] is actually valid for best_params['d_model']
    if final_nhead not in final_nhead_options:
        logging.warning(f"Best nhead {final_nhead} from Optuna is not compatible with best d_model {final_d_model}. Selecting first valid option: {final_nhead_options[0]}")
        final_nhead = final_nhead_options[0] # Choose the smallest valid head count

final_num_layers = best_params.get("num_layers", NUM_LAYERS_ORIG)
final_lr = best_params.get("lr", 1e-4)
final_weight_decay = best_params.get("weight_decay", 1e-4)
final_dropout_rate = best_params.get("dropout_rate", 0.1) # Dropout for the head
final_batch_size = best_params.get("batch_size", DEFAULT_BATCH_SIZE)

# Initialize the final LOB Transformer backbone model
final_model = LOBTransformer(
    input_dim=INPUT_DIM_EXPECTED,
    d_model=final_d_model,
    nhead=final_nhead,
    num_layers=final_num_layers
).to(DEVICE)

# Initialize the final prediction head
final_prediction_head = PredictionHead(
    input_dim=final_d_model, # Input dimension matches backbone's output dim
    num_classes=NUM_CLASSES,
    dropout_rate=final_dropout_rate # Use the best dropout rate found
).to(DEVICE)

logging.info("Final Model Architecture:")
logging.info(f"  Input Dim: {INPUT_DIM_EXPECTED}")
logging.info(f"  d_model: {final_d_model}")
logging.info(f"  nhead: {final_nhead}")
logging.info(f"  num_layers: {final_num_layers}")
logging.info(f"  Head Dropout: {final_dropout_rate}")

# Optimizer and Loss for the final model
final_combined_params = list(final_model.parameters()) + list(final_prediction_head.parameters())
final_optimizer = optim.AdamW(
    final_combined_params,
    lr=final_lr,
    weight_decay=final_weight_decay
)
final_criterion = nn.CrossEntropyLoss()
# Learning rate scheduler for potential fine-tuning during final training
final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    final_optimizer,
    mode='min',       # Reduce LR when validation loss stops decreasing
    factor=0.5,       # Reduce LR by half
    patience=5,       # Wait 5 epochs with no improvement before reducing
    verbose=True
)

# --- Create Final DataLoaders with the best batch size ---
logging.info(f"Creating final DataLoaders with optimal batch size: {final_batch_size}")
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
