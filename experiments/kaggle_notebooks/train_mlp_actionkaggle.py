import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import logging
import os
import joblib
import argparse
import json
import platform
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple, Optional, Dict, Any
import warnings
from sklearn.base import BaseEstimator, RegressorMixin

# --- Lọc bỏ các warnings cụ thể ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message=".*The verbose parameter is deprecated.*")
#------------------------------------

# --- Cấu hình Logging ---
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Giữ INFO để xem các bước chính
for handler in logger.handlers[:]: logger.removeHandler(handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# --------------------------

# --- Định nghĩa Lớp MLPAction (Cập nhật Activation, Gradient Clipping) ---
class MLPAction(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[128, 64, 32], output_dim=1, dropout_p=0.2, activation_fn=nn.SiLU, device=None): # <<< Default mới
        super().__init__()
        self.input_dim = input_dim; self.hidden_dims = hidden_dims; self.output_dim = output_dim
        self.dropout_p = dropout_p; self.activation_fn = activation_fn # <<< Lưu activation
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logging.info(f"{self.__class__.__name__} Training: Device: {self.device}, Activation: {activation_fn.__name__}")
        self._build_network(); self.is_trained = False; self.best_params = None; self.to(self.device)
        try:
            if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2: self.network = torch.compile(self.network)
        except Exception as compile_e:
             logging.warning(f"torch.compile failed: {compile_e}")

    def _build_network(self):
        layers = []; in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            # <<< Sử dụng self.activation_fn >>>
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), self.activation_fn(), nn.Dropout(p=self.dropout_p)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.output_dim)); self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        elif x.ndim != 2 or x.shape[1] != self.input_dim: raise ValueError("Invalid input shape/dim")
        return self.network(x.to(self.device))

    # <<< Thêm clip_norm vào train_model >>>
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=1e-3, weight_decay=0.0, patience=5, loss_fn=None, optimizer_cls=None, scheduler_cls=None, num_data_workers=0, clip_norm=None):
        self.train(); train_losses, val_losses = [], []
        try: # Data prep
            X_train_t=torch.tensor(X_train,dtype=torch.float32); y_train_t=torch.tensor(y_train,dtype=torch.float32)
            X_val_t=torch.tensor(X_val,dtype=torch.float32); y_val_t=torch.tensor(y_val,dtype=torch.float32)
            if y_train_t.shape[-1] != self.output_dim: y_train_t = y_train_t.view(-1, self.output_dim)
            if y_val_t.shape[-1] != self.output_dim: y_val_t = y_val_t.view(-1, self.output_dim)
            train_dataset=TensorDataset(X_train_t,y_train_t); val_dataset=TensorDataset(X_val_t,y_val_t)
            train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))
            val_loader=DataLoader(val_dataset,batch_size=batch_size, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))
        except Exception as e: logging.error(f"Data prep error: {e}"); return None, None

        criterion=loss_fn if loss_fn else nn.MSELoss(); optimizer_class=optimizer_cls if optimizer_cls else optim.AdamW
        optimizer=optimizer_class(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = scheduler_cls(optimizer, patience=max(1, patience // 2), factor=0.5) if scheduler_cls else None

        best_val_loss=float('inf'); epochs_no_improve=0
        logging.info(f"Training started: {epochs} epochs, LR={learning_rate:.1E}, WD={weight_decay:.1E}, Batch={batch_size}, ClipNorm={clip_norm}, Workers={num_data_workers}...")
        for epoch in range(epochs):
            self.train(); running_loss=0.0
            for inputs,labels in train_loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(); outputs=self(inputs); loss=criterion(outputs,labels); loss.backward()
                # <<< THÊM GRADIENT CLIPPING >>>
                if clip_norm is not None and clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_norm)
                # <<< -------------------- >>>
                optimizer.step(); running_loss+=loss.item()
            avg_train_loss=running_loss/max(1, len(train_loader)); train_losses.append(avg_train_loss)
            self.eval(); val_loss=0.0
            with torch.no_grad():
                for inputs_val,labels_val in val_loader:
                     inputs_val, labels_val = inputs_val.to(self.device, non_blocking=True), labels_val.to(self.device, non_blocking=True)
                     outputs_val=self(inputs_val); loss=criterion(outputs_val,labels_val); val_loss+=loss.item()
            avg_val_loss=val_loss/max(1, len(val_loader)); val_losses.append(avg_val_loss)
            # <<< Giảm log để đỡ nhiễu hơn trong Optuna >>>
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1: # In mỗi 5 epochs và epoch cuối
                 logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            if scheduler: scheduler.step(avg_val_loss)
            if avg_val_loss<best_val_loss: best_val_loss=avg_val_loss; epochs_no_improve=0
            else:
                epochs_no_improve+=1
                if epochs_no_improve>=patience: logging.info(f"Early stop @ epoch {epoch+1}"); break
        self.is_trained=True; self.eval(); logging.info(f"Training done. Best Val Loss: {best_val_loss:.6f}")
        return train_losses, val_losses

    # <<< Cập nhật _objective để tối ưu activation, clip_norm >>>
    def _objective(self, trial, X_train, y_train, X_val, y_val, num_data_workers):
        # Model params
        n_layers=trial.suggest_int("n_layers", 2, 4); # Tập trung 2-4 lớp
        hidden_dims=[]
        last_h_dim = trial.suggest_int("n_units_l0", 32, 256, log=True) # Lớp đầu
        hidden_dims.append(last_h_dim)
        for i in range(1, n_layers):
            # Lớp sau nhỏ hơn hoặc bằng lớp trước
            next_h_dim = trial.suggest_int(f"n_units_l{i}", 16, last_h_dim, log=True)
            hidden_dims.append(next_h_dim)
            last_h_dim = next_h_dim

        dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5) # Tăng dropout min
        activation_name = trial.suggest_categorical("activation", ["ReLU", "SiLU", "GELU"]) # Thêm GELU

        # Optimizer params
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        optimizer_name=trial.suggest_categorical("optimizer", ["AdamW", "Adam", "RMSprop"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if optimizer_name=="AdamW" else 0.0
        clip_norm = trial.suggest_float("clip_norm", 0.1, 1.0) # Tối ưu clip_norm

        # Loss and training params
        loss_type = trial.suggest_categorical("loss", ["mse", "huber", "smoothl1"])
        batch_size=trial.suggest_categorical("batch_size",[32, 64, 128])
        epochs=trial.suggest_int("epochs", 50, 120) # Tăng max epochs chút

        # Chọn activation function
        activation_cls = getattr(nn, activation_name) if hasattr(nn, activation_name) else nn.SiLU # Default SiLU

        # Tạo model tạm với params mới
        temp_model=MLPAction(input_dim=self.input_dim, hidden_dims=hidden_dims, output_dim=self.output_dim,
                             dropout_p=dropout_p, activation_fn=activation_cls, device=self.device) # <<< Truyền activation
        temp_model.to(self.device)

        try: # Data prep
            X_train_t=torch.tensor(X_train,dtype=torch.float32); y_train_t=torch.tensor(y_train,dtype=torch.float32)
            X_val_t=torch.tensor(X_val,dtype=torch.float32); y_val_t=torch.tensor(y_val,dtype=torch.float32)
            if y_train_t.shape[-1]!=self.output_dim: y_train_t=y_train_t.view(-1,self.output_dim)
            if y_val_t.shape[-1]!=self.output_dim: y_val_t=y_val_t.view(-1,self.output_dim)
            train_dataset=TensorDataset(X_train_t,y_train_t); val_dataset=TensorDataset(X_val_t,y_val_t)
            train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))
            val_loader=DataLoader(val_dataset,batch_size=batch_size, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))

            # Chọn Loss Function
            if loss_type == "huber": criterion = nn.HuberLoss()
            elif loss_type == "smoothl1": criterion = nn.SmoothL1Loss()
            else: criterion = nn.MSELoss()

            optimizer_cls=getattr(optim,optimizer_name)
            optimizer=optimizer_cls(temp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False) # Tăng patience scheduler chút

            best_trial_val_loss=float('inf'); epochs_no_improve=0; patience=10 # Tăng patience early stopping

            for epoch in range(epochs): # Train loop
                temp_model.train(); running_loss = 0.0
                for inputs,labels in train_loader:
                    inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    optimizer.zero_grad(); outputs=temp_model(inputs); loss=criterion(outputs,labels); loss.backward()
                    # <<< Áp dụng clipping trong trial >>>
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=clip_norm)
                    optimizer.step(); running_loss += loss.item()

                temp_model.eval(); current_val_loss=0.0
                with torch.no_grad():
                    for inputs_val,labels_val in val_loader:
                         inputs_val, labels_val = inputs_val.to(self.device, non_blocking=True), labels_val.to(self.device, non_blocking=True)
                         outputs_val=temp_model(inputs_val); loss=criterion(outputs_val,labels_val); current_val_loss+=loss.item()
                avg_val_loss=current_val_loss / max(1, len(val_loader))
                scheduler.step(avg_val_loss)
                if avg_val_loss<best_trial_val_loss: best_trial_val_loss=avg_val_loss; epochs_no_improve=0
                else:
                    epochs_no_improve+=1
                    if epochs_no_improve>=patience: break
                trial.report(avg_val_loss,epoch)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            return best_trial_val_loss
        except optuna.exceptions.TrialPruned: logging.debug("Trial pruned."); return float('inf')
        except Exception as e: logging.error(f"Optuna trial error: {e}", exc_info=True); return float('inf') # Thêm exc_info

    # optimize_hyperparameters: Cập nhật để sử dụng activation và clip_norm từ best_params
    def optimize_hyperparameters(self, X, y, n_trials=50, test_size=0.2, val_size=0.2, num_data_workers=0):
        if X is None or y is None or len(X)==0: logging.error("Empty data for optimization."); return
        logging.info(f"Optimizing hyperparameters (TPESampler, {n_trials} trials)...")
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
            relative_val_size=val_size/(1.0-test_size) if (1.0-test_size) > 0 else val_size
            X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train_val,y_train_val,test_size=relative_val_size,random_state=42)
            logging.info(f"Data split: Train_opt={len(X_train_opt)}, Val_opt={len(X_val_opt)}, Test={len(X_test)}")

            study=optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10))
            objective_func=lambda trial: self._objective(trial,X_train_opt,y_train_opt,X_val_opt,y_val_opt, num_data_workers)
            study.optimize(objective_func, n_trials=n_trials, n_jobs=1)

            # <<< Sửa phần lấy tham số mặc định >>>
            default_params = { 'n_layers': 3, 'n_units_l0': 128, 'n_units_l1': 64, 'n_units_l2': 32,
                               'lr': 1e-3, 'optimizer': 'AdamW', 'batch_size': 64, 'epochs': 80,
                               'dropout_p': 0.2, 'weight_decay': 1e-5, 'loss': 'huber',
                               'activation': 'SiLU', 'clip_norm': 0.5 }
            if not study.best_trial:
                logging.warning("Optuna found no valid trials. Using default parameters.")
                self.best_params = default_params
            else:
                self.best_params = study.best_params
                logging.info(f"Best trial: Value={study.best_value:.6f}, Params={self.best_params}")
                # Đảm bảo các key mặc định tồn tại nếu trial không tối ưu chúng
                for key, value in default_params.items():
                     self.best_params.setdefault(key, value)


            # Lấy các tham số tốt nhất
            best_hidden_dims=[self.best_params[f"n_units_l{i}"] for i in range(self.best_params["n_layers"])]
            best_lr=self.best_params["lr"]; best_optimizer=self.best_params["optimizer"]
            best_batch_size=self.best_params["batch_size"]; best_epochs=self.best_params.get("epochs",80)
            best_dropout_p = self.best_params.get("dropout_p", 0.2)
            best_weight_decay = self.best_params.get("weight_decay", 0.0) if best_optimizer=="AdamW" else 0.0
            best_loss_type = self.best_params.get("loss", "huber")
            best_activation_name = self.best_params.get("activation", "SiLU") # Lấy activation
            best_clip_norm = self.best_params.get("clip_norm", 0.5) # Lấy clip_norm

            # Chọn activation class
            best_activation_cls = getattr(nn, best_activation_name) if hasattr(nn, best_activation_name) else nn.SiLU

            # Rebuild nếu cần (thêm kiểm tra activation)
            if self.hidden_dims!=best_hidden_dims or self.dropout_p != best_dropout_p or self.activation_fn != best_activation_cls:
                logging.info(f"Rebuilding model: hidden={best_hidden_dims}, dropout={best_dropout_p}, activation={best_activation_name}")
                self.hidden_dims=best_hidden_dims; self.dropout_p = best_dropout_p; self.activation_fn = best_activation_cls
                self._build_network(); self.to(self.device)
                try:
                     if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2: self.network = torch.compile(self.network)
                except Exception as compile_e: logging.warning(f"Re-compile failed: {compile_e}")

            logging.info("Retraining main model with best params...")
            # Chọn Loss và Scheduler cho lần train cuối
            if best_loss_type == "huber": final_criterion = nn.HuberLoss()
            elif best_loss_type == "smoothl1": final_criterion = nn.SmoothL1Loss()
            else: final_criterion = nn.MSELoss()
            final_scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau

            # Huấn luyện trên Train + Validation, truyền cả clip_norm
            train_losses, val_losses = self.train_model(
                X_train_val,y_train_val,X_test,y_test, # Dùng Test làm val set cuối
                epochs=best_epochs, batch_size=best_batch_size, learning_rate=best_lr, weight_decay=best_weight_decay,
                patience=15, # Tăng patience cho train cuối
                loss_fn=final_criterion, optimizer_cls=getattr(optim,best_optimizer),
                scheduler_cls=final_scheduler_cls,
                num_data_workers=num_data_workers,
                clip_norm=best_clip_norm # <<< Truyền clip_norm
            )
            logging.info("Hyperparameter optimization and final model training complete.")
            return train_losses, val_losses, X_test, y_test

        except Exception as e: logging.error(f"Hyperparameter optimization error: {e}", exc_info=True); return None, None, None, None

    # save_model, load_model, predict giữ nguyên như trước
    def save_model(self, file_path="mlp_action_model.pth", fallback_pred=0.0): # Nhận fallback từ ngoài
        try:
            save_data={'state_dict':self.state_dict(),'input_dim':self.input_dim,
                       'hidden_dims':self.hidden_dims,'output_dim':self.output_dim,
                       'dropout_p': self.dropout_p,
                       'activation_name': self.activation_fn.__name__, # <<< Lưu tên activation
                       'best_params':self.best_params,
                       'fallback_prediction': fallback_pred } # <<< Lưu fallback đã tính
            torch.save(save_data,file_path); logging.info(f"Model saved to {file_path} (Fallback: {fallback_pred:.4f})")
        except Exception as e: logging.error(f"Model saving error: {e}")

    def load_model(self, file_path="mlp_action_model.pth"):
        if not os.path.exists(file_path):
            logging.error(f"{self.__class__.__name__}: Model file not found: {file_path}.")
            self.is_trained = False; return False, None # Trả về thêm fallback value
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
                 logging.error(f"{self.__class__.__name__}: Invalid checkpoint format: {file_path}.")
                 self.is_trained = False; return False, None

            loaded_input_dim = checkpoint.get('input_dim')
            loaded_hidden_dims = checkpoint.get('hidden_dims')
            loaded_output_dim = checkpoint.get('output_dim')
            loaded_dropout_p = checkpoint.get('dropout_p', self.dropout_p)
            loaded_activation_name = checkpoint.get('activation_name', 'SiLU') # <<< Load activation name
            loaded_fallback_value = checkpoint.get('fallback_prediction', 0.0)

            if loaded_input_dim != self.input_dim or loaded_output_dim != self.output_dim:
                 logging.error(f"{self.__class__.__name__}: Input/Output dim mismatch!"); self.is_trained = False; return False, None

            # Chọn activation class
            loaded_activation_cls = getattr(nn, loaded_activation_name) if hasattr(nn, loaded_activation_name) else nn.SiLU

            # Rebuild nếu cần (thêm kiểm tra activation)
            if loaded_hidden_dims != self.hidden_dims or loaded_dropout_p != self.dropout_p or self.activation_fn != loaded_activation_cls:
                 logging.info(f"{self.__class__.__name__}: Rebuilding network. hidden={loaded_hidden_dims}, dropout={loaded_dropout_p}, activation={loaded_activation_name}")
                 self.hidden_dims = loaded_hidden_dims; self.dropout_p = loaded_dropout_p; self.activation_fn = loaded_activation_cls
                 self._build_network(); self.to(self.device)
                 try: # Re-compile
                      if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2: self.network = torch.compile(self.network, mode="reduce-overhead")
                 except Exception: pass

            self.load_state_dict(checkpoint['state_dict'])
            self.is_trained = True
            self.eval()
            logging.info(f"{self.__class__.__name__} model loaded from {file_path}")
            return True, loaded_fallback_value

        except Exception as e:
            logging.error(f"Error loading {self.__class__.__name__} model: {e}", exc_info=True)
            self.is_trained = False; return False, None

    @torch.no_grad()
    def predict(self, x: np.ndarray, fallback_value: float = 0.0) -> np.ndarray:
        # (Giữ nguyên logic predict)
        default_output_shape = (x.shape[0], self.output_dim) if x.ndim == 2 else (self.output_dim,)
        if not self.is_trained:
            # logging.warning(f"{self.__class__.__name__}: Predict on untrained model. Returning fallback: {fallback_value}") # Giảm log
            return np.full(default_output_shape, fallback_value, dtype=np.float32)
        self.eval()
        try:
            x_tensor = torch.tensor(x, dtype=torch.float32)
            predictions_tensor = self(x_tensor)
            result = predictions_tensor.cpu().numpy()
            if not np.all(np.isfinite(result)):
                 logging.warning(f"{self.__class__.__name__}: Prediction resulted in NaN/Inf. Returning fallback: {fallback_value}")
                 return np.full(default_output_shape, fallback_value, dtype=np.float32)
            return result
        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} prediction: {e}", exc_info=True)
            logging.warning(f"{self.__class__.__name__}: Prediction failed. Returning fallback value: {fallback_value}")
            return np.full(default_output_shape, fallback_value, dtype=np.float32)


# --- Hàm Tải và Chuẩn bị Dữ liệu (Cập nhật Features và Target) ---
def load_and_prepare_data(
    data_file_path: str,
    timeframe: str = '15m',
    future_periods: int = 5,
    input_features_override: Optional[List[str]] = None,
    atr_proxy_factor: float = 0.05,
    target_type: str = 'percent_change', # Thêm lựa chọn mới
    use_cache: bool = True,
    use_hl_proxy_feature: bool = False, # Giữ lại tùy chọn này
    output_dir: str = "."
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[StandardScaler], Optional[float]]:
    """
    Tải dữ liệu, chuẩn bị features (đã cập nhật) và target (có lựa chọn mới).
    """
    # --- Tạo Cache Path ---
    hl_suffix = "_hlproxy" if use_hl_proxy_feature else ""
    # <<< Cập nhật tên cache để phản ánh target_type >>>
    target_suffix = f"_target{target_type}_fut{future_periods}"
    proxy_suffix = f"_proxy{atr_proxy_factor:.3f}".replace(".","p")
    cache_path = None
    if output_dir:
        try:
            cache_file_name = os.path.basename(data_file_path).replace(".pkl", f"_{timeframe}{target_suffix}{proxy_suffix}{hl_suffix}_mlp_preprocessed.joblib")
            cache_subdir = os.path.join(output_dir, "cache_mlp")
            os.makedirs(cache_subdir, exist_ok=True)
            cache_path = os.path.join(cache_subdir, cache_file_name)
        except Exception as path_e:
            logging.warning(f"Error creating cache path: {path_e}. Disabling caching.")
            use_cache = False

    # --- Khởi tạo ---
    X_raw, y, feature_names, fallback_pred = None, None, None, None
    data_dict = None

    # --- Tải từ Cache ---
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            logging.info(f"Loading preprocessed data from cache: {cache_path}")
            X_raw_cached, y_cached, feature_names_cached, fallback_pred_cached = joblib.load(cache_path)
            # Kiểm tra cache (như cũ)
            if isinstance(X_raw_cached, np.ndarray) and isinstance(y_cached, np.ndarray) and \
               isinstance(feature_names_cached, list) and \
               isinstance(fallback_pred_cached, (float, np.float32, np.float64)) and \
               X_raw_cached.ndim == 2 and y_cached.ndim == 2 and y_cached.shape[1] == 1 and \
               len(feature_names_cached) == X_raw_cached.shape[1]:
                 logging.info(f"Loaded RAW data from cache: X_raw={X_raw_cached.shape}, y={y_cached.shape}, Fallback={fallback_pred_cached:.4f}")
                 X_raw, y, feature_names, fallback_pred = X_raw_cached, y_cached, feature_names_cached, fallback_pred_cached
                 return X_raw, y, feature_names, None, fallback_pred
            else: logging.warning("Invalid cache data. Re-processing.")
        except Exception as cache_load_e: logging.warning(f"Failed to load cache: {cache_load_e}. Re-processing.")

    # --- Tải và Xử lý từ File gốc ---
    logging.info(f"Processing data from file: {data_file_path}, timeframe: {timeframe}")
    try:
        # Tải file (joblib/pickle)
        if not os.path.exists(data_file_path): # ... xử lý lỗi file not found ...
             return None, None, None, None, None
        data_dict = None
        try: data_dict = joblib.load(data_file_path); logging.info(f"Loaded with joblib: {data_file_path}")
        except Exception:
            try:
                with open(data_file_path, 'rb') as f: data_dict = pickle.load(f)
                logging.info(f"Loaded with pickle: {data_file_path}")
            except Exception as e: logging.error(f"Failed load {data_file_path}: {e}"); return None, None, None, None, None
        # Kiểm tra dict, timeframe, df
        if not isinstance(data_dict, dict): # ... xử lý lỗi ...
             return None, None, None, None, None
        if timeframe not in data_dict: # ... xử lý lỗi ...
             return None, None, None, None, None
        df_original = data_dict[timeframe]
        if not isinstance(df_original, pd.DataFrame) or df_original.empty: # ... xử lý lỗi ...
             return None, None, None, None, None
        logging.debug(f"DataFrame loaded. Shape: {df_original.shape}")
        df = df_original.copy()

        # --- Xác định và Tính toán Features ---
        if input_features_override:
            feature_names = input_features_override
        else:
            # <<< BỘ FEATURES ĐÃ LOẠI BỎ NHIỄU/TƯƠNG QUAN >>>
            feature_names = [
                "RSI",             # Giữ lại
                "ATR",             # Giữ lại
                "EMA_diff",        # Feature mới (tính bên dưới)
                "MACD",            # Giữ lại
                "MACD_signal",     # Giữ lại
                "ADX",             # Giữ lại
                "momentum",        # Giữ lại (có thể thử bỏ sau)
                "log_volume",      # Feature mới (tính bên dưới)
                # "OBV",             # Bỏ (thường nhiễu và tương quan với volume)
                "VWAP",            # Giữ lại
                # "order_imbalance", # Bỏ (importance thấp)
                "BB_width",        # Feature mới (tính bên dưới)
                "VAH",             # Giữ lại (quan trọng nếu ổn định)
                # "divergence",      # Bỏ (importance thấp)
                # "rejection",       # Bỏ (importance thấp)
                "regime",          # Giữ lại
                "hour",            # Giữ lại
                # === Features mới thêm ===
                "VWAP_ADX_interaction", # Tính bên dưới
                "BB_EMA_sync",          # Tính bên dưới
                "volume_anomaly",       # Tính bên dưới
            ]
            # Thêm hl_proxy nếu được yêu cầu
            if use_hl_proxy_feature:
                feature_names.append("hl_proxy")

        # --- Tính toán các features phái sinh cần thiết ---
        calculation_errors = []
        required_base_cols = set(['close', 'high', 'low', 'volume', 'ATR', 'RSI', # Base cho tính toán
                                 'EMA_50', 'EMA_200', 'MACD', 'MACD_signal', 'ADX', 'momentum',
                                 'OBV', 'VWAP', 'BB_upper', 'BB_lower', 'BB_middle',
                                 'VAH', 'divergence', 'rejection', 'regime', 'hour'])
        # Kiểm tra các cột cơ sở cần thiết cho features *dự định* tính
        base_cols_needed = set()
        if 'EMA_diff' in feature_names: base_cols_needed.update(['EMA_50', 'EMA_200', 'close'])
        if 'log_volume' in feature_names or 'volume_anomaly' in feature_names: base_cols_needed.add('volume')
        if 'BB_width' in feature_names or 'BB_EMA_sync' in feature_names: base_cols_needed.update(['BB_upper', 'BB_lower', 'BB_middle'])
        if 'VWAP_ADX_interaction' in feature_names: base_cols_needed.update(['VWAP', 'ADX'])
        if use_hl_proxy_feature and 'hl_proxy' in feature_names: base_cols_needed.update(['high', 'low'])
        # Luôn cần ATR nếu tính proxy
        if 'bid_ask_spread_proxy' in feature_names: base_cols_needed.add('ATR')
        # Thêm các features gốc trong list feature_names
        base_cols_needed.update([f for f in feature_names if f in required_base_cols])

        missing_base = list(base_cols_needed - set(df.columns))
        if missing_base:
            logging.error(f"Missing base columns needed for feature calculation: {missing_base}")
            return None, None, None, None, None

        # Bắt đầu tính toán
        close_safe = df['close'].replace(0, np.nan)
        if 'EMA_diff' in feature_names:
            try: df['EMA_diff'] = (df['EMA_50'] - df['EMA_200']) / close_safe
            except Exception as e: calculation_errors.append(f"EMA_diff ({e})")
        if 'log_volume' in feature_names:
            try: df['log_volume'] = np.log1p(df['volume'].clip(lower=0))
            except Exception as e: calculation_errors.append(f"log_volume ({e})")
        if 'BB_width' in feature_names:
            try:
                 bb_middle_safe = df['BB_middle'].replace(0, np.nan)
                 df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_middle_safe
            except Exception as e: calculation_errors.append(f"BB_width ({e})")
        if 'VWAP_ADX_interaction' in feature_names:
            try: df['VWAP_ADX_interaction'] = df['VWAP'] * df['ADX'] / 100
            except Exception as e: calculation_errors.append(f"VWAP_ADX_interaction ({e})")
        if 'BB_EMA_sync' in feature_names:
            # Cần EMA_diff và BB_width đã được tính
            if 'EMA_diff' in df.columns and 'BB_width' in df.columns:
                 try: df['BB_EMA_sync'] = df['BB_width'] * df['EMA_diff']
                 except Exception as e: calculation_errors.append(f"BB_EMA_sync ({e})")
            else: calculation_errors.append("BB_EMA_sync (dependency missing)")
        if 'volume_anomaly' in feature_names:
            try:
                 vol_anomaly_window = 24*3
                 log_vol_mean = df['log_volume'].rolling(window=vol_anomaly_window).mean()
                 log_vol_std = df['log_volume'].rolling(window=vol_anomaly_window).std()
                 df['volume_anomaly'] = (df['log_volume'] > log_vol_mean + 2*log_vol_std).astype(int)
            except Exception as e: calculation_errors.append(f"volume_anomaly ({e})")
        if use_hl_proxy_feature and 'hl_proxy' in feature_names:
             try: df['hl_proxy'] = (df['high'] - df['low']).clip(lower=1e-9) * 0.1
             except Exception as e: calculation_errors.append(f"hl_proxy ({e})")
        # Tính bid_ask_spread_proxy (nếu vẫn muốn giữ)
        # if 'bid_ask_spread_proxy' in feature_names:
        #     try:
        #         atr_safe = df['ATR'].ffill().bfill().fillna(1e-9).clip(lower=1e-9)
        #         df['bid_ask_spread_proxy'] = atr_safe * atr_proxy_factor
        #     except Exception as e: calculation_errors.append(f"bid_ask_spread_proxy ({e})")

        # Xử lý lỗi và cập nhật final_feature_names
        final_feature_names = list(feature_names)
        if calculation_errors:
            logging.warning(f"Could not calculate some features for {data_file_path}:")
            for err in calculation_errors:
                 logging.warning(f"- {err}")
                 feature_name_to_remove = err.split(" ")[0]
                 if feature_name_to_remove in final_feature_names:
                      final_feature_names.remove(feature_name_to_remove)
            logging.warning(f"Using features: {final_feature_names}")
            if not final_feature_names: return None, None, None, None, None

        # Kiểm tra cuối cùng
        missing_final = [col for col in final_feature_names if col not in df.columns]
        if missing_final:
            logging.error(f"Features still missing after calculation: {missing_final}")
            return None, None, None, None, None

        logging.info(f"Final features used: {final_feature_names} (Dim: {len(final_feature_names)})")

        # --- Tính toán Target (Thêm Volatility-Normalized) ---
        df['future_close'] = df['close'].shift(-future_periods)
        if target_type == 'vol_normalized_returns': # <<< Lựa chọn target mới
            logging.info(f"Using Volatility-Normalized target (periods={future_periods}). Calculating volatility...")
            # <<< Tính Volatility cho target (có thể dùng window khác) >>>
            vol_window_target = 24 # Ví dụ: 1 ngày
            # Dùng pct_change().rolling().std()
            # ffill/bfill để xử lý NaN đầu chuỗi rolling vol
            volatility_target = df['close'].pct_change().rolling(window=vol_window_target, min_periods=vol_window_target//2).std().ffill().bfill().fillna(1e-9).clip(lower=1e-9)
            # Tính future return (không nhân 100)
            future_return = df['future_close'] / close_safe - 1
            df['target'] = future_return / volatility_target
            logging.info("Finished calculating Volatility-Normalized target.")
        else: # Mặc định 'percent_change'
            df['target'] = (df['future_close'] / close_safe - 1) * 100
            logging.info(f"Using Percent Change target (periods={future_periods}).")


        # --- Làm sạch dữ liệu cuối cùng (Sử dụng ffill/bfill) ---
        required_cols_final = final_feature_names + ['target']
        df_subset = df[required_cols_final]
        df_target_valid = df_subset.dropna(subset=['target'])
        logging.debug(f"Shape after dropping NaN targets: {df_target_valid.shape}")
        df_features_only = df_target_valid[final_feature_names]
        df_features_filled = df_features_only.ffill().bfill() # <<< Fill NaN features
        rows_with_remaining_nan_mask = df_features_filled.isnull().any(axis=1)
        num_rows_remaining_nan = rows_with_remaining_nan_mask.sum()
        if num_rows_remaining_nan > 0:
            logging.warning(f"{num_rows_remaining_nan} rows dropped due to remaining NaN features after ffill/bfill.")
        # Kết hợp lại và dropna cuối cùng
        df_clean = pd.concat([df_features_filled, df_target_valid.loc[df_features_filled.index, 'target']], axis=1)
        df_clean = df_clean.loc[~rows_with_remaining_nan_mask]
        logging.info(f"Shape after final cleaning: {df_clean.shape}")

        # Kiểm tra số mẫu tối thiểu
        min_samples_threshold = 100
        if len(df_clean) < min_samples_threshold: # ... xử lý lỗi ...
             return None, None, None, None, None

        # Trích xuất X_raw, y
        X_raw = df_clean[final_feature_names].values.astype(np.float32)
        y = df_clean['target'].values.astype(np.float32)
        if y.ndim == 1: y = y.reshape(-1, 1)

        # --- Tính Fallback Prediction ---
        fallback_pred = 0.0
        if len(X_raw) > 20:
             try:
                  y_train_fb = y[:int(len(X_raw) * 0.2)]
                  if len(y_train_fb) > 0: fallback_pred = float(np.nanmedian(y_train_fb))
             except Exception as fb_calc_e: logging.warning(f"Fallback calc error: {fb_calc_e}")
        logging.info(f"Calculated fallback prediction: {fallback_pred:.4f}")

        # --- Lưu Cache ---
        if use_cache and cache_path:
            try:
                if X_raw is not None and y is not None and final_feature_names is not None and fallback_pred is not None:
                    joblib.dump((X_raw, y, final_feature_names, fallback_pred), cache_path)
                    logging.info(f"Preprocessed RAW data cached to {cache_path}")
            except Exception as cache_save_e: logging.warning(f"Failed to save cache: {cache_save_e}")

        logging.info(f"Data prep complete for {data_file_path}. Prepared X_raw={X_raw.shape}, y={y.shape}")
        return X_raw, y, final_feature_names, None, fallback_pred

    except Exception as e:
        logging.error(f"Error during data processing stage for {data_file_path}: {e}", exc_info=True)
        return None, None, None, None, None

# --- Hàm load_and_prepare_combined_data (Cập nhật tính Correlation) ---
def load_and_prepare_combined_data(
    data_file_paths: List[str],
    timeframe: str = '15m',
    future_periods: int = 5,
    input_features_override: Optional[List[str]] = None,
    atr_proxy_factor: float = 0.05,
    target_type: str = 'percent_change', # Thêm lựa chọn target
    use_cache: bool = True,
    use_hl_proxy_feature: bool = False,
    output_dir: str = "."
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[StandardScaler], Optional[float]]:
    """
    Tải và xử lý dữ liệu từ nhiều file, kết hợp, tính correlation, scale chung.
    """
    all_X_raw = []
    all_y = []
    combined_feature_names = None
    combined_fallback_pred = 0.0
    fallback_preds_list = []
    logging.info(f"Starting combined data loading for: {data_file_paths}")

    for file_path in data_file_paths:
        logging.info(f"--- Processing individual file: {file_path} ---")
        # <<< Truyền target_type vào hàm load đơn lẻ >>>
        X_raw_single, y_single, feature_names_single, _, fallback_pred_single = load_and_prepare_data(
            file_path, timeframe, future_periods, input_features_override,
            atr_proxy_factor, target_type, use_cache, use_hl_proxy_feature, output_dir
        )
        # (Kiểm tra và kết hợp như cũ)
        if X_raw_single is not None:
             all_X_raw.append(X_raw_single); all_y.append(y_single)
             if fallback_pred_single is not None: fallback_preds_list.append(fallback_pred_single)
             if combined_feature_names is None: combined_feature_names = feature_names_single
             elif set(combined_feature_names) != set(feature_names_single):
                  logging.error(f"Feature set mismatch!"); return None, None, None, None, None
        else: logging.warning(f"Failed process {file_path}. Skipping.")

    if not all_X_raw: # ... xử lý lỗi ...
         return None, None, None, None, None

    # Kết hợp dữ liệu
    try:
        combined_X_raw = np.concatenate(all_X_raw, axis=0)
        combined_y = np.concatenate(all_y, axis=0)
        logging.info(f"Combined data shape: X_raw={combined_X_raw.shape}, y={combined_y.shape}")
    except ValueError as concat_err: # ... xử lý lỗi ...
         return None, None, None, None, None

    # --- Tính Correlation Matrix ---
    if combined_X_raw is not None and combined_feature_names:
        logging.info("Calculating feature correlation matrix...")
        try:
            sample_size_corr = min(50000, combined_X_raw.shape[0])
            if sample_size_corr > 0:
                indices_corr = np.random.choice(combined_X_raw.shape[0], sample_size_corr, replace=False)
                if np.max(indices_corr) < combined_X_raw.shape[0]:
                    corr_df = pd.DataFrame(combined_X_raw[indices_corr,:], columns=combined_feature_names)
                    correlation_matrix = corr_df.corr()
                    logging.info("Feature Correlation Matrix (Pairs with abs corr > 0.8):")
                    high_corr_pairs = correlation_matrix.unstack()
                    high_corr_pairs = high_corr_pairs[(abs(high_corr_pairs) > 0.8) & (high_corr_pairs < 1.0)]
                    if not high_corr_pairs.empty:
                         processed_pairs = set()
                         for idx, value in high_corr_pairs.sort_values(ascending=False).items():
                              pair = tuple(sorted(idx))
                              if pair not in processed_pairs:
                                   print(f"  {idx[0]:<25} vs {idx[1]:<25} : {value:.3f}")
                                   processed_pairs.add(pair)
                    else: logging.info("  No feature pairs found with absolute correlation > 0.8.")
                else: logging.warning("Indices out of bounds for correlation.")
            else: logging.warning("Not enough data for correlation.")
        except Exception as corr_e: logging.error(f"Correlation calc error: {corr_e}", exc_info=True)
    # --- Kết thúc Correlation ---

    # --- Scaling Chung ---
    logging.info("Applying StandardScaler to combined data...")
    combined_scaler = StandardScaler()
    combined_X_scaled = combined_scaler.fit_transform(combined_X_raw)

    # --- Tính Fallback Prediction Chung ---
    if fallback_preds_list: combined_fallback_pred = float(np.nanmedian(fallback_preds_list))
    else: combined_fallback_pred = 0.0
    logging.info(f"Combined fallback prediction (median): {combined_fallback_pred:.4f}")

    return combined_X_scaled, combined_y, combined_feature_names, combined_scaler, combined_fallback_pred


# --- Khối Chạy Chính (Cập nhật tham số, kiến trúc mặc định, feature importance) ---
if __name__ == "__main__":

    # --- Định nghĩa Cấu hình Thủ công ---
    class Args: pass
    args = Args()
    KAGGLE_INPUT_DIR = "/kaggle/input/dataset"
    KAGGLE_OUTPUT_DIR = "/kaggle/working/"

    # --- Gán các giá trị cấu hình (CẬP NHẬT) ---
    args.timeframe = "15m"
    args.future_periods = 5
    # <<< Sử dụng kiến trúc mặc định mới >>>
    args.hidden_dims_init = '[128, 64, 32]'
    args.train_only = False
    args.n_trials = 50 # Giữ nguyên hoặc giảm nếu cần
    # <<< Thêm lựa chọn target mới >>>
    args.target_type = 'vol_normalized_returns' # Hoặc 'percent_change'
    args.no_cache = False
    args.plot_loss = True
    args.use_hl_proxy = False # Giữ là False trừ khi muốn thử nghiệm
    args.scaler_output = os.path.join(KAGGLE_OUTPUT_DIR, "mlp_action_feature_scaler_refined.pkl") # Đổi tên file
    args.output_model = os.path.join(KAGGLE_OUTPUT_DIR, "mlp_action_model_refined.pth") # Đổi tên file
    args.loss_plot = os.path.join(KAGGLE_OUTPUT_DIR, "mlp_action_loss_curve_refined.png") # Đổi tên file
    args.num_workers = 0
    # ---------------------------------------------------------

    num_workers = args.num_workers
    if platform.system() == "Windows" and num_workers > 0: num_workers = 0

    # --- Danh sách file dữ liệu & Kiểm tra ---
    data_files_to_combine = [ os.path.join(KAGGLE_INPUT_DIR, "BTC_USDT_data.pkl"), os.path.join(KAGGLE_INPUT_DIR, "ETH_USDT_data.pkl") ]
    missing_files = [f for f in data_files_to_combine if not os.path.exists(f)]
    if missing_files: logging.error(f"Input files not found: {missing_files}"); exit()
    else: logging.info(f"Found input data files: {data_files_to_combine}")

    # --- Vòng lặp kiểm tra ATR Proxy Factor (Giữ nguyên logic) ---
    # (Có thể bỏ qua nếu bạn đã hài lòng với factor 0.080)
    logging.info("--- Starting ATR Proxy Factor Evaluation ---")
    atr_proxy_factors_to_test = [0.01, 0.03, 0.05, 0.08, 0.1, 0.12]
    factor_results = {}
    best_prelim_val_loss = float('inf')
    best_atr_proxy_factor = 0.08 # Đặt mặc định là factor tốt nhất tìm được lần trước

    for factor in atr_proxy_factors_to_test:
        # ... (Logic đánh giá factor như cũ, sử dụng load_and_prepare_data đã cập nhật) ...
        logging.info(f"Evaluating ATR Proxy Factor: {factor:.3f}")
        factor_X_raw_list = []
        factor_y_list = []
        factor_feature_names_check = None
        valid_factor_data = True
        for file_path in data_files_to_combine:
            X_raw_single, y_single, f_names_single, _, _ = load_and_prepare_data(
                file_path, args.timeframe, args.future_periods,
                atr_proxy_factor=factor, target_type=args.target_type, # <<< Truyền target_type
                use_cache=(not args.no_cache), use_hl_proxy_feature=args.use_hl_proxy,
                output_dir=KAGGLE_OUTPUT_DIR)
            if X_raw_single is None: valid_factor_data = False; break
            factor_X_raw_list.append(X_raw_single); factor_y_list.append(y_single)
            if factor_feature_names_check is None: factor_feature_names_check = set(f_names_single)
            elif factor_feature_names_check != set(f_names_single):
                logging.error(f"Feat mismatch factor {factor}. Skip."); valid_factor_data = False; break
        if not valid_factor_data or not factor_X_raw_list: continue
        try: # Kết hợp, scale, train nhanh
            combined_X_raw_factor = np.concatenate(factor_X_raw_list, axis=0)
            combined_y_factor = np.concatenate(factor_y_list, axis=0)
            scaler_temp = StandardScaler(); X_scaled_temp = scaler_temp.fit_transform(combined_X_raw_factor)
            X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(X_scaled_temp, combined_y_factor, test_size=0.25, random_state=42)
            if len(X_train_temp) == 0 or len(X_val_temp) == 0: continue
            # <<< Dùng kiến trúc và activation mặc định mới cho test nhanh >>>
            model_temp = MLPAction(input_dim=X_scaled_temp.shape[1], hidden_dims=[64, 32], dropout_p=0.3, activation_fn=nn.SiLU)
            _, prelim_val_losses = model_temp.train_model(X_train_temp, y_train_temp, X_val_temp, y_val_temp, epochs=25, batch_size=64, learning_rate=0.002, optimizer_cls=optim.AdamW, loss_fn=nn.HuberLoss(), patience=5, num_data_workers=num_workers, clip_norm=0.5) # Thêm clip norm, dùng Huber
            if prelim_val_losses:
                final_prelim_val_loss = prelim_val_losses[-1]
                logging.info(f"Factor {factor:.3f}: Prelim Val Loss = {final_prelim_val_loss:.6f}")
                factor_results[factor] = final_prelim_val_loss
                if final_prelim_val_loss < best_prelim_val_loss:
                    best_prelim_val_loss = final_prelim_val_loss; best_atr_proxy_factor = factor
                    logging.info(f"*** New best ATR Proxy Factor: {best_atr_proxy_factor:.3f} ***")
        except Exception as e: logging.error(f"Error factor {factor}: {e}"); continue
    logging.info(f"Selected ATR Proxy Factor: {best_atr_proxy_factor:.3f}")

    # --- Tải và Chuẩn bị Dữ liệu Cuối cùng ---
    logging.info(f"Loading final combined data (ATR factor: {best_atr_proxy_factor:.3f}, Target: {args.target_type})")
    # <<< Gọi hàm load kết hợp với atr factor và target type tốt nhất >>>
    X_data, y_data, final_feature_names, combined_feature_scaler, combined_fallback_prediction = load_and_prepare_combined_data(
        data_files_to_combine, args.timeframe, args.future_periods,
        atr_proxy_factor=best_atr_proxy_factor, target_type=args.target_type,
        use_cache=(not args.no_cache), use_hl_proxy_feature=args.use_hl_proxy,
        output_dir=KAGGLE_OUTPUT_DIR )

    if X_data is None: logging.error("Failed final data load. Exiting."); exit()
    logging.info(f"Final data ready: X={X_data.shape}, y={y_data.shape}, Features={len(final_feature_names)}")

    # --- Lưu Scaler ---
    try: joblib.dump(combined_feature_scaler, args.scaler_output); logging.info(f"Scaler saved: {args.scaler_output}")
    except Exception as e: logging.error(f"Failed save scaler: {e}")

    # --- Khởi tạo mô hình chính với kiến trúc mặc định mới ---
    try:
        initial_hidden_dims = json.loads(args.hidden_dims_init) # Vẫn thử parse từ args
        if not isinstance(initial_hidden_dims, list): raise ValueError
    except:
        initial_hidden_dims = [128, 64, 32] # Default mới
        logging.warning(f"Using default hidden_dims: {initial_hidden_dims}")
    # <<< input_dim lấy từ final_feature_names, dùng SiLU mặc định >>>
    model = MLPAction(input_dim=len(final_feature_names), output_dim=1, hidden_dims=initial_hidden_dims, activation_fn=nn.SiLU)

    # --- Huấn luyện hoặc Tối ưu ---
    train_losses, val_losses, X_test_final, y_test_final = None, None, None, None
    if args.train_only:
        logging.info("Starting training only with final combined data...")
        X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
            X_data, y_data, test_size=0.15, random_state=42
        )
        if len(X_train_full) == 0 or len(X_test_final) == 0:
            logging.error("Empty train or test set after split. Exiting.")
            exit()
        X_train_direct, X_val_direct, y_train_direct, y_val_direct = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
        if len(X_train_direct) == 0 or len(X_val_direct) == 0:
            logging.error("Empty train or validation set after split. Exiting.")
            exit()
        train_lr = 1e-3
        train_wd = 1e-5
        train_opt = 'AdamW'
        train_bs = 64
        train_loss = nn.MSELoss()
        train_sched = optim.lr_scheduler.ReduceLROnPlateau
        train_losses, val_losses = model.train_model(
            X_train_direct, y_train_direct, X_val_direct, y_val_direct,
            epochs=100, batch_size=train_bs, learning_rate=train_lr, weight_decay=train_wd,
            optimizer_cls=getattr(optim, train_opt), loss_fn=train_loss, scheduler_cls=train_sched,
            num_data_workers=num_workers, patience=15, clip_norm=0.5
        )
    else:
        logging.info("Starting hyperparameter optimization...")
        # Hàm optimize_hyperparameters đã được cập nhật để xử lý activation và clip_norm
        train_losses, val_losses, X_test_final, y_test_final = model.optimize_hyperparameters(
            X_data, y_data, n_trials=args.n_trials, test_size=0.15, val_size=0.2,
            num_data_workers=num_workers )

    # --- Lưu mô hình ---
    if model.is_trained:
        model.save_model(args.output_model, fallback_pred=combined_fallback_prediction)
        logging.info(f"Model training complete. Saved to {args.output_model}")

        # --- Vẽ Loss Curves ---
        if args.plot_loss and train_losses and val_losses:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label="Training Loss")
                plt.plot(val_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss Curves")
                plt.legend()
                plt.grid(True)
                plt.savefig(args.loss_plot)
                plt.close()
                logging.info(f"Loss curve plot saved to {args.loss_plot}")
            except Exception as e:
                logging.error(f"Failed to plot loss curves: {e}")

        # --- Đánh giá Cuối cùng và Phân tích ---
        if X_test_final is not None and y_test_final is not None and len(X_test_final)>0:
            logging.info("--- Final Model Evaluation on Hold-Out Test Set ---")
            model.eval()
            test_preds_final = model.predict(X_test_final, fallback_value=combined_fallback_prediction)

            # 1. Metrics Hồi quy (như cũ)
            test_mse = mean_squared_error(y_test_final, test_preds_final)
            test_r2 = r2_score(y_test_final, test_preds_final)
            logging.info(f"Test Set MSE: {test_mse:.6f}")
            logging.info(f"Test Set R-squared: {test_r2:.4f}")

            # 2. Directional Accuracy (như cũ)
            valid_preds_mask = ~np.isnan(test_preds_final.flatten()) & ~np.isnan(y_test_final.flatten())
            if np.any(valid_preds_mask):
                dir_acc = np.mean(np.sign(test_preds_final.flatten()[valid_preds_mask]) == np.sign(y_test_final.flatten()[valid_preds_mask]))
                logging.info(f"Test Set Directional Accuracy: {dir_acc:.4f}")
            else:
                logging.warning("Could not calculate directional accuracy due to NaN values.")

            # 3. Simplified backtest & Sharpe ratio
            logging.info("Running simplified backtest...")
            trades_signal = np.sign(test_preds_final.flatten())
            valid_trades_mask = (trades_signal != 0) & valid_preds_mask
            if np.any(valid_trades_mask):
                simulated_returns = trades_signal[valid_trades_mask] * (y_test_final.flatten()[valid_trades_mask] / 100.0)
                if len(simulated_returns) > 1:
                    mean_return = np.mean(simulated_returns)
                    std_return = np.std(simulated_returns)
                    trades_per_day = (24 * 60) / (args.future_periods * 15)  # Ước tính thô
                    annualization_factor = np.sqrt(365 * trades_per_day)
                    sharpe_ratio = (mean_return / (std_return + 1e-9)) * annualization_factor
                    logging.info(f"Simplified Backtest Sharpe Ratio (Annualized est.): {sharpe_ratio:.4f}")
                else:
                    logging.warning("Not enough returns for Sharpe ratio calculation.")
            else:
                logging.warning("No trades in simplified backtest.")

            # 4. Feature Importance (với Wrapper và final_feature_names)
            logging.info("Calculating feature importance...")
            try:
                class PyTorchWrapper(BaseEstimator, RegressorMixin):
                    def __init__(self, model, fallback):
                        self.model = model
                        self.fallback = fallback
                        if not getattr(self.model, 'is_trained', False):
                            raise ValueError("Model is not trained.")
                        self.model.eval()
                    def fit(self, X, y):
                        return self
                    def predict(self, X):
                        return self.model.predict(X, fallback_value=self.fallback)

                pytorch_estimator = PyTorchWrapper(model, combined_fallback_prediction)
                perm_importance = permutation_importance(
                    estimator=pytorch_estimator,
                    X=X_test_final,
                    y=y_test_final,
                    scoring='neg_mean_squared_error',
                    n_repeats=10,
                    random_state=42,
                    n_jobs=2
                )

                sorted_idx = perm_importance.importances_mean.argsort()[::-1]
                logging.info("Feature Importances (Permutation on Test Set):")
                for i in sorted_idx:
                    if i < len(final_feature_names):
                        print(f"  {final_feature_names[i]:<25}: {perm_importance.importances_mean[i]:.5f} +/- {perm_importance.importances_std[i]:.5f}")
                    else:
                        logging.warning(f"Index {i} out of bounds for feature names.")
            except Exception as fi_e: logging.error(f"Permutation importance error: {fi_e}", exc_info=True)
    else:
        logging.error("Model was not trained successfully.")

    logging.info("MLP Training Script Finished.")