import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # <<< Import
from sklearn.metrics import r2_score, mean_squared_error # <<< Import
from sklearn.inspection import permutation_importance # <<< Import
import logging
import os
import joblib
import argparse
import json
import platform
import matplotlib.pyplot as plt # <<< Import cho loss curves
import pickle


# --- Định nghĩa Lớp MLPAction (Cập nhật) ---
class MLPAction(nn.Module):
    """MLP Implementation. Phiên bản huấn luyện đầy đủ."""
    def __init__(self, input_dim=5, hidden_dims=[64, 32], output_dim=1, dropout_p=0.2, device=None):
        super().__init__()
        self.input_dim = input_dim; self.hidden_dims = hidden_dims; self.output_dim = output_dim
        self.dropout_p = dropout_p; self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logging.info(f"{self.__class__.__name__} Training: Device: {self.device}")
        self._build_network(); self.is_trained = False; self.best_params = None; self.to(self.device)
        try: # Compile
            if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2: self.network = torch.compile(self.network)
        except Exception: pass

    def _build_network(self):
        layers = []; in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(p=self.dropout_p)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.output_dim)); self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(0)
        elif x.ndim != 2 or x.shape[1] != self.input_dim: raise ValueError("Invalid input shape/dim")
        return self.network(x.to(self.device))

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=1e-3, weight_decay=0.0, patience=5, loss_fn=None, optimizer_cls=None, scheduler_cls=None, num_data_workers=0): # Thêm scheduler_cls
        self.train(); train_losses, val_losses = [], [] # <<< Track losses
        try: # Data prep
            X_train_t=torch.tensor(X_train,dtype=torch.float32); y_train_t=torch.tensor(y_train,dtype=torch.float32)
            X_val_t=torch.tensor(X_val,dtype=torch.float32); y_val_t=torch.tensor(y_val,dtype=torch.float32)
            # >>> Label Reshaping động <<<
            if y_train_t.shape[-1] != self.output_dim: y_train_t = y_train_t.view(-1, self.output_dim)
            if y_val_t.shape[-1] != self.output_dim: y_val_t = y_val_t.view(-1, self.output_dim)

            train_dataset=TensorDataset(X_train_t,y_train_t); val_dataset=TensorDataset(X_val_t,y_val_t)
            train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))
            val_loader=DataLoader(val_dataset,batch_size=batch_size, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))
        except Exception as e: logging.error(f"Data prep error: {e}"); return None, None # Trả về losses

        criterion=loss_fn if loss_fn else nn.MSELoss(); optimizer_class=optimizer_cls if optimizer_cls else optim.AdamW
        optimizer=optimizer_class(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # >>> Khởi tạo Scheduler <<<
        scheduler = scheduler_cls(optimizer, patience=max(1, patience // 2), factor=0.5) if scheduler_cls else None # Ví dụ: ReduceLROnPlateau

        best_val_loss=float('inf'); epochs_no_improve=0
        logging.info(f"Training started: {epochs} epochs, LR={learning_rate:.1E}, WD={weight_decay:.1E}, Batch={batch_size}, Workers={num_data_workers}...")
        for epoch in range(epochs):
            self.train(); running_loss=0.0
            for inputs,labels in train_loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(); outputs=self(inputs); loss=criterion(outputs,labels); loss.backward(); optimizer.step(); running_loss+=loss.item()
            avg_train_loss=running_loss/max(1, len(train_loader)); train_losses.append(avg_train_loss) # <<< Store loss
            self.eval(); val_loss=0.0
            with torch.no_grad():
                for inputs_val,labels_val in val_loader:
                     inputs_val, labels_val = inputs_val.to(self.device, non_blocking=True), labels_val.to(self.device, non_blocking=True)
                     outputs_val=self(inputs_val); loss=criterion(outputs_val,labels_val); val_loss+=loss.item()
            avg_val_loss=val_loss/max(1, len(val_loader)); val_losses.append(avg_val_loss) # <<< Store loss
            logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            # >>> Scheduler Step <<<
            if scheduler: scheduler.step(avg_val_loss)
            if avg_val_loss<best_val_loss: best_val_loss=avg_val_loss; epochs_no_improve=0
            else:
                epochs_no_improve+=1
                if epochs_no_improve>=patience: logging.info(f"Early stop @ epoch {epoch+1}"); break
        self.is_trained=True; self.eval(); logging.info(f"Training done. Best Val Loss: {best_val_loss:.6f}")
        # >>> Trả về losses <<<
        return train_losses, val_losses

    def _objective(self, trial, X_train, y_train, X_val, y_val, num_data_workers):
        self.train() # Model chính có thể bị ảnh hưởng nếu không tạo temp_model? -> Vẫn nên tạo temp
        n_layers=trial.suggest_int("n_layers", 1, 4); hidden_dims=[trial.suggest_int(f"n_units_l{i}", 16, 256, log=True) for i in range(n_layers)] # Mở rộng hidden units
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-2, log=True); optimizer_name=trial.suggest_categorical("optimizer", ["AdamW", "Adam", "RMSprop"])
        dropout_p = trial.suggest_float("dropout_p", 0.05, 0.5) # Mở rộng dropout
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if optimizer_name=="AdamW" else 0.0
        # >>> Tối ưu loss và atr_proxy_factor <<<
        loss_type = trial.suggest_categorical("loss", ["mse", "huber", "smoothl1"])
        # atr_proxy_factor = trial.suggest_float("atr_proxy_factor", 0.01, 0.15) # Chỉ thêm nếu load_data trong objective
        # >>> Không nên load data trong objective, load 1 lần bên ngoài <<<

        batch_size=trial.suggest_categorical("batch_size",[32, 64, 128]); epochs=trial.suggest_int("epochs",40, 100) # Tăng max epochs

        # Tạo model tạm
        temp_model=MLPAction(input_dim=self.input_dim, hidden_dims=hidden_dims, output_dim=self.output_dim, dropout_p=dropout_p, device=self.device)
        temp_model.to(self.device)

        try: # Data prep
            X_train_t=torch.tensor(X_train,dtype=torch.float32); y_train_t=torch.tensor(y_train,dtype=torch.float32)
            X_val_t=torch.tensor(X_val,dtype=torch.float32); y_val_t=torch.tensor(y_val,dtype=torch.float32)
            if y_train_t.shape[-1]!=self.output_dim: y_train_t=y_train_t.view(-1,self.output_dim)
            if y_val_t.shape[-1]!=self.output_dim: y_val_t=y_val_t.view(-1,self.output_dim)
            train_dataset=TensorDataset(X_train_t,y_train_t); val_dataset=TensorDataset(X_val_t,y_val_t)
            train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))
            val_loader=DataLoader(val_dataset,batch_size=batch_size, num_workers=num_data_workers, pin_memory=(self.device.type=='cuda'))

            # >>> Chọn Loss Function <<<
            if loss_type == "huber": criterion = nn.HuberLoss()
            elif loss_type == "smoothl1": criterion = nn.SmoothL1Loss()
            else: criterion = nn.MSELoss()

            optimizer_cls=getattr(optim,optimizer_name)
            optimizer=optimizer_cls(temp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            # >>> Thêm Scheduler vào trial <<<
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=False)

            best_trial_val_loss=float('inf'); epochs_no_improve=0; patience=7 # Tăng patience cho trial

            for epoch in range(epochs): # Train loop
                temp_model.train(); running_loss = 0.0
                for inputs,labels in train_loader:
                    inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    optimizer.zero_grad(); outputs=temp_model(inputs); loss=criterion(outputs,labels); loss.backward(); optimizer.step(); running_loss += loss.item()
                temp_model.eval(); current_val_loss=0.0
                with torch.no_grad():
                    for inputs_val,labels_val in val_loader:
                         inputs_val, labels_val = inputs_val.to(self.device, non_blocking=True), labels_val.to(self.device, non_blocking=True)
                         outputs_val=temp_model(inputs_val); loss=criterion(outputs_val,labels_val); current_val_loss+=loss.item()
                avg_val_loss=current_val_loss / max(1, len(val_loader))
                scheduler.step(avg_val_loss) # <<< Scheduler step
                if avg_val_loss<best_trial_val_loss: best_trial_val_loss=avg_val_loss; epochs_no_improve=0
                else:
                    epochs_no_improve+=1
                    if epochs_no_improve>=patience: break
                trial.report(avg_val_loss,epoch)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            return best_trial_val_loss
        except optuna.exceptions.TrialPruned: logging.debug("Trial pruned."); return float('inf') # Debug thay Info
        except Exception as e: logging.error(f"Optuna trial error: {e}"); return float('inf')

    def optimize_hyperparameters(self, X, y, n_trials=50, test_size=0.2, val_size=0.2, num_data_workers=0): # Tăng n_trials
        if X is None or y is None or len(X)==0: logging.error("Empty data for optimization."); return
        logging.info(f"Optimizing hyperparameters (TPESampler, {n_trials} trials)...")
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
            relative_val_size=val_size/(1.0-test_size) if (1.0-test_size) > 0 else val_size
            X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train_val,y_train_val,test_size=relative_val_size,random_state=42)
            logging.info(f"Data split: Train_opt={len(X_train_opt)}, Val_opt={len(X_val_opt)}, Test={len(X_test)}")

            study=optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10)) # Thêm n_startup
            objective_func=lambda trial: self._objective(trial,X_train_opt,y_train_opt,X_val_opt,y_val_opt, num_data_workers)
            study.optimize(objective_func, n_trials=n_trials, n_jobs=1)

            if not study.best_trial:
                logging.warning("Optuna found no valid trials. Using default parameters.")
                self.best_params={'n_layers':2,'n_units_l0':64,'n_units_l1':32,'lr':1e-3,'optimizer':'AdamW','batch_size':32,'epochs':50, 'dropout_p': 0.2, 'weight_decay': 1e-5, 'loss': 'mse'} # Thêm loss mặc định
            else: self.best_params=study.best_params; logging.info(f"Best trial: Value={study.best_value:.6f}, Params={self.best_params}")

            best_hidden_dims=[self.best_params[f"n_units_l{i}"] for i in range(self.best_params["n_layers"])]
            best_lr=self.best_params["lr"]; best_optimizer=self.best_params["optimizer"]
            best_batch_size=self.best_params["batch_size"]; best_epochs=self.best_params.get("epochs",50)
            best_dropout_p = self.best_params.get("dropout_p", 0.2)
            best_weight_decay = self.best_params.get("weight_decay", 0.0) if best_optimizer=="AdamW" else 0.0
            best_loss_type = self.best_params.get("loss", "mse") # <<< Lấy loss tốt nhất

            # Rebuild nếu cần
            if self.hidden_dims!=best_hidden_dims or self.dropout_p != best_dropout_p:
                logging.info(f"Rebuilding model architecture: hidden={best_hidden_dims}, dropout={best_dropout_p}")
                self.hidden_dims=best_hidden_dims; self.dropout_p = best_dropout_p; self._build_network(); self.to(self.device)
                try: # Re-compile
                     if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2: self.network = torch.compile(self.network)
                except Exception: pass

            logging.info("Retraining main model with best params...")
            # >>> Chọn Loss và Scheduler cho lần train cuối <<<
            if best_loss_type == "huber": final_criterion = nn.HuberLoss()
            elif best_loss_type == "smoothl1": final_criterion = nn.SmoothL1Loss()
            else: final_criterion = nn.MSELoss()
            final_scheduler_cls = optim.lr_scheduler.ReduceLROnPlateau # Dùng scheduler này cho train cuối

            # Huấn luyện trên Train + Validation
            train_losses, val_losses = self.train_model(
                X_train_val,y_train_val,X_test,y_test, # Dùng Test làm val set cuối
                epochs=best_epochs, batch_size=best_batch_size, learning_rate=best_lr, weight_decay=best_weight_decay,
                patience=10, # Tăng patience cho train cuối
                loss_fn=final_criterion, optimizer_cls=getattr(optim,best_optimizer),
                scheduler_cls=final_scheduler_cls, # <<< Truyền scheduler
                num_data_workers=num_data_workers
            )
            logging.info("Hyperparameter optimization and final model training complete.")
            # >>> Trả về losses để vẽ biểu đồ <<<
            return train_losses, val_losses, X_test, y_test # Trả về cả dữ liệu test

        except Exception as e: logging.error(f"Hyperparameter optimization error: {e}", exc_info=True); return None, None, None, None

    def save_model(self, file_path="mlp_action_model.pth"):
        try:
            # >>> Lưu thêm fallback prediction <<<
            fallback_pred = self.predict(np.zeros((1, self.input_dim), dtype=np.float32)).item() # Dự đoán trên zeros
            save_data={'state_dict':self.state_dict(),'input_dim':self.input_dim,
                       'hidden_dims':self.hidden_dims,'output_dim':self.output_dim,
                       'dropout_p': self.dropout_p, 'best_params':self.best_params,
                       'fallback_prediction': fallback_pred } # <<< Lưu fallback
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
            # >>> Lấy fallback value <<<
            loaded_fallback_value = checkpoint.get('fallback_prediction', 0.0)

            if loaded_input_dim != self.input_dim or loaded_output_dim != self.output_dim:
                 logging.error(f"{self.__class__.__name__}: Input/Output dim mismatch!"); self.is_trained = False; return False, None

            if loaded_hidden_dims != self.hidden_dims or loaded_dropout_p != self.dropout_p:
                 logging.info(f"{self.__class__.__name__}: Rebuilding network. hidden={loaded_hidden_dims}, dropout={loaded_dropout_p}")
                 self.hidden_dims = loaded_hidden_dims
                 self.dropout_p = loaded_dropout_p
                 self._build_network(); self.to(self.device)
                 try: # Re-compile
                      if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2: self.network = torch.compile(self.network, mode="reduce-overhead")
                 except Exception: pass

            self.load_state_dict(checkpoint['state_dict'])
            self.is_trained = True
            self.eval()
            logging.info(f"{self.__class__.__name__} model loaded from {file_path}")
            # >>> Trả về cả fallback value <<<
            return True, loaded_fallback_value

        except Exception as e:
            logging.error(f"Error loading {self.__class__.__name__} model: {e}", exc_info=True)
            self.is_trained = False; return False, None

    @torch.no_grad()
    def predict(self, x: np.ndarray, fallback_value: float = 0.0) -> np.ndarray:
        default_output_shape = (x.shape[0], self.output_dim) if x.ndim == 2 else (self.output_dim,)

        if not self.is_trained:
            logging.warning(f"{self.__class__.__name__}: Predict on untrained model. Returning fallback: {fallback_value}")
            return np.full(default_output_shape, fallback_value, dtype=np.float32)

        self.eval()
        try:
            x_tensor = torch.tensor(x, dtype=torch.float32)
            predictions_tensor = self(x_tensor)
            result = predictions_tensor.cpu().numpy()
            # Kiểm tra NaN/Inf trong kết quả predict
            if not np.all(np.isfinite(result)):
                 logging.warning(f"{self.__class__.__name__}: Prediction resulted in NaN/Inf. Returning fallback: {fallback_value}")
                 return np.full(default_output_shape, fallback_value, dtype=np.float32)
            return result
        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} prediction: {e}", exc_info=True)
            # Fallback về giá trị đã tính từ training set
            logging.warning(f"{self.__class__.__name__}: Prediction failed. Returning fallback value: {fallback_value}")
            return np.full(default_output_shape, fallback_value, dtype=np.float32)

# --- Hàm Tải và Chuẩn bị Dữ liệu (Cập nhật với Cache, Proxy Norm, trả về feature_names) ---
def load_and_prepare_data(data_file_path, timeframe='15m', future_periods=5, input_features_override=None, atr_proxy_factor=0.05, target_type='percent_change', use_cache=True,use_hl_proxy_feature=False):

    # --- Tạo Cache Path ---
    target_suffix = f"_target{target_type}_fut{future_periods}"
    proxy_suffix = f"_proxy{atr_proxy_factor:.3f}".replace(".","p")
    # Đảm bảo cache_path hợp lệ và tạo thư mục cache nếu cần
    try:
        cache_file_name = os.path.basename(data_file_path).replace(".pkl", f"_{timeframe}{target_suffix}{proxy_suffix}_preprocessed.joblib")
        cache_dir = os.path.dirname(data_file_path) or "." # Lấy thư mục chứa file gốc hoặc thư mục hiện tại
        cache_subdir = os.path.join(cache_dir, "cache") # Lưu cache vào thư mục con "cache"
        os.makedirs(cache_subdir, exist_ok=True) # Tạo thư mục cache nếu chưa có
        cache_path = os.path.join(cache_subdir, cache_file_name)
    except Exception as path_e:
        logging.error(f"Error creating cache path: {path_e}. Disabling caching for this run.")
        use_cache = False
        cache_path = None # Đặt là None nếu không thể tạo path

    # --- Khởi tạo các biến sẽ trả về ---
    X, y, feature_names, scaler, fallback_pred = None, None, None, None, None
    data_dict = None # Khởi tạo data_dict là None

    # --- Tải từ Cache ---
    if use_cache and cache_path and os.path.exists(cache_path): # Thêm kiểm tra cache_path is not None
        try:
            logging.info(f"Loading preprocessed data from cache: {cache_path}")
            X_cached, y_cached, feature_names_cached, scaler_cached, fallback_pred_cached = joblib.load(cache_path)
            # Kiểm tra nhanh shape và loại dữ liệu
            # (Thêm kiểm tra loại dữ liệu để chắc chắn hơn)
            if isinstance(X_cached, np.ndarray) and isinstance(y_cached, np.ndarray) and \
               isinstance(feature_names_cached, list) and hasattr(scaler_cached, 'transform') and \
               isinstance(fallback_pred_cached, (float, np.float32, np.float64)) and \
               len(feature_names_cached) == X_cached.shape[1]:
                 logging.info(f"Loaded from cache: X={X_cached.shape}, y={y_cached.shape}, Fallback={fallback_pred_cached:.4f}")
                 # Gán giá trị từ cache cho các biến trả về
                 X, y, feature_names, scaler, fallback_pred = X_cached, y_cached, feature_names_cached, scaler_cached, fallback_pred_cached
                 return X, y, feature_names, scaler, fallback_pred # Trả về từ cache
            else:
                 logging.warning("Invalid data structure or shape found in cache. Re-processing.")
        except Exception as cache_load_e:
            logging.warning(f"Failed to load from cache {cache_path}: {cache_load_e}. Re-processing.")
            # Không return, tiếp tục để load file gốc

    # --- Tải và Xử lý từ File gốc ---
    # Chỉ thực hiện phần này nếu cache không được dùng hoặc load cache thất bại
    logging.info(f"Processing data from file: {data_file_path}, timeframe: {timeframe}")
    try:
        # <<< BƯỚC LOAD FILE GỐC >>>
        if not os.path.exists(data_file_path):
            logging.error(f"Data file not found: {data_file_path}")
            return None, None, None, None, None

        # Load file .pkl (giả sử là pickle)
        try:
            with open(data_file_path, 'rb') as f:
                data_dict = pickle.load(f)
            logging.debug(f"Successfully loaded data with pickle from {data_file_path}")
        except Exception as load_err:
            logging.error(f"Failed to load data file {data_file_path}: {load_err}")
            return None, None, None, None, None

        # Kiểm tra data_dict sau khi load
        if not isinstance(data_dict, dict):
            logging.error(f"Loaded data from {data_file_path} is not a dictionary.")
            return None, None, None, None, None

        # <<< TRUY CẬP TIMEFRAME SAU KHI LOAD THÀNH CÔNG >>>
        if timeframe not in data_dict:
            logging.error(f"Timeframe '{timeframe}' not found in the loaded data dictionary from {data_file_path}.")
            return None, None, None, None, None

        df_original = data_dict[timeframe] # Bây giờ truy cập df an toàn

        if not isinstance(df_original, pd.DataFrame) or df_original.empty:
             logging.error(f"No valid DataFrame found for timeframe '{timeframe}' in {data_file_path}.")
             return None, None, None, None, None
        logging.debug(f"DataFrame for {timeframe} loaded successfully. Shape: {df_original.shape}")

        # >>> TẠO BẢN SAO ĐỂ TRÁNH THAY ĐỔI DỮ LIỆU GỐC <<<
        df = df_original.copy()

        # --- Xác định input features (Logic không đổi) ---
        if input_features_override:
            feature_names = input_features_override
            INPUT_DIM_ACTUAL = len(feature_names)
            # Kiểm tra xem các feature này có trong df không
            missing_override = [col for col in feature_names if col not in df.columns]
            if missing_override:
                 logging.error(f"Missing columns specified in input_features_override: {missing_override}")
                 return None, None, None, None, None
        else:
            feature_names = ["close", "RSI", "ATR", "volatility"]
            # Cần thêm 'high', 'low' vào check_cols nếu logic if args.use_hl_proxy tồn tại và có thể True
            check_cols = feature_names + ['ATR', 'high', 'low'] # Kiểm tra cả high/low phòng trường hợp logic hl_proxy được kích hoạt sau này
            missing_check = [col for col in check_cols if col not in df.columns]
            if missing_check:
                logging.error(f"Missing base columns required for features: {missing_check}")
                return None, None, None, None, None

            # Chuẩn hóa Proxy ATR (Logic không đổi)
            # Sử dụng fillna + clip để xử lý NaN và giá trị âm/zero một cách an toàn
            atr_safe = df['ATR'].fillna(method='ffill').fillna(method='bfill').fillna(1e-9).clip(lower=1e-9)
            df['bid_ask_spread_proxy'] = atr_safe * atr_proxy_factor
            feature_names.append('bid_ask_spread_proxy')
            if use_hl_proxy_feature:
               logging.info("Calculating and adding High-Low proxy feature.")
               # Kiểm tra lại high, low trước khi tính
               if 'high' in df.columns and 'low' in df.columns:
                   df['hl_proxy'] = (df['high'] - df['low']).clip(lower=1e-9) * 0.1 # Ví dụ factor 0.1
                   feature_names.append('hl_proxy')
               else:
                    logging.warning("Cannot add hl_proxy feature: 'high' or 'low' column missing.")

            INPUT_DIM_ACTUAL = len(feature_names)
            logging.info(f"Using features: {feature_names} (Dim: {INPUT_DIM_ACTUAL})")

        # --- Tính toán Target (Logic không đổi) ---
        df['future_close'] = df['close'].shift(-future_periods)
        if target_type == 'vol_adjusted_change':
            future_atr = df['ATR'].shift(-future_periods).fillna(method='ffill').fillna(method='bfill').fillna(1e-9).clip(lower=1e-9) # Lấy ATR tương lai, fill/clip an toàn
            df['target'] = (df['future_close'] - df['close']) / future_atr
            logging.info(f"Using Volatility-Adjusted target (periods={future_periods}).")
        else: # Mặc định 'percent_change'
            # Xử lý chia cho 0 cho cột 'close'
            close_safe = df['close'].replace(0, np.nan) # Thay 0 bằng NaN để phép chia không lỗi
            df['target'] = (df['future_close'] / close_safe - 1) * 100
            logging.info(f"Using Percent Change target (periods={future_periods}).")

        # --- Làm sạch dữ liệu cuối cùng ---
        # Bỏ NaN trong target trước
        df_target_valid = df.dropna(subset=['target'])

        # Các cột cần thiết cuối cùng (features + target)
        required_cols_final = feature_names + ['target']
        # Bỏ các hàng có NaN trong *bất kỳ* feature nào đã chọn
        df_clean = df_target_valid[required_cols_final].dropna(subset=feature_names)

        # Kiểm tra số lượng mẫu tối thiểu *sau khi* làm sạch
        min_samples_threshold = 100
        if len(df_clean) < min_samples_threshold:
            logging.error(f"Not enough samples ({len(df_clean)}) after cleaning NaN in features/target. Min required: {min_samples_threshold}.")
            return None, None, None, None, None

        # Trích xuất X và y (Logic không đổi)
        X_raw = df_clean[feature_names].values.astype(np.float32)
        y = df_clean['target'].values.astype(np.float32)
        if y.ndim == 1: y = y.reshape(-1, 1)

        # Log correlation (Logic không đổi)
        try:
             if X_raw.shape[1] > 1:
                  corr = pd.DataFrame(X_raw, columns=feature_names).corr()
                  logging.debug(f"Feature Correlations:\n{corr}") # Đổi thành debug
        except Exception: pass

        # --- Feature Scaling (Logic không đổi) ---
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw) # Fit và transform
        logging.info("Input features scaled using StandardScaler.")

        # --- Tính Fallback Prediction (Logic không đổi) ---
        fallback_pred = 0.0 # Khởi tạo mặc định
        if len(X) > 20: # Cần đủ mẫu
             try:
                  # Sử dụng chỉ số integer để tách an toàn
                  train_size_fb = int(len(X) * 0.2)
                  if train_size_fb > 0:
                       y_train_fb = y[:train_size_fb]
                       # Kiểm tra xem y_train_fb có dữ liệu không trước khi tính median
                       if len(y_train_fb) > 0:
                            fallback_pred = float(np.nanmedian(y_train_fb)) # Dùng nanmedian an toàn hơn
                       else:
                            logging.warning("Fallback prediction calculation: y_train_fb is empty.")
                  else:
                       logging.warning("Fallback prediction calculation: train_size_fb is zero.")
             except Exception as fb_calc_e:
                  logging.error(f"Error calculating fallback prediction: {fb_calc_e}")
                  fallback_pred = 0.0 # Reset về 0 nếu lỗi
        else:
             logging.warning(f"Not enough samples ({len(X)}) to calculate fallback prediction reliably. Using 0.0.")
        logging.info(f"Calculated fallback prediction (median of initial 20% y_train): {fallback_pred:.4f}")

        # --- Lưu Cache (Logic không đổi) ---
        if use_cache and cache_path: # Thêm kiểm tra cache_path
            try:
                # Đảm bảo tất cả các biến trả về đều hợp lệ trước khi lưu
                if X is not None and y is not None and feature_names is not None and scaler is not None and fallback_pred is not None:
                    joblib.dump((X, y, feature_names, scaler, fallback_pred), cache_path)
                    logging.info(f"Preprocessed data cached to {cache_path}")
                else:
                    logging.warning("Skipping cache saving due to invalid data components.")
            except Exception as cache_save_e:
                logging.warning(f"Failed to save preprocessed data to cache: {cache_save_e}")

        logging.info(f"Data preparation complete. Prepared X={X.shape}, y={y.shape}")
        return X, y, feature_names, scaler, fallback_pred

    except Exception as e:
        # Log lỗi xảy ra trong quá trình xử lý chính
        logging.error(f"Error during main data processing stage for {data_file_path}: {e}", exc_info=True)
        return None, None, None, None, None
    
# --- Khối Chạy Chính cho Huấn luyện (Cập nhật) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MLPAction model.")
    parser.add_argument("--scaler_output", type=str, default="mlp_action_feature_scaler.pkl", help="Path to save the feature scaler") # <<< Đổi tên file
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--atr_proxy_factor", type=float, default=0.05, help="ATR proxy factor")
    parser.add_argument("--target_type", type=str, default="percent_change", choices=['percent_change', 'vol_adjusted_change'], help="Type of target variable")
    parser.add_argument("--no_cache", action='store_true', help="Disable loading/saving preprocessed data cache.")
    parser.add_argument("--plot_loss", action='store_true', help="Plot training/validation loss curves.")
    parser.add_argument("--use_hl_proxy", action="store_true", help="Use High-Low difference * 0.1 as an additional input feature.")

    args = parser.parse_args()
    num_workers = args.num_workers # ... (Xử lý num_workers cho Windows như cũ) ...
    if platform.system() == "Windows" and num_workers > 0: num_workers = 0

    for factor in [0.01, 0.03, 0.05, 0.1]:
        X_temp, y_temp, _, _, _ = load_and_prepare_data(..., atr_proxy_factor=factor)
        model_temp = MLPAction(input_dim=X_temp.shape[1])
        _, _, X_test_temp, y_test_temp = model_temp.optimize_hyperparameters(X_temp, y_temp, n_trials=10)
        mse = mean_squared_error(y_test_temp, model_temp.predict(X_test_temp))
        logging.info(f"ATR Proxy Factor {factor}: Test MSE {mse:.6f}")

    # --- Tải/Chuẩn bị dữ liệu, Scaler, Fallback ---
    X_data, y_data, feature_names, feature_scaler, fallback_prediction = load_and_prepare_data(
        args.data_file, args.timeframe, args.future_periods,
        atr_proxy_factor=args.atr_proxy_factor, target_type=args.target_type,
        use_cache=(not args.no_cache), use_hl_proxy_feature=args.use_hl_proxy # <<< Sử dụng cache
    )

    if X_data is None or y_data is None or feature_scaler is None: exit()

    # --- Lưu Scaler ---
    try:
        joblib.dump(feature_scaler, args.scaler_output)
        logging.info(f"Feature scaler saved to {args.scaler_output}")
    except Exception as e: logging.error(f"Failed to save scaler: {e}")

    # --- Khởi tạo mô hình ---
    # ... (Parse hidden_dims như cũ) ...
    try: initial_hidden_dims = json.loads(args.hidden_dims_init) # ...
    except: initial_hidden_dims = [64, 32]; logging.warning("Using default hidden_dims.")
    # >>> input_dim lấy từ data đã chuẩn bị <<<
    model = MLPAction(input_dim=X_data.shape[1], output_dim=1, hidden_dims=initial_hidden_dims)

    # --- Huấn luyện hoặc Tối ưu ---
    train_losses, val_losses, X_test_final, y_test_final = None, None, None, None
    if args.train_only:
        logging.info("Starting training only...")
        X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(X_data, y_data, test_size=0.15, random_state=42)
        X_train_direct, X_val_direct, y_train_direct, y_val_direct = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
        # ... (Lấy params mặc định nếu cần như cũ) ...
        train_lr = 1e-3; train_wd = 1e-5; train_opt = 'AdamW'; train_bs = 32; train_loss = nn.MSELoss(); train_sched = optim.lr_scheduler.ReduceLROnPlateau
        train_losses, val_losses = model.train_model(X_train_direct, y_train_direct, X_val_direct, y_val_direct,
                          epochs=100, batch_size=train_bs, learning_rate=train_lr, weight_decay=train_wd,
                          optimizer_cls=getattr(optim, train_opt), loss_fn=train_loss, scheduler_cls=train_sched,
                          num_data_workers=num_workers, patience=15) # Tăng patience
    else:
        # Trả về cả dữ liệu test từ optimize
        train_losses, val_losses, X_test_final, y_test_final = model.optimize_hyperparameters(
            X_data, y_data, n_trials=args.n_trials, test_size=0.15, val_size=0.2,
            num_data_workers=num_workers
        )

    # --- Lưu mô hình (bao gồm fallback) ---
    if model.is_trained:
        model.save_model(args.output_model) # Save giờ đã bao gồm fallback
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
                  plt.savefig("mlp_action_loss_curve.png")
                  plt.close()
                  logging.info("Loss curve plot saved to mlp_action_loss_curve.png")
             except Exception as plot_e:
                  logging.error(f"Failed to plot loss curves: {plot_e}")

        # --- Đánh giá Cuối cùng và Phân tích ---
        if X_test_final is not None and y_test_final is not None and len(X_test_final)>0:
             logging.info("--- Final Model Evaluation on Hold-Out Test Set ---")
             model.eval()
             # <<< Sử dụng predict của model (đã bao gồm fallback) >>>
             test_preds_final = model.predict(X_test_final, fallback_value=fallback_prediction)

             # 1. Metrics Hồi quy
             test_mse = mean_squared_error(y_test_final, test_preds_final)
             test_r2 = r2_score(y_test_final, test_preds_final)
             logging.info(f"Test Set MSE: {test_mse:.6f}")
             logging.info(f"Test Set R-squared: {test_r2:.4f}")

             # 2. Directional Accuracy
             valid_preds_mask = ~np.isnan(test_preds_final.flatten()) & ~np.isnan(y_test_final.flatten())
             if np.any(valid_preds_mask):
                  dir_acc = np.mean(np.sign(test_preds_final.flatten()[valid_preds_mask]) == np.sign(y_test_final.flatten()[valid_preds_mask]))
                  logging.info(f"Test Set Directional Accuracy: {dir_acc:.4f}")
             else: logging.warning("Could not calculate directional accuracy.")

             # 3. Simplified Backtest & Sharpe Ratio
             logging.info("Running simplified backtest...")
             trades_signal = np.sign(test_preds_final.flatten())
             valid_trades_mask = (trades_signal != 0) & valid_preds_mask # Chỉ xét các dự đoán có hướng
             if np.any(valid_trades_mask):

                  # <<< THÊM KIỂM TRA TARGET TYPE >>>
                  if args.target_type == 'percent_change':
                       # Tính returns thực tế (chia 100) chỉ khi target là % change
                       simulated_returns = trades_signal[valid_trades_mask] * (y_test_final.flatten()[valid_trades_mask] / 100.0)
                       if len(simulated_returns) > 1:
                            mean_return = np.mean(simulated_returns); std_return = np.std(simulated_returns)
                            trades_per_day = (24 * 60) / (args.future_periods * 15) # Ước tính
                            annualization_factor = np.sqrt(365 * trades_per_day) # Hoặc 252
                            sharpe_ratio = (mean_return / (std_return + 1e-9)) * annualization_factor
                            logging.info(f"Simplified Backtest Sharpe Ratio (Annualized est., Target: %change): {sharpe_ratio:.4f}")
                       else: logging.warning("Not enough returns for Sharpe calculation.")

                  elif args.target_type == 'vol_normalized_returns':
                       # Khi target là vol-normalized, báo cáo PnL chuẩn hóa
                       simulated_norm_returns = trades_signal[valid_trades_mask] * y_test_final.flatten()[valid_trades_mask]
                       logging.warning("Target is vol_normalized. Reporting normalized PnL stats instead of Sharpe.")
                       if len(simulated_norm_returns) > 1:
                            mean_norm_return = np.mean(simulated_norm_returns)
                            std_norm_return = np.std(simulated_norm_returns)
                            skew_norm_return = pd.Series(simulated_norm_returns).skew() # Thêm skewness
                            kurt_norm_return = pd.Series(simulated_norm_returns).kurt() # Thêm kurtosis
                            logging.info(f"Simplified Backtest Mean Norm Return: {mean_norm_return:.4f}")
                            logging.info(f"Simplified Backtest Std Dev Norm Return: {std_norm_return:.4f}")
                            logging.info(f"Simplified Backtest Skewness Norm Return: {skew_norm_return:.4f}")
                            logging.info(f"Simplified Backtest Kurtosis Norm Return: {kurt_norm_return:.4f}")
                       else: logging.warning("Not enough normalized returns for stats.")
                  else:
                       logging.warning(f"Simplified backtest not implemented for target type: {args.target_type}")
                  # <<< KẾT THÚC KIỂM TRA TARGET TYPE >>>

             else: logging.warning("No valid (non-zero signal) trades found in simplified backtest.")

             # 4. Feature Importance (Permutation)
             logging.info("Calculating feature importance...")
             try:
                  def torch_predict_wrapper(X_wrap):
                      return model.predict(X_wrap, fallback_value=fallback_prediction) # Dùng predict wrapper

                  perm_importance = permutation_importance(
                       estimator={'predict': torch_predict_wrapper}, X=X_test_final, y=y_test_final,
                       scoring='neg_mean_squared_error', n_repeats=10, random_state=42, n_jobs=2
                  )

                  sorted_idx = perm_importance.importances_mean.argsort()[::-1]
                  logging.info("Feature Importances (Permutation on Test Set):")
                  for i in sorted_idx: print(f"  {feature_names[i]:<25}: {perm_importance.importances_mean[i]:.5f} +/- {perm_importance.importances_std[i]:.5f}")
             except Exception as fi_e: logging.error(f"Permutation importance error: {fi_e}")

    else:
        logging.error("Model was not trained successfully.")