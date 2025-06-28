import os
import sys
import logging
from pathlib import Path
import traceback
import json # <<< THÊM IMPORT JSON >>>
import gc # Import garbage collector

# ... (Phần Logging và xử lý sys.path giữ nguyên) ...
log_file_path = "train_final_gnn_model.log" # <<< Đổi tên file log cho phù hợp >>>
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)
logging.info(f"Logging initialized for FINAL GNN training. Log file: {log_file_path}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from typing import List, Tuple, Optional

# ... (Phần import PyTorch Geometric và Fallback giữ nguyên) ...
try:
    from torch_geometric.nn import GATConv as GraphConv # Dùng GAT như đề xuất
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader # Dùng DataLoader của torch_geometric
    from torch_geometric.nn import global_mean_pool, global_max_pool # Import pooling
    _torch_geometric_available = True
    logging.info("Successfully imported torch_geometric.")
except ImportError:
     logging.error("torch_geometric required for final training. Exiting.")
     exit() # <<< Thoát nếu không có PyG >>>

# ... (Phần import IntelligentStopSystem và CONFIG giữ nguyên) ...
try:
    from api import IntelligentStopSystem, CONFIG
    logging.info("Successfully imported ISS and CONFIG from api.py")
    if not isinstance(CONFIG, dict) or not CONFIG: CONFIG={}
except ImportError as e:
    logging.warning(f"Could not import from api.py: {e}. Using defaults.")
    class IntelligentStopSystem:
        def __init__(self,t=5):
            self.gnn_features=["close","RSI","ATR","volatility","volume"]
    CONFIG = {}
except Exception as import_err:
    logging.error(f"Unexpected error importing from api.py: {import_err}", exc_info=True)
    class IntelligentStopSystem:
        def __init__(self,t=5):
            self.gnn_features=["close","RSI","ATR","volatility","volume"]
    CONFIG = {}


# ... (Phần Device Setup giữ nguyên) ...
if torch.cuda.is_available(): DEVICE = torch.device("cuda:0"); torch.cuda.set_device(DEVICE)
else: DEVICE = torch.device("cpu")
logging.info(f"Explicitly set device: {DEVICE}")


# --- Paths Setup ---
# <<< Lấy đường dẫn từ thư mục chứa kết quả Optuna >>>
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or 'KAGGLE_WORKING_DIR' in os.environ
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_OUTPUT_DIR = Path("/kaggle/working")

if IS_KAGGLE and KAGGLE_INPUT_DIR.exists():
    DATASET_NAME = "data123" # <<< THAY TÊN DATASET DỮ LIỆU GỐC >>>
    OPTUNA_RESULTS_DATASET = "optuna-gnn-results-dataset" # <<< THAY TÊN DATASET CHỨA KẾT QUẢ OPTUNA >>>
    DATA_DIR = KAGGLE_INPUT_DIR / DATASET_NAME
    OPTUNA_DIR = KAGGLE_INPUT_DIR / OPTUNA_RESULTS_DATASET # Đọc từ dataset kết quả Optuna
    MODEL_SAVE_DIR = KAGGLE_OUTPUT_DIR / "final_trained_gnn" # Thư mục lưu model cuối cùng
    if not DATA_DIR.exists(): exit(f"Data Dataset '{DATA_DIR}' not found.")
    if not OPTUNA_DIR.exists(): exit(f"Optuna Results Dataset '{OPTUNA_DIR}' not found.")
else:
    try: SCRIPT_DIR = Path(__file__).parent
    except NameError: SCRIPT_DIR = Path.cwd()
    DATA_DIR = SCRIPT_DIR
    OPTUNA_DIR = SCRIPT_DIR / "optuna_gnn_results" # Thư mục chứa kết quả Optuna local
    MODEL_SAVE_DIR = SCRIPT_DIR / "final_trained_gnn" # Thư mục lưu model cuối cùng local

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_HPARAMS_PATH = OPTUNA_DIR / "best_gnn_hparams.json" # Đường dẫn tới file hparams
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "final_iss_gnn_anomaly.pth" # Tên file model cuối cùng
SCALER_SAVE_PATH = MODEL_SAVE_DIR / "final_iss_gnn_scaler.pkl" # Scaler dùng cho model cuối

logging.info(f"Using DATA_DIR: {DATA_DIR}")
logging.info(f"Loading best hyperparameters from: {BEST_HPARAMS_PATH}")
logging.info(f"Final model will be saved to: {MODEL_SAVE_PATH}")
logging.info(f"Scaler for final model will be saved to: {SCALER_SAVE_PATH}")

# --- Load Best Hyperparameters ---
if not BEST_HPARAMS_PATH.exists():
    exit(f"Best hyperparameters file not found at {BEST_HPARAMS_PATH}. Please run Optuna first.")
try:
    with open(BEST_HPARAMS_PATH, 'r') as f:
        best_hparams = json.load(f)
    logging.info(f"Loaded best hyperparameters: {best_hparams}")
except Exception as e:
    exit(f"Error loading best hyperparameters from {BEST_HPARAMS_PATH}: {e}")

# --- Parameters ---
SYMBOLS_TO_TRAIN = CONFIG.get("symbols", ["BTC/USDT", "ETH/USDT"])
TIMEFRANE = "15m"
GRAPH_LOOKBACK = 30 # <<< Giữ cố định như lúc chạy Optuna >>>
try: temp_iss = IntelligentStopSystem(); NODE_FEATURES = temp_iss.gnn_features; NODE_DIM = len(NODE_FEATURES)
except Exception: NODE_FEATURES = ["close", "RSI", "ATR", "volatility", "volume"]; NODE_DIM = 5
TARGET_FEATURE = "volatility"
REQUIRED_COLS_LOAD = list(set(NODE_FEATURES + [TARGET_FEATURE]))
VOL_SPIKE_THRESHOLD = 1.4
EDGE_DIM = 2 # Giữ cố định

# <<< SỬ DỤNG HYPERPARAMETERS TỪ OPTUNA >>>
EPOCHS = 150 # <<< Tăng số Epoch cho huấn luyện cuối cùng >>>
BATCH_SIZE = best_hparams.get('batch_size', 256) # Lấy từ best_hparams, có giá trị mặc định
LEARNING_RATE = best_hparams.get('lr', 0.0005)
VALIDATION_SPLIT = 0.15 # Giữ nguyên tỷ lệ split
EARLY_STOPPING_PATIENCE = 15 # <<< Tăng Patience cho huấn luyện cuối cùng >>>
MIN_SAMPLES_FOR_TRAINING = 1000
WEIGHT_DECAY = best_hparams.get('weight_decay', 1e-5)
GRADIENT_CLIP_NORM = best_hparams.get('grad_clip', 1.0) # Lấy từ best_hparams
FOCAL_LOSS_ALPHA = best_hparams.get('focal_alpha', 0.25)
FOCAL_LOSS_GAMMA = best_hparams.get('focal_gamma', 2.0)
GAT_HEADS = best_hparams.get('heads', 4)
HIDDEN_DIM_GNN = best_hparams.get('hidden_dim', 64)
DROPOUT_GNN = best_hparams.get('dropout', 0.4)
GNN_LAYERS = best_hparams.get('gnn_layers', 3) # <<< Lấy số lớp GNN >>>

logging.info("--- Final Training Parameters ---")
logging.info(f"EPOCHS: {EPOCHS}")
logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
logging.info(f"LEARNING_RATE: {LEARNING_RATE}")
logging.info(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
logging.info(f"GRADIENT_CLIP_NORM: {GRADIENT_CLIP_NORM}")
logging.info(f"FOCAL_LOSS_ALPHA: {FOCAL_LOSS_ALPHA}")
logging.info(f"FOCAL_LOSS_GAMMA: {FOCAL_LOSS_GAMMA}")
logging.info(f"GNN_LAYERS: {GNN_LAYERS}")
logging.info(f"HIDDEN_DIM_GNN: {HIDDEN_DIM_GNN}")
logging.info(f"GAT_HEADS: {GAT_HEADS}")
logging.info(f"DROPOUT_GNN: {DROPOUT_GNN}")
logging.info("-------------------------------")


# ... (Phần Helper Functions: load_and_prepare_gnn_data, build_market_graph giữ nguyên) ...
def load_and_prepare_gnn_data(symbol: str, required_cols: List[str]) -> Optional[Tuple[str, pd.DataFrame]]:
    symbol_safe = symbol.replace('/', '_').replace(':', '')
    file_path = DATA_DIR / f"{symbol_safe}_data.pkl"
    if not file_path.exists(): logging.warning(f"Data file not found: {file_path}"); return None
    try:
        data_dict = joblib.load(file_path); df = data_dict.get(TIMEFRANE)
        if not isinstance(df, pd.DataFrame) or df.empty: return None; df = df.copy()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: logging.error(f"Missing columns for {symbol}: {missing_cols}"); return None
        df_processed = df[required_cols].ffill().bfill().fillna(0)
        if df_processed.isnull().values.any(): logging.error(f"NaNs remain for {symbol}"); return None
        return symbol, df_processed
    except Exception as e: logging.error(f"Err load/prep {symbol}: {e}"); return None

def build_market_graph(df_slice_scaled: np.ndarray, lookback: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    current_node_dim = df_slice_scaled.shape[1]
    if len(df_slice_scaled) != lookback or df_slice_scaled.ndim != 2 or current_node_dim != NODE_DIM: # <<< KIỂM TRA NODE_DIM CHÍNH XÁC >>>
        logging.error(f"build_market_graph: Invalid input shape {df_slice_scaled.shape}, expected ({lookback}, {NODE_DIM})")
        return None, None, None
    node_features = torch.tensor(df_slice_scaled, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long); edge_attrs_list = []; full_edge_list = []
    try: close_idx = NODE_FEATURES.index('close'); vol_idx = NODE_FEATURES.index('volume'); has_indices = True
    except ValueError: logging.warning("Cannot find 'close'/'volume'. Edge attrs will be None."); has_indices = False
    if lookback >= 2:
        for i in range(lookback - 1):
            src, dst = i, i + 1; full_edge_list.extend([(src, dst), (dst, src)])
            if has_indices:
                try:
                    price_change = abs(node_features[dst, close_idx].item() - node_features[src, close_idx].item())
                    vol_change = abs(node_features[dst, vol_idx].item() - node_features[src, vol_idx].item())
                    attr = [price_change, vol_change]; edge_attrs_list.extend([attr, attr])
                except Exception as e: logging.error(f"Err calc edge attrs i={i}: {e}"); has_indices = False; edge_attrs_list = []
        if full_edge_list: edge_index = torch.tensor(full_edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float32) if has_indices and edge_attrs_list and edge_index.shape[1] > 0 else None
    if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]: logging.error(f"Edge attr/index shape mismatch"); edge_attr = None
    elif edge_index.shape[1] == 0: edge_attr = None
    return node_features, edge_index, edge_attr


# ... (Phần Model GNN, FocalLoss, Dataset, time_based_split, collate_fn giữ nguyên) ...
# <<< Sử dụng GNN_LAYERS từ hparams >>>
class GraphNeuralNetPyG_Advanced(nn.Module):
    def __init__(self, node_dim, num_layers=GNN_LAYERS, edge_dim=EDGE_DIM, hidden_dim=HIDDEN_DIM_GNN, out_dim=1, dropout=DROPOUT_GNN, heads=GAT_HEADS):
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
        if x is None or edge_index is None or x.nelement()==0: batch_size=batch.max().item()+1 if batch is not None and batch.numel()>0 else 1; return torch.zeros((batch_size,1),device=DEVICE)
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
        except Exception as e: logging.error(f"GNN forward error: {e}"); batch_size=batch.max().item()+1 if batch is not None and batch.numel()>0 else 1; return torch.zeros((batch_size,1),device=DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, reduction='mean'): # <<< DÙNG HPARAMS >>>
        super(FocalLoss, self).__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction; self.bce_with_logits=nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, inputs, targets):
        targets=targets.type_as(inputs); BCE_loss=self.bce_with_logits(inputs, targets); pt=torch.exp(-BCE_loss); alpha_t=self.alpha*targets+(1-self.alpha)*(1-targets); F_loss=alpha_t*(1-pt)**self.gamma*BCE_loss
        if self.reduction=='mean': return torch.mean(F_loss)
        elif self.reduction=='sum': return torch.sum(F_loss)
        else: return F_loss

class GraphDatasetPyG(Dataset):
    def __init__(self, df, node_features_list, target_col, graph_lookback, vol_spike_threshold, scaler):
        self.df=df; self.node_features_list=node_features_list; self.num_node_features=len(node_features_list); self.target_col=target_col; self.graph_lookback=graph_lookback; self.vol_spike_threshold=vol_spike_threshold; self.scaler=scaler
        self.targets,self.valid_indices=self._calculate_targets_and_indices(); self.pos_weight_value=self._calculate_pos_weight()
        if not self.valid_indices.size>0: logging.warning("No valid indices.")
    def _calculate_targets_and_indices(self):
        targets=[]; valid_indices=[]; vol_values=self.df[self.target_col].values; num_rows=len(self.df)
        for i in range(num_rows-self.graph_lookback):
            end_idx=i+self.graph_lookback; target_idx=end_idx;
            if target_idx>=num_rows: continue
            current_vol=vol_values[end_idx-1]; next_vol=vol_values[target_idx]
            target=1.0 if pd.notna(next_vol) and pd.notna(current_vol) and current_vol>1e-9 and next_vol>current_vol*self.vol_spike_threshold else 0.0
            targets.append(target); valid_indices.append(i)
        return np.array(targets),np.array(valid_indices,dtype=int)
    def _calculate_pos_weight(self) -> Optional[torch.Tensor]: # Thêm type hint
        """Calculates the positive class weight for imbalanced datasets."""
        if self.targets is None or len(self.targets) == 0:
            logging.warning("Cannot calculate pos_weight: Targets are missing or empty.")
            return None

        try:
            # Chuyển đổi sang NumPy nếu cần và tính toán
            targets_np = self.targets.cpu().numpy() if isinstance(self.targets, torch.Tensor) else np.array(self.targets)
            num_total = len(targets_np)
            num_pos = np.sum(targets_np == 1)
            num_neg = num_total - num_pos # Tính toán an toàn hơn

            if num_pos == 0:
                logging.warning("No positive samples found in targets. Cannot calculate pos_weight.")
                return None # Trả về None khi không có mẫu dương
            elif num_neg == 0:
                 logging.warning("No negative samples found in targets. pos_weight calculation might be ill-defined (division by zero if used inversely). Returning None for safety.")
                 return None # Cũng trả về None nếu không có mẫu âm để tránh lỗi tiềm ẩn

            # Tính toán pos_weight chỉ khi có cả mẫu dương và âm
            pos_w_value = num_neg / num_pos
            logging.info(f"Calculated pos_weight: {pos_w_value:.2f} (Positives: {num_pos}/{num_total})")

            # Trả về dưới dạng Tensor trên đúng device
            # Đảm bảo DEVICE được định nghĩa ở scope class hoặc global
            try:
                 # Lấy device từ tham số model nếu có thể, nếu không dùng DEVICE global
                 target_device = next(self.model.parameters()).device if hasattr(self,'model') and next(self.model.parameters(), None) is not None else DEVICE
            except:
                 target_device = DEVICE # Fallback về DEVICE global

            return torch.tensor([pos_w_value], dtype=torch.float32, device=target_device)

        except Exception as e:
            logging.error(f"Error calculating pos_weight: {e}", exc_info=True)
            return None
    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx) -> Optional[Data]:
        # --- Bước 1: Kiểm tra chỉ số idx đầu vào ---
        if not isinstance(idx, int) or idx < 0:
             # logging.debug(f"Invalid index type or value received: {idx}")
             return None
        if idx >= len(self.valid_indices):
            # logging.debug(f"Index {idx} out of bounds for valid_indices (len: {len(self.valid_indices)})")
            # Điều này có thể xảy ra nếu DataLoader hoặc sampler có vấn đề
            # Hoặc nếu __len__ trả về giá trị sai
            return None

        # --- Bước 2: Lấy chỉ số bắt đầu và tính chỉ số kết thúc ---
        try:
            actual_start_idx = self.valid_indices[idx]
            # Kiểm tra kiểu dữ liệu của chỉ số bắt đầu
            if not isinstance(actual_start_idx, (int, np.integer)):
                 logging.error(f"Invalid start index type at valid_indices[{idx}]: {type(actual_start_idx)}")
                 return None
        except IndexError:
             # Lỗi này không nên xảy ra nếu kiểm tra idx ở trên đúng
             logging.error(f"IndexError accessing valid_indices[{idx}] (len: {len(self.valid_indices)})")
             return None
        except Exception as e:
             logging.error(f"Error retrieving start index for idx {idx}: {e}")
             return None

        end_idx = actual_start_idx + self.graph_lookback # End index (exclusive for iloc)

        # --- Bước 3: Kiểm tra phạm vi chỉ số trong DataFrame gốc ---
        # Giả sử self.df là DataFrame đầy đủ
        df_len = len(self.df)
        if actual_start_idx < 0 or end_idx > df_len:
            # logging.debug(f"Calculated slice indices out of bounds: start={actual_start_idx}, end={end_idx}, df_len={df_len}")
            return None

        # --- Bước 4: Trích xuất và Scale Features ---
        try:
            # Lấy slice các feature cần thiết
            df_slice_unscaled = self.df.iloc[actual_start_idx:end_idx][self.node_features_list]

            # Kiểm tra lại kích thước slice trước khi scale
            if len(df_slice_unscaled) != self.graph_lookback:
                 logging.warning(f"Slice length mismatch for idx {idx}: expected {self.graph_lookback}, got {len(df_slice_unscaled)}. Start={actual_start_idx}, End={end_idx}")
                 return None

            # Scale dữ liệu
            if self.scaler:
                scaled_slice_values = self.scaler.transform(df_slice_unscaled.values)
            else:
                scaled_slice_values = df_slice_unscaled.values # Dùng giá trị không scale nếu không có scaler

            # Kiểm tra NaN/Inf sau khi scale hoặc lấy giá trị
            if not np.all(np.isfinite(scaled_slice_values)):
                 logging.warning(f"NaN or Inf found in feature values for idx {idx} after scaling/extraction.")
                 return None

        except KeyError as ke:
             logging.error(f"KeyError extracting features for idx {idx}: Missing columns {ke}. Features expected: {self.node_features_list}")
             return None
        except ValueError as ve:
             # Thường do scaler không khớp số feature
             logging.error(f"ValueError during scaling for idx {idx}: {ve}. Check scaler dimensions vs data.")
             return None
        except Exception as e:
            logging.error(f"Error slicing/scaling features for idx {idx}: {e}", exc_info=True)
            return None

        # --- Bước 5: Xây dựng Graph ---
        # Hàm này cần được định nghĩa ở ngoài hoặc import
        node_features, edge_index, edge_attr = build_market_graph(scaled_slice_values, self.graph_lookback)

        if node_features is None or edge_index is None:
            # Hàm build_market_graph nên log lỗi bên trong nó
            # logging.debug(f"build_market_graph returned None for idx {idx}.")
            return None

        # --- Bước 6: Lấy Target ---
        try:
            # Lấy target tương ứng với chỉ số idx gốc (trong self.targets)
            target_value = self.targets[idx]
            target = torch.tensor([target_value], dtype=torch.float32) # Hoặc torch.long cho classification
        except IndexError:
            logging.error(f"IndexError accessing targets[{idx}] (len: {len(self.targets)})")
            return None
        except Exception as e:
            logging.error(f"Error retrieving target for idx {idx}: {e}")
            return None

        # --- Bước 7: Tạo và Trả về đối tượng Data ---
        try:
            data_obj = Data(x=node_features, edge_index=edge_index, y=target, edge_attr=edge_attr)
            return data_obj
        except Exception as e:
             logging.error(f"Error creating PyG Data object for idx {idx}: {e}", exc_info=True)
             return None
def time_based_split(dataset, val_ratio=VALIDATION_SPLIT): # <<< DÙNG HPARAMS >>>
    n=len(dataset); val_size=int(n*val_ratio); train_size=n-val_size
    if train_size<=0 or val_size<=0: raise ValueError(f"Cannot split dataset size {n}")
    train_indices=list(range(train_size)); val_indices=list(range(train_size,n))
    return Subset(dataset,train_indices),Subset(dataset,val_indices)

def collate_fn(batch):
    batch=[d for d in batch if d is not None];
    if not batch: return None
    try: return Batch.from_data_list(batch)
    except Exception as e: logging.error(f"Collate err: {e}"); return None


# --- Main Script Execution ---
if __name__ == "__main__":
    logging.info("--- Starting FINAL GNN Model Training ---")

    # 1. Load Data -> loaded_data_list (Giữ nguyên)
    loaded_data_list = []; required_gnn_cols = list(set(NODE_FEATURES + [TARGET_FEATURE]))
    for symbol in SYMBOLS_TO_TRAIN:
        load_result = load_and_prepare_gnn_data(symbol, required_gnn_cols)
        if load_result: loaded_data_list.append(load_result)
    if not loaded_data_list: exit("No GNN data loaded. Exiting.")

    # 2. Fit or Load Scaler -> scaler
    if SCALER_SAVE_PATH.exists():
        try:
            scaler = joblib.load(SCALER_SAVE_PATH)
            logging.info(f"Loaded existing scaler from {SCALER_SAVE_PATH}")
            # <<< THÊM KIỂM TRA DIMENSION SCALER >>>
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != NODE_DIM:
                logging.warning(f"Loaded scaler dimension ({scaler.n_features_in_}) mismatch expected NODE_DIM ({NODE_DIM}). Refitting scaler.")
                scaler = None # Đặt lại để fit mới
            elif not hasattr(scaler, 'transform'): # Kiểm tra cơ bản
                 logging.warning("Loaded object is not a valid scaler. Refitting scaler.")
                 scaler = None
        except Exception as e:
            logging.error(f"Error loading scaler from {SCALER_SAVE_PATH}: {e}. Refitting scaler.")
            scaler = None
    else:
        scaler = None # Scaler chưa tồn tại

    if scaler is None: # Fit scaler nếu chưa load được hoặc cần refit
        logging.info("Fitting a new scaler...")
        cols_to_scale = NODE_FEATURES
        feature_dfs_to_concat = []
        for _, df in loaded_data_list:
            if all(col in df.columns for col in cols_to_scale): feature_dfs_to_concat.append(df[cols_to_scale])
        if not feature_dfs_to_concat: exit("No valid data for scaler fitting.")
        full_df_combined_features = pd.concat(feature_dfs_to_concat, ignore_index=True)
        if np.any(np.isnan(full_df_combined_features.values)) or np.any(np.isinf(full_df_combined_features.values)):
            exit("NaN/Inf detected in features before scaling. Cannot fit scaler.")
        scaler = StandardScaler()
        scaler.fit(full_df_combined_features.values)
        try:
            joblib.dump(scaler, SCALER_SAVE_PATH)
            logging.info(f"New scaler fitted and saved to {SCALER_SAVE_PATH}")
        except Exception as e:
             logging.error(f"Error saving newly fitted scaler: {e}")
             # Vẫn tiếp tục với scaler đã fit trong bộ nhớ


    # 3. Tạo Dataset -> full_dataset (Giữ nguyên)
    logging.info("Creating PyG Dataset...")
    original_dfs_to_concat = []; required_gnn_cols = list(set(NODE_FEATURES + [TARGET_FEATURE]))
    for _, df in loaded_data_list:
        if all(col in df.columns for col in required_gnn_cols): original_dfs_to_concat.append(df[required_gnn_cols])
    if not original_dfs_to_concat: exit("No valid dataframes for dataset creation.")
    full_original_df_combined = pd.concat(original_dfs_to_concat, ignore_index=True)
    full_dataset = GraphDatasetPyG(full_original_df_combined, NODE_FEATURES, TARGET_FEATURE, GRAPH_LOOKBACK, VOL_SPIKE_THRESHOLD, scaler)
    dataset_len = len(full_dataset)
    logging.info(f"PyG Dataset created with {dataset_len} samples.")
    if dataset_len < MIN_SAMPLES_FOR_TRAINING: exit(f"Dataset size {dataset_len} < min {MIN_SAMPLES_FOR_TRAINING}.")


    # 4. Time-based Split -> train_dataset, val_dataset (Giữ nguyên)
    logging.info("Performing time-based data split...")
    try:
        train_dataset, val_dataset = time_based_split(full_dataset, val_ratio=VALIDATION_SPLIT)
        logging.info(f"Time-based split: Train={len(train_dataset)}, Validation={len(val_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             raise ValueError("Train or validation set is empty after split.")
    except ValueError as split_err: exit(f"Error splitting dataset: {split_err}")


    # 5. Weighted Sampler -> train_sampler (Giữ nguyên)
    logging.info("Calculating weights for WeightedRandomSampler...")
    train_sampler = None
    try:
        train_indices = train_dataset.indices; train_targets = full_dataset.targets[train_indices]
        if len(train_targets) > 0:
             class_counts = np.bincount(train_targets.astype(int))
             if len(class_counts) >= 2 and class_counts[1] > 0:
                  class_weights=1./class_counts; sample_weights=np.array([class_weights[t] for t in train_targets.astype(int)])
                  train_sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True)
                  logging.info(f"WeightedRandomSampler created for training set.")
             elif class_counts[1] == 0: logging.warning("No positive samples in training set for sampler.")
             else: logging.warning("Only one class in training set for sampler.")
        else: logging.warning("Training dataset empty for sampler.")
    except Exception as sampler_e: logging.error(f"Sampler error: {sampler_e}")


    # 6. Tạo DataLoader -> train_loader, val_loader (DÙNG BATCH_SIZE TỪ HPARAMS)
    logging.info("Creating DataLoaders...")
    if _torch_geometric_available:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, collate_fn=collate_fn, shuffle=(train_sampler is None))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
        logging.info(f"Using PyG DataLoader with OPTIMIZED batch_size={BATCH_SIZE}. Train sampler: {'Enabled' if train_sampler else 'Disabled'}")
    else: exit("torch_geometric DataLoader required.")


    # 7. Khởi tạo Model, Optimizer, Loss, Scheduler (DÙNG HPARAMS TỪ OPTUNA)
    logging.info("Initializing model, optimizer, loss, and scheduler with OPTIMIZED hyperparameters...")
    # <<< SỬ DỤNG CÁC THAM SỐ ĐÃ LOAD >>>
    model = GraphNeuralNetPyG_Advanced(
        node_dim=NODE_DIM,
        num_layers=GNN_LAYERS, # Từ hparams
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN_DIM_GNN, # Từ hparams
        dropout=DROPOUT_GNN, # Từ hparams
        heads=GAT_HEADS # Từ hparams
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Từ hparams
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7) # Patience có thể giữ nguyên hoặc lấy từ hparams nếu có
    criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA) # Từ hparams


    # 8. Training Loop (Giống như trước nhưng dùng các thành phần đã khởi tạo với hparams)
    best_val_metric = -1.0 # Theo dõi Val F1
    no_improve_epochs = 0
    logging.info(f"--- Starting FINAL GNN training for {EPOCHS} epochs using best hparams ---")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train(); train_loss_sum=0.0; all_train_targets=[]; all_train_preds_probs=[]; processed_train_samples=0
        for i, batch in enumerate(train_loader):
            if batch is None: continue
            try:
                 batch=batch.to(DEVICE)
                 if batch.x is None or batch.edge_index is None or batch.y is None or batch.batch is None or batch.x.nelement()==0: continue
                 optimizer.zero_grad(); edge_attr=batch.edge_attr if hasattr(batch,'edge_attr') else None
                 logits=model(batch.x, batch.edge_index, edge_attr=edge_attr, batch=batch.batch)
                 target=batch.y.view_as(logits).float(); loss=criterion(logits, target)
                 if torch.isnan(loss) or torch.isinf(loss): logging.warning(f"NaN/Inf train loss batch {i}"); continue
                 loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM); optimizer.step() # <<< DÙNG GRADIENT_CLIP_NORM TỪ HPARAMS >>>
                 train_loss_sum+=loss.item()*target.size(0); processed_train_samples+=target.size(0)
                 all_train_targets.extend(batch.y.squeeze().cpu().tolist())
                 all_train_preds_probs.extend(torch.sigmoid(logits.squeeze()).cpu().tolist())
            except Exception as e: logging.error(f"E{epoch+1} Train Batch {i} Err: {e}"); continue
        train_loss_avg=train_loss_sum/processed_train_samples if processed_train_samples > 0 else float('inf')
        # Tính Train Metrics
        train_precision, train_recall, train_f1, train_auc = 0.0, 0.0, 0.0, 0.0
        if all_train_targets:
            try:
                train_preds=(np.array(all_train_preds_probs)>0.5).astype(int); train_precision=precision_score(all_train_targets,train_preds,zero_division=0); train_recall=recall_score(all_train_targets,train_preds,zero_division=0); train_f1=f1_score(all_train_targets,train_preds,zero_division=0)
                if len(np.unique(all_train_targets))>1: train_auc=roc_auc_score(all_train_targets,all_train_preds_probs)
            except Exception as e: logging.error(f"Calc train metrics err: {e}")

        # --- Validation Phase ---
        model.eval(); val_loss_sum=0.0; all_val_targets=[]; all_val_preds_probs=[]; processed_val_samples=0
        with torch.no_grad():
             for i, batch in enumerate(val_loader):
                  if batch is None: continue
                  try:
                       batch=batch.to(DEVICE);
                       if batch.x is None or batch.edge_index is None or batch.y is None or batch.batch is None or batch.x.nelement()==0: continue
                       edge_attr=batch.edge_attr if hasattr(batch,'edge_attr') else None
                       logits=model(batch.x, batch.edge_index, edge_attr=edge_attr, batch=batch.batch)
                       target=batch.y.view_as(logits).float(); loss=criterion(logits, target)
                       if not torch.isnan(loss) and not torch.isinf(loss):
                           val_loss_sum+=loss.item()*target.size(0); processed_val_samples+=target.size(0)
                           all_val_targets.extend(batch.y.squeeze().cpu().tolist())
                           all_val_preds_probs.extend(torch.sigmoid(logits.squeeze()).cpu().tolist())
                  except Exception as e: logging.error(f"E{epoch+1} Val Batch {i} Err: {e}"); continue
        val_loss_avg=val_loss_sum/processed_val_samples if processed_val_samples > 0 else float('inf')
        # Tính Val Metrics
        val_precision, val_recall, val_f1, val_auc = 0.0, 0.0, 0.0, 0.0
        if all_val_targets:
             try:
                 val_preds=(np.array(all_val_preds_probs)>0.5).astype(int); val_precision=precision_score(all_val_targets,val_preds,zero_division=0); val_recall=recall_score(all_val_targets,val_preds,zero_division=0); val_f1=f1_score(all_val_targets,val_preds,zero_division=0)
                 if len(np.unique(all_val_targets))>1: val_auc=roc_auc_score(all_val_targets,all_val_preds_probs)
             except Exception as e: logging.error(f"Calc val metrics err: {e}")

        # --- Logging ---
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")
        logging.info(f"          Train P/R/F1/AUC: {train_precision:.4f}/{train_recall:.4f}/{train_f1:.4f}/{train_auc:.4f}")
        logging.info(f"          Val P/R/F1/AUC:   {val_precision:.4f}/{val_recall:.4f}/{val_f1:.4f}/{val_auc:.4f}")

        # --- Early Stopping & Save Best Model (Dựa trên Val F1) ---
        current_metric = val_f1 if not np.isnan(val_f1) else -1.0
        if epoch == 0 and best_val_metric == -1.0: best_val_metric = current_metric # Khởi tạo ở epoch đầu
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            try:
                torch.save(model.state_dict(), MODEL_SAVE_PATH) # <<< LƯU VÀO PATH CUỐI CÙNG >>>
                logging.info(f"*** Best FINAL GNN model saved (Epoch {epoch+1}) *** -> Val F1: {best_val_metric:.4f}")
            except Exception as save_e: logging.error(f"Error saving FINAL GNN model: {save_e}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logging.info(f"Val F1 did not improve for {no_improve_epochs} epoch(s). Best F1: {best_val_metric:.4f}")
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE: # <<< DÙNG PATIENCE CUỐI CÙNG >>>
                logging.info(f"FINAL GNN training early stopping triggered at epoch {epoch+1}.")
                break

        # --- Step Scheduler ---
        metric_for_scheduler = current_metric if current_metric > 0 else (-val_loss_avg if val_loss_avg != float('inf') else 0)
        scheduler.step(metric_for_scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        logging.debug(f"Epoch {epoch+1} - LR: {current_lr:.7f} - Scheduler metric: {metric_for_scheduler:.4f}")


    logging.info("--- FINAL GNN Training Finished ---")
    logging.info(f"Best validation F1 score achieved: {best_val_metric:.4f}")
    logging.info(f"Final model saved to: {MODEL_SAVE_PATH}")
    logging.info(f"Associated scaler saved to: {SCALER_SAVE_PATH}")