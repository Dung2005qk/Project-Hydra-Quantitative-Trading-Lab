import torch
import torch.nn as nn
import torch.nn.functional as F # Thêm F
import torch.optim as optim
# <<< THÊM IMPORT >>>
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import scheduler
# <<< SỬA IMPORT DATASET/SUBSET >>>
from torch.utils.data import Dataset, Subset, WeightedRandomSampler # Import Sampler
import pandas as pd
import numpy as np
import joblib
import os
import logging
from pathlib import Path
import traceback # Import traceback
from sklearn.preprocessing import StandardScaler
# <<< THÊM IMPORT METRICS và roc_auc_score >>>
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from typing import List, Tuple, Optional
# <<< THÊM IMPORT OPTUNA >>>
import optuna
import sys
import gc # Import garbage collector
print("--- Imports Complete ---")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# <<< SỬ DỤNG TORCH GEOMETRIC >>>
try:
    from torch_geometric.nn import GATConv as GraphConv # Dùng GAT như đề xuất
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader # Dùng DataLoader của torch_geometric
    from torch_geometric.nn import global_mean_pool, global_max_pool # Import pooling
    _torch_geometric_available = True
    logging.info("Successfully imported torch_geometric.")
except ImportError:
    logging.error("torch_geometric is required. Install it (pip install torch_geometric). Falling back to dummy structure.")
    _torch_geometric_available = False
    # --- Định nghĩa lớp giả (Cải thiện lớp giả) ---
    class GraphConv:
        def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, edge_dim=None): self.out_channels = out_channels * heads if concat else out_channels; self.edge_dim=edge_dim
        def __call__(self, x, edge_index, edge_attr=None): return torch.zeros(x.shape[0], self.out_channels, device=x.device)
    class Data:
        def __init__(self, x=None, edge_index=None, y=None, edge_attr=None): self.x=x; self.edge_index=edge_index; self.y=y; self.edge_attr=edge_attr; self.num_nodes = x.shape[0] if x is not None else 0
        def to(self, device):
            for key, item in self.__dict__.items():
                if torch.is_tensor(item):
                    try: self.__dict__[key] = item.to(device)
                    except Exception as e: logging.error(f"Error moving tensor '{key}' to {device}: {e}")
            return self
        def __cat_dim__(self, key, value):
             if key == 'edge_index': return 1
             elif key in ['edge_attr', 'x', 'y']: return 0
             else: return 0
    class Batch(Data):
        @classmethod
        def from_data_list(cls, data_list):
             valid_data_list = [d for d in data_list if d is not None and hasattr(d,'x') and d.x is not None and hasattr(d,'edge_index') and d.edge_index is not None and hasattr(d,'y') and d.y is not None and d.x.nelement() > 0]
             if not valid_data_list: return None
             try:
                 keys = list(valid_data_list[0].keys) # Lấy keys từ Data object thật
                 batch = cls(); batch.__num_graphs__ = len(valid_data_list)
                 for key in keys:
                     if key not in ['batch', 'ptr']: batch[key] = []
                 batch.batch = []
                 cumsum_node = 0
                 for i, data in enumerate(valid_data_list):
                     num_nodes = data.num_nodes; batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                     for key, item in data:
                         if key not in ['batch', 'ptr', 'num_nodes']:
                             if torch.is_tensor(item) and key == 'edge_index': item = item + cumsum_node
                             if key in batch: batch[key].append(item)
                     cumsum_node += num_nodes
                 for key in keys:
                     if key not in ['batch', 'ptr', 'num_nodes']:
                         items = batch[key]
                         if isinstance(items, list) and items:
                             if torch.is_tensor(items[0]):
                                 try: batch[key] = torch.cat(items, dim=valid_data_list[0].__cat_dim__(key, items[0]))
                                 except Exception as cat_e: logging.error(f"Concat error key {key}: {cat_e}"); batch[key]=None
                             elif isinstance(items[0], (int, float)): # Handle scalar attributes if any
                                 batch[key] = torch.tensor(items)
                             # else: keep as list if not tensors or numbers
                         elif not items: batch[key]=None
                 batch.batch = torch.cat(batch.batch, dim=0) if batch.batch else None
                 # Check edge_attr again
                 if hasattr(batch,'edge_attr') and batch.edge_attr is not None and hasattr(batch,'edge_index') and batch.edge_index is not None and batch.edge_attr.shape[0] != batch.edge_index.shape[1]: batch.edge_attr = None
                 return batch
             except Exception as e: logging.error(f"Dummy Batch.from_data_list err: {e}"); return None
    # DataLoader chuẩn
    from torch.utils.data import DataLoader as StandardDataLoader

# --- Import IntelligentStopSystem và CONFIG ---
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


# --- Device Setup ---
if torch.cuda.is_available(): DEVICE = torch.device("cuda:0"); torch.cuda.set_device(DEVICE)
else: DEVICE = torch.device("cpu")
logging.info(f"Explicitly set device: {DEVICE}")

# --- Paths Setup ---
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or 'KAGGLE_WORKING_DIR' in os.environ
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
if IS_KAGGLE and KAGGLE_INPUT_DIR.exists():
    DATASET_NAME = "datacry" # <<< THAY TÊN DATASET >>>
    DATA_DIR = KAGGLE_INPUT_DIR / DATASET_NAME
    MODEL_SAVE_DIR = KAGGLE_OUTPUT_DIR / "optuna_gnn_results" # Thư mục output riêng cho Optuna
    if not DATA_DIR.exists(): exit(f"Dataset '{DATA_DIR}' not found.")
else:
    try: SCRIPT_DIR = Path(__file__).parent
    except NameError: SCRIPT_DIR = Path.cwd()
    DATA_DIR = SCRIPT_DIR
    MODEL_SAVE_DIR = SCRIPT_DIR / "optuna_gnn_results"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# <<< LƯU STUDY DATABASE >>>
OPTUNA_DB_PATH = MODEL_SAVE_DIR / "gnn_optuna_study.db"
# Scaler path (load hoặc fit)
SCALER_SAVE_PATH = MODEL_SAVE_DIR / "iss_gnn_scaler_for_optuna.pkl" # Đặt tên riêng

# --- Data & Model Parameters ---
# ... (Lấy SYMBOLS_TO_TRAIN, TIMEFRANE, NODE_FEATURES, TARGET_FEATURE, etc. từ CONFIG/default) ...
SYMBOLS_TO_TRAIN = CONFIG.get("symbols", ["BTC/USDT", "ETH/USDT"])
TIMEFRANE = "15m"
GRAPH_LOOKBACK = 30 # Giữ cố định trong Optuna này
try: temp_iss=IntelligentStopSystem(); NODE_FEATURES=temp_iss.gnn_features; NODE_DIM=len(NODE_FEATURES)
except Exception: NODE_FEATURES=["close","RSI","ATR","volatility","volume"]; NODE_DIM=5
TARGET_FEATURE="volatility"
REQUIRED_COLS_LOAD = list(set(NODE_FEATURES + [TARGET_FEATURE]))
VOL_SPIKE_THRESHOLD = 1.4
EDGE_DIM = 2 # Đã thêm edge features

# --- Optuna Parameters ---
VALIDATION_SPLIT = 0.15
MIN_SAMPLES_FOR_TRAINING = 1000
N_TRIALS = 0 # <<< Tăng số trials >>>
N_EPOCHS_PER_TRIAL = 35 # <<< Tăng số epoch/trial >>>
EARLY_STOPPING_PATIENCE_TRIAL = 5 # <<< Patience cho early stopping trong trial >>>
PRUNER_N_STARTUP_TRIALS = 10 # <<< Chạy 10 trial đầu đủ trước khi prune >>>
PRUNER_N_WARMUP_STEPS = 8    # <<< Chạy 8 epoch trước khi prune trong 1 trial >>>
# <<< Metric kết hợp để tối ưu >>>
METRIC_WEIGHT_F1 = 0.7
METRIC_WEIGHT_AUC = 0.3
STUDY_EARLY_STOPPING_PATIENCE = 15

# --- Helper Functions ---
# load_and_prepare_gnn_data
def load_and_prepare_gnn_data(symbol: str, required_cols: List[str]) -> Optional[Tuple[str, pd.DataFrame]]:
    # ... (Giữ nguyên logic) ...
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
# build_market_graph (đã sửa với edge_attr)
def build_market_graph(df_slice_scaled: np.ndarray, lookback: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Xây dựng input đồ thị bao gồm node features đã scale, edge index,
    và edge attributes (thay đổi giá/volume đã scale).
    """
    # <<< Đảm bảo NODE_DIM được định nghĩa hoặc lấy từ shape >>>
    current_node_dim = df_slice_scaled.shape[1] # Lấy dim từ dữ liệu thực tế

    if len(df_slice_scaled) != lookback or df_slice_scaled.ndim != 2 or current_node_dim == 0: # Kiểm tra dim > 0
        logging.error(f"build_market_graph: Invalid input shape {df_slice_scaled.shape}, expected ({lookback}, >0)")
        return None, None, None

    node_features = torch.tensor(df_slice_scaled, dtype=torch.float32)
    # Khởi tạo rỗng
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attrs_list = [] # Dùng list để append
    full_edge_list = []

    try:
        close_idx = NODE_FEATURES.index('close')
        vol_idx = NODE_FEATURES.index('volume')
        has_indices = True
    except ValueError:
        logging.warning("build_market_graph: Cannot find 'close' or 'volume' index in NODE_FEATURES. Edge attributes will be None.")
        has_indices = False

    if lookback >= 2:
        for i in range(lookback - 1): # Vòng lặp sẽ tự dừng đúng chỗ
            src, dst = i, i + 1
            full_edge_list.append((src, dst)) # Cạnh thuận
            full_edge_list.append((dst, src)) # Cạnh nghịch

            if has_indices: # Chỉ tính attr nếu có index
                try: # Thêm try-except cho .item() phòng trường hợp tensor trống/lỗi
                     price_change = abs(node_features[dst, close_idx].item() - node_features[src, close_idx].item())
                     vol_change = abs(node_features[dst, vol_idx].item() - node_features[src, vol_idx].item())
                     attr = [price_change, vol_change]
                     edge_attrs_list.extend([attr, attr]) # Thêm attr cho cả 2 chiều cạnh
                except IndexError:
                     logging.error(f"IndexError in build_market_graph attr calculation at i={i}. Node features shape: {node_features.shape}")
                     # Nếu lỗi ở đây, edge_attr sẽ không khớp edge_index -> cần xử lý
                     has_indices = False # Đánh dấu là không thể tính attr nữa
                     edge_attrs_list = [] # Reset attr list
                     # Không cần break, vẫn tạo edge_index
                except Exception as e:
                     logging.error(f"Error calculating edge attrs at i={i}: {e}")
                     has_indices = False
                     edge_attrs_list = []


        if full_edge_list:
            edge_index = torch.tensor(full_edge_list, dtype=torch.long).t().contiguous()

    # Tạo tensor edge_attr chỉ khi tính toán thành công và có cạnh
    edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float32) if has_indices and edge_attrs_list and edge_index.shape[1] > 0 else None

    # Kiểm tra shape lần cuối (quan trọng)
    if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]:
        logging.error(f"Final check: Edge attr shape {edge_attr.shape} mismatch edge index {edge_index.shape}. Setting edge_attr to None.")
        edge_attr = None
    elif edge_index.shape[1] == 0: # Đảm bảo edge_attr là None nếu không có cạnh
        edge_attr = None

    return node_features, edge_index, edge_attr

class EarlyStoppingCallback:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self._patience = patience
        self._min_delta = min_delta
        self._no_improvement_count = 0
        self._best_value = -float('inf') # Giả định đang maximize
        logging.info(f"EarlyStoppingCallback initialized with patience={patience}, min_delta={min_delta}")

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Chỉ kiểm tra các trial đã hoàn thành thành công
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        current_value = trial.value
        if current_value is None: # Bỏ qua trial không có giá trị
             return

        # Lấy giá trị tốt nhất hiện tại từ study (an toàn hơn)
        # study.best_value có thể là None nếu chưa có trial nào COMPLETE
        best_value_so_far = study.best_value if study.best_value is not None else -float('inf')

        # So sánh giá trị hiện tại với giá trị tốt nhất đã thấy TRƯỚC trial này
        # (Hoặc có thể so sánh trực tiếp với study.best_value)
        # if current_value > self._best_value + self._min_delta:
        if best_value_so_far > self._best_value + self._min_delta:
            logging.debug(f"EarlyStoppingCallback: Trial {trial.number} improved! New best: {best_value_so_far:.4f} (Old: {self._best_value:.4f})")
            self._best_value = best_value_so_far # Cập nhật best_value của callback
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
            logging.info(f"EarlyStoppingCallback: No significant improvement for {self._no_improvement_count} consecutive trial(s). Best value: {self._best_value:.4f}")

        if self._no_improvement_count >= self._patience:
            logging.warning(f"Early stopping study '{study.study_name}' at trial {trial.number} due to no improvement for {self._patience} trials.")
            # Dừng study bằng cách raise ngoại lệ đặc biệt
            study.stop()

# --- GNN Model (Kiến trúc nâng cao) ---
# <<< Sửa __init__ để nhận num_layers >>>
class GraphNeuralNetPyG_Advanced(nn.Module):
    def __init__(self, node_dim, num_layers=3, edge_dim=EDGE_DIM, hidden_dim=64, out_dim=1, dropout=0.4, heads=4): # Thêm num_layers
        super().__init__()
        if not _torch_geometric_available: raise ImportError("torch_geometric required")
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim if isinstance(edge_dim, int) and edge_dim > 0 else None
        intermediate_dim = hidden_dim // heads
        self.num_layers = num_layers # Lưu số lớp

        self.input_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skips = nn.ModuleList()

        in_channels = node_dim
        # Tạo các lớp động
        for i in range(num_layers):
            out_channels_conv = hidden_dim if i < num_layers - 1 else hidden_dim # Lớp cuối cùng có thể khác nếu heads=1
            current_heads = heads if i < num_layers - 1 else 1 # Lớp cuối dùng 1 head
            concat_final = True if i < num_layers - 1 else False # Không concat ở lớp cuối nếu head=1
            out_channels_actual = out_channels_conv # Dim output thực tế của lớp conv (sau concat hoặc không)

            # Thêm skip connection (Linear để chiếu lên dim của lớp conv hiện tại nếu cần)
            if i == 0: # Lớp đầu tiên
                 self.skips.append(nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity())
            elif in_channels != hidden_dim: # Các lớp giữa nếu dim thay đổi
                 self.skips.append(nn.Linear(in_channels, hidden_dim))
            else: # Dim không đổi
                 self.skips.append(nn.Identity())

            self.convs.append(GraphConv(in_channels, hidden_dim // current_heads if current_heads > 0 else hidden_dim, heads=current_heads, dropout=dropout, edge_dim=self.edge_dim, concat=concat_final))
            self.bns.append(nn.BatchNorm1d(out_channels_actual))
            in_channels = out_channels_actual # Cập nhật input dim cho lớp sau

        self.dropout_layer = nn.Dropout(p=dropout) # Một lớp dropout dùng chung

        # Predictor nhận input từ pooling kép
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout/2), nn.Linear(hidden_dim, out_dim))
        logging.info(f"Initialized Dyn GNN (GAT) Layers={num_layers}, Hidden={hidden_dim}, Heads={heads}, Dropout={dropout}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]=None, batch: Optional[torch.Tensor]=None) -> torch.Tensor:
        if x is None or edge_index is None or x.nelement()==0:
             batch_size = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 1
             return torch.zeros((batch_size, self.predictor[-1].out_features), device=DEVICE)
        try:
            use_edge_attr = self.edge_dim is not None and edge_attr is not None
            if use_edge_attr and edge_attr.shape[1] != self.edge_dim: use_edge_attr = False

            # --- Lặp qua các lớp động ---
            identity = x # Giữ input gốc cho skip đầu tiên (nếu cần)
            for i in range(self.num_layers):
                 layer_identity = self.skips[i](identity) # Chiếu identity phù hợp với output lớp i
                 x_in = x # Input cho lớp conv hiện tại
                 x = self.convs[i](x_in, edge_index, edge_attr=edge_attr) if use_edge_attr else self.convs[i](x_in, edge_index)
                 x = self.bns[i](x)
                 x = F.elu(x)
                 x = self.dropout_layer(x) # Dùng cùng lớp dropout
                 # Residual connection
                 try:
                      if x.shape == layer_identity.shape: x = x + layer_identity
                      else: logging.warning(f"Skip connection {i+1} shape mismatch. Skipping add.")
                 except RuntimeError as res_e: logging.error(f"RuntimeError Residual {i+1}: {res_e}. Skipping.")
                 identity = x # Lưu output của lớp này làm identity cho lớp sau (nếu skip dùng Identity)

            # Pooling và Predictor
            if batch is None: # Single graph
                 graph_embedding = torch.mean(x, dim=0, keepdim=True)
                 graph_embedding = torch.cat([graph_embedding, graph_embedding], dim=1)
            else: # Batched graph
                 mean_pool = global_mean_pool(x, batch); max_pool = global_max_pool(x, batch)
                 graph_embedding = torch.cat([mean_pool, max_pool], dim=1)

            out_logit = self.predictor(graph_embedding)
            return out_logit
        except Exception as e:
             logging.error(f"GNN forward error: {e}", exc_info=True)
             batch_size = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 1
             return torch.zeros((batch_size, 1), device=DEVICE)


# --- Focal Loss Class (Giữ nguyên) ---
class FocalLoss(nn.Module):
    # ... (Giữ nguyên) ...
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        # Kiểm tra kiểu và giá trị đầu vào
        if not isinstance(alpha, float) or not (0 < alpha < 1):
             # logging.warning(f"Invalid alpha value {alpha}. Using default 0.25.") # Có thể log nếu muốn
             alpha = 0.25
        if not isinstance(gamma, (float, int)) or gamma < 0:
             # logging.warning(f"Invalid gamma value {gamma}. Using default 2.0.")
             gamma = 2.0
        if reduction not in ['mean', 'sum', 'none']:
             # logging.warning(f"Invalid reduction type {reduction}. Using default 'mean'.")
             reduction = 'mean'

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, inputs, targets):
        targets = targets.type_as(inputs)
        BCE_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# --- PyTorch Geometric Dataset (Giữ nguyên) ---
class GraphDatasetPyG(Dataset):
    # ... (Giữ nguyên __init__, _calculate_*, __len__, __getitem__) ...
    def __init__(
        self,
        df: pd.DataFrame,
        node_features_list: List[str],
        target_col: str,
        graph_lookback: int,
        vol_spike_threshold: float,
        scaler: StandardScaler
    ):
        self.df = df
        self.node_features_list = node_features_list
        self.num_node_features = len(node_features_list)
        self.target_col = target_col
        self.graph_lookback = graph_lookback
        self.vol_spike_threshold = vol_spike_threshold
        self.scaler = scaler

        self.targets, self.valid_indices = self._calculate_targets_and_indices()
        self.pos_weight_value = self._calculate_pos_weight()

        assert len(self.targets) == len(self.valid_indices), "Target and index length mismatch!"

        if not self.valid_indices.size > 0:
            logging.warning("No valid indices.")

    def _calculate_targets_and_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        targets = []
        valid_indices = []
        vol_values = self.df[self.target_col].values
        num_rows = len(self.df)

        for i in range(num_rows - self.graph_lookback):
            end_idx = i + self.graph_lookback
            target_idx = end_idx

            current_vol = vol_values[end_idx - 1]
            next_vol = vol_values[target_idx]

            if (
                pd.notna(next_vol)
                and pd.notna(current_vol)
                and current_vol > 1e-9
                and next_vol > current_vol * self.vol_spike_threshold
            ):
                target = 1.0
            else:
                target = 0.0

            targets.append(target)
            valid_indices.append(i)

        return np.array(targets), np.array(valid_indices, dtype=int)

    def _calculate_pos_weight(self) -> Optional[float]:
        # Kiểm tra nếu targets rỗng ngay từ đầu
        if len(self.targets) == 0:
            logging.warning("Cannot calculate pos_weight: targets array is empty.")
            return None # Trả về None nếu rỗng

        # <<< DI CHUYỂN TÍNH TOÁN RA NGOÀI KHỐI IF >>>
        num_pos = np.sum(self.targets == 1)
        num_neg = np.sum(self.targets == 0) # Hoặc len(self.targets) - num_pos
        # Tính pos_w, xử lý trường hợp num_pos = 0 để tránh chia cho 0
        pos_w = num_neg / num_pos if num_pos > 0 else 1.0

        # <<< LOGGING DÙNG BIẾN ĐÃ TÍNH >>>
        if num_pos > 0:
             logging.info(f"Pos weight: {pos_w:.2f} ({num_pos} pos / {len(self.targets)} total)")
        else:
             logging.warning("No positive samples found for pos_weight calculation. Using default weight 1.0.")

        # Chỉ trả về giá trị nếu có mẫu dương, nếu không có thể dùng trọng số mặc định hoặc None
        # Quyết định hiện tại là trả về giá trị (kể cả khi pos_w = 1.0 do không có mẫu dương)
        # Hoặc bạn có thể muốn trả về None nếu num_pos == 0:
        return pos_w if num_pos > 0 else None

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        # 1. Kiểm tra index idx có hợp lệ cho valid_indices không
        if idx >= len(self.valid_indices):
            logging.error(f"GraphDatasetPyG __getitem__: Index {idx} out of bounds for valid_indices (len: {len(self.valid_indices)})")
            return None # Trả về None nếu idx không hợp lệ

        # 2. Lấy index bắt đầu thực tế và tính index kết thúc
        actual_start_idx = self.valid_indices[idx]
        end_idx = actual_start_idx + self.graph_lookback

        # 3. Kiểm tra xem slice có nằm trong phạm vi dataframe gốc không
        if end_idx > len(self.df):
            logging.error(f"GraphDatasetPyG __getitem__: Calculated end_idx {end_idx} exceeds DataFrame length {len(self.df)} (start_idx: {actual_start_idx}, lookback: {self.graph_lookback})")
            return None # Trả về None nếu slice vượt quá giới hạn df

        # 4. Lấy slice dữ liệu chưa scale
        # Thêm kiểm tra actual_start_idx >= 0 đề phòng trường hợp lạ
        if actual_start_idx < 0:
             logging.error(f"GraphDatasetPyG __getitem__: Invalid actual_start_idx {actual_start_idx}")
             return None
        df_slice_unscaled = self.df.iloc[actual_start_idx:end_idx][self.node_features_list]

        # 5. Scale slice
        try:
            scaled_slice_values = self.scaler.transform(df_slice_unscaled.values)
        except Exception as scale_e:
            logging.error(f"GraphDatasetPyG __getitem__: Scaling error at index {idx} (actual_start_idx {actual_start_idx}): {scale_e}")
            return None

        # 6. Xây dựng đồ thị
        node_features, edge_index, edge_attr = build_market_graph(scaled_slice_values, self.graph_lookback)
        if node_features is None or edge_index is None:
            logging.warning(f"GraphDatasetPyG __getitem__: build_market_graph returned None for index {idx} (actual_start_idx {actual_start_idx}).")
            return None

        # 7. Lấy target
        target = torch.tensor([self.targets[idx]], dtype=torch.float32)

        # 8. Trả về Data object
        return Data(x=node_features, edge_index=edge_index, y=target, edge_attr=edge_attr)


# --- Time-based Split Function (Giữ nguyên) ---
def time_based_split(dataset: Dataset, val_ratio: float = 0.15) -> Tuple[Subset, Subset]:
    # ... (Giữ nguyên) ...
    n = len(dataset); val_size = int(n * val_ratio); train_size = n - val_size
    if train_size <= 0 or val_size <= 0: raise ValueError(f"Cannot split dataset size {n}")
    train_indices = list(range(train_size)); val_indices = list(range(train_size, n))
    logging.info(f"Time-based split: Train={train_size}, Val={val_size}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# --- Collate Function (Giữ nguyên) ---
def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    if not batch: return None
    try: return Batch.from_data_list(batch)
    except Exception as collate_e: logging.error(f"Collate error: {collate_e}"); return None


# --- Hàm Objective cho Optuna (Đã Cải Tiến) ---
import time # Thêm import time
import gc # Import garbage collector

# ... (các import khác và setup giữ nguyên) ...

def objective(trial: optuna.Trial, train_dataset: Subset, val_dataset: Subset, full_dataset: Dataset) -> float:
    # <<< Log ngay khi vào hàm >>>
    logging.info(f"++++++++++ ENTERING objective for Trial {trial.number} ++++++++++")
    logging.info(f"Trial {trial.number} PARAMS: {trial.params}")
    trial_start_time = time.time() # Ghi lại thời gian bắt đầu trial

    # --- Bọc toàn bộ logic trial trong try...except để bắt lỗi sớm ---
    try:
        # --- 1. Đề xuất Hyperparameters ---
        logging.debug(f"Trial {trial.number}: Suggesting hyperparameters...")
        cfg = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 192]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.6),
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
            "focal_alpha": trial.suggest_float("focal_alpha", 0.1, 0.5),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.5, 4.0),
            "gnn_layers": trial.suggest_int("gnn_layers", 2, 4),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
        }
        logging.info(f"Trial {trial.number} - Hyperparameters: {cfg}")

        # --- 2. Tạo Weighted Sampler ---
        logging.debug(f"Trial {trial.number}: Creating WeightedRandomSampler...")
        train_sampler = None
        try:
            train_indices = train_dataset.indices
            train_targets = full_dataset.targets[train_indices]
            if len(train_targets) > 0:
                 class_counts = np.bincount(train_targets.astype(int))
                 if len(class_counts) >= 2 and class_counts[1] > 0:
                      class_weights=1./class_counts
                      sample_weights=np.array([class_weights[t] for t in train_targets.astype(int)])
                      train_sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True)
                      logging.debug(f"Trial {trial.number}: WeightedRandomSampler created.")
                 else:
                     logging.warning(f"Trial {trial.number}: Not enough classes or positive samples for sampler.")
            else:
                logging.warning(f"Trial {trial.number}: Training dataset appears empty for sampler calculation.")
        except IndexError as sampler_idx_e:
             logging.error(f"Trial {trial.number}: IndexError during sampler creation - likely issue with train_indices or full_dataset.targets access: {sampler_idx_e}", exc_info=True)
             raise # Dừng trial nếu lỗi sampler nghiêm trọng
        except Exception as sampler_e:
             logging.warning(f"Trial {trial.number}: Non-critical sampler error {sampler_e}")

        # --- 3. Tạo DataLoader ---
        logging.debug(f"Trial {trial.number}: Creating DataLoaders with Batch Size {cfg['batch_size']}...")
        if not _torch_geometric_available: raise RuntimeError("torch_geometric required.")
        try:
            # <<< Thêm persistent_workers=False và pin_memory >>>
            train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], sampler=train_sampler,
                                      num_workers=0, collate_fn=collate_fn, shuffle=(train_sampler is None),
                                      pin_memory=torch.cuda.is_available(), persistent_workers=False)
            val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                                    num_workers=0, collate_fn=collate_fn,
                                    pin_memory=torch.cuda.is_available(), persistent_workers=False)
            logging.debug(f"Trial {trial.number}: DataLoaders CREATED.")
        except Exception as loader_e:
            logging.error(f"Trial {trial.number}: DataLoader CRITICAL error {loader_e}", exc_info=True)
            raise # Dừng trial

        # --- 4. Khởi tạo Model ---
        logging.debug(f"Trial {trial.number}: Initializing Model (Layers={cfg['gnn_layers']}, Hidden={cfg['hidden_dim']}, Heads={cfg['heads']})...")
        try:
            model = GraphNeuralNetPyG_Advanced(
                node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_layers=cfg['gnn_layers'],
                hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'], heads=cfg['heads']
            )
            logging.debug(f"Trial {trial.number}: Model initialized. Moving to device {DEVICE}...")
            model.to(DEVICE)
            logging.debug(f"Trial {trial.number}: Model moved to device.")
        except Exception as model_init_e:
             logging.error(f"Trial {trial.number}: Model Init CRITICAL error {model_init_e}", exc_info=True)
             raise

        # --- 5. Khởi tạo Optimizer, Loss, Scheduler ---
        logging.debug(f"Trial {trial.number}: Initializing Optimizer, Loss, Scheduler...")
        try:
            optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)
            criterion = FocalLoss(alpha=cfg['focal_alpha'], gamma=cfg['focal_gamma'])
            logging.debug(f"Trial {trial.number}: Optimizer, Loss, Scheduler INITIALIZED.")
        except Exception as optim_init_e:
             logging.error(f"Trial {trial.number}: Optimizer/Loss/Scheduler Init CRITICAL error {optim_init_e}", exc_info=True)
             raise

        logging.info(f"Trial {trial.number}: Initialization complete. Time: {time.time() - trial_start_time:.2f}s. Starting epochs...")

        # --- 6. Huấn luyện Ngắn Gọn ---
        best_trial_val_metric = -1.0
        no_improve_trial_epochs = 0

        for epoch in range(N_EPOCHS_PER_TRIAL):
            epoch_start_time = time.time() # <<< Thời gian bắt đầu epoch
            logging.debug(f"Trial {trial.number} --- Starting Epoch {epoch+1}/{N_EPOCHS_PER_TRIAL} ---")

            # --- Training Phase ---
            model.train()
            train_loss_sum = 0.0; processed_train_batches = 0; processed_train_samples = 0
            all_train_targets = []; all_train_preds_probs = []
            logging.debug(f"Trial {trial.number} Epoch {epoch+1}: Starting training batches...")
            for i, batch in enumerate(train_loader):
                logging.debug(f"T{trial.number} E{epoch+1} - Processing Train Batch {i}")
                if batch is None: logging.warning(f"T{trial.number} E{epoch+1} Train Batch {i}: Received None batch."); continue
                try:
                    batch_start_time = time.time() # <<< Thời gian bắt đầu batch
                    batch = batch.to(DEVICE, non_blocking=True) # <<< Thêm non_blocking
                    logging.debug(f"T{trial.number} E{epoch+1} B{i}: Batch moved to device. Time: {time.time() - batch_start_time:.4f}s")

                    if batch.x is None or batch.edge_index is None or batch.y is None or batch.batch is None or batch.x.nelement()==0:
                        logging.warning(f"T{trial.number} E{epoch+1} Train Batch {i}: Invalid batch attributes after moving to device."); continue

                    optimizer.zero_grad()
                    edge_attr = batch.edge_attr if hasattr(batch,'edge_attr') else None

                    forward_start_time = time.time() # <<< Thời gian forward
                    logits = model(batch.x, batch.edge_index, edge_attr=edge_attr, batch=batch.batch)
                    logging.debug(f"T{trial.number} E{epoch+1} B{i}: Forward pass done. Time: {time.time() - forward_start_time:.4f}s")

                    target = batch.y.view_as(logits).float()
                    loss = criterion(logits, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"T{trial.number} E{epoch+1} Train Batch {i}: NaN/Inf loss detected. Skipping backward/step.")
                        continue

                    backward_start_time = time.time() # <<< Thời gian backward
                    loss.backward()
                    logging.debug(f"T{trial.number} E{epoch+1} B{i}: Backward pass done. Time: {time.time() - backward_start_time:.4f}s")

                    clip_start_time = time.time() # <<< Thời gian clip
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['grad_clip'])
                    logging.debug(f"T{trial.number} E{epoch+1} B{i}: Grad clip done. Time: {time.time() - clip_start_time:.4f}s")

                    step_start_time = time.time() # <<< Thời gian step
                    optimizer.step()
                    logging.debug(f"T{trial.number} E{epoch+1} B{i}: Optimizer step done. Time: {time.time() - step_start_time:.4f}s")

                    current_batch_samples = target.size(0)
                    train_loss_sum += loss.item() * current_batch_samples
                    processed_train_samples += current_batch_samples
                    processed_train_batches += 1

                    # Thu thập sau khi xử lý xong batch
                    all_train_targets.extend(batch.y.squeeze().cpu().tolist())
                    all_train_preds_probs.extend(torch.sigmoid(logits.squeeze()).cpu().tolist())
                    logging.debug(f"T{trial.number} E{epoch+1} B{i}: Batch processing complete. Total time: {time.time() - batch_start_time:.4f}s")


                except RuntimeError as e: # <<< Bắt lỗi Runtime cụ thể (ví dụ: OOM) >>>
                     if "out of memory" in str(e).lower():
                         logging.error(f"T{trial.number} E{epoch+1} Train Batch {i}: CUDA OUT OF MEMORY", exc_info=True)
                         # <<< Giải phóng bộ nhớ và thử lại hoặc bỏ qua >>>
                         gc.collect()
                         torch.cuda.empty_cache()
                         # Có thể raise lỗi để dừng trial nếu OOM liên tục
                         # raise e # Dừng trial nếu OOM
                         continue # Bỏ qua batch này
                     else:
                         logging.error(f"T{trial.number} E{epoch+1} Train Batch {i} Runtime Err: {e}", exc_info=True)
                         continue # Bỏ qua batch lỗi khác
                except Exception as e:
                     logging.error(f"T{trial.number} E{epoch+1} Train Batch {i} UNEXPECTED Err: {e}", exc_info=True)
                     continue # Bỏ qua batch lỗi không mong muốn
            logging.debug(f"Trial {trial.number} Epoch {epoch+1}: Finished training batches. Processed: {processed_train_batches}")

            # --- Validation Phase ---
            model.eval()
            val_loss_sum = 0.0; processed_val_batches = 0; processed_val_samples = 0
            all_val_targets = []; all_val_preds_probs = []
            logging.debug(f"Trial {trial.number} Epoch {epoch+1}: Starting validation batches...")
            with torch.no_grad():
                 for i, batch in enumerate(val_loader):
                      logging.debug(f"T{trial.number} E{epoch+1} - Processing Val Batch {i}")
                      if batch is None: logging.warning(f"T{trial.number} E{epoch+1} Val Batch {i}: Received None batch."); continue
                      try:
                           batch = batch.to(DEVICE, non_blocking=True)
                           if batch.x is None or batch.edge_index is None or batch.y is None or batch.batch is None or batch.x.nelement()==0:
                                logging.warning(f"T{trial.number} E{epoch+1} Val Batch {i}: Invalid batch attributes."); continue

                           edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
                           logits = model(batch.x, batch.edge_index, edge_attr=edge_attr, batch=batch.batch)
                           target = batch.y.view_as(logits).float()
                           loss = criterion(logits, target)

                           if not torch.isnan(loss) and not torch.isinf(loss):
                               current_batch_samples = target.size(0)
                               val_loss_sum += loss.item() * current_batch_samples
                               processed_val_samples += current_batch_samples
                               processed_val_batches += 1
                               all_val_targets.extend(batch.y.squeeze().cpu().tolist())
                               all_val_preds_probs.extend(torch.sigmoid(logits.squeeze()).cpu().tolist())
                           else:
                                logging.warning(f"T{trial.number} E{epoch+1} Val Batch {i}: NaN/Inf loss detected.")

                      except RuntimeError as e:
                           if "out of memory" in str(e).lower(): logging.error(f"T{trial.number} E{epoch+1} Val Batch {i}: CUDA OUT OF MEMORY", exc_info=True); gc.collect(); torch.cuda.empty_cache(); continue
                           else: logging.error(f"T{trial.number} E{epoch+1} Val Batch {i} Runtime Err: {e}", exc_info=True); continue
                      except Exception as e:
                           logging.error(f"T{trial.number} E{epoch+1} Val Batch {i} UNEXPECTED Err: {e}", exc_info=True); continue
            logging.debug(f"Trial {trial.number} Epoch {epoch+1}: Finished validation batches. Processed: {processed_val_batches}")


            # --- Tính toán Metrics và Logging ---
            val_loss_avg = val_loss_sum / processed_val_samples if processed_val_samples > 0 else float('inf')
            val_f1, val_auc = 0.0, 0.0
            if all_val_targets and processed_val_samples > 0: # <<< Thêm kiểm tra processed_val_samples >>>
                 try:
                     val_preds=(np.array(all_val_preds_probs) > 0.5).astype(int)
                     val_f1=f1_score(all_val_targets, val_preds, zero_division=0)
                     if len(np.unique(all_val_targets)) > 1:
                         val_auc=roc_auc_score(all_val_targets, all_val_preds_probs)
                     else:
                         val_auc = 0.0 # Hoặc np.nan
                 except Exception as e:
                     logging.error(f"T{trial.number} E{epoch+1} Calc Val Metrics Err: {e}")

            current_metric = METRIC_WEIGHT_F1 * val_f1 + METRIC_WEIGHT_AUC * val_auc
            epoch_duration = time.time() - epoch_start_time # <<< Thời gian chạy epoch
            logging.info(f"Trial {trial.number} Epoch {epoch+1}/{N_EPOCHS_PER_TRIAL} - Val Loss: {val_loss_avg:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Combined Metric: {current_metric:.4f}, Duration: {epoch_duration:.2f}s")

            # --- Pruning, Early Stopping, Scheduler Step ---
            try: # <<< Bọc trong try để tránh lỗi ảnh hưởng trial >>>
                 trial.report(current_metric, epoch)
                 if trial.should_prune():
                      logging.info(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                      # <<< QUAN TRỌNG: Raise TrialPruned để Optuna biết >>>
                      raise optuna.exceptions.TrialPruned()
            except Exception as report_e:
                 logging.error(f"T{trial.number} E{epoch+1} Error during Optuna report/prune: {report_e}", exc_info=True)


            # Early Stopping bên trong trial
            if current_metric > best_trial_val_metric :
                 best_trial_val_metric = current_metric
                 no_improve_trial_epochs = 0
            else:
                 no_improve_trial_epochs += 1
            if no_improve_trial_epochs >= EARLY_STOPPING_PATIENCE_TRIAL:
                 logging.info(f"Trial {trial.number} internal early stopping at epoch {epoch+1}.")
                 break # Dừng sớm trial này

            # Step the scheduler
            scheduler.step(current_metric)
            # <<< Giải phóng bộ nhớ GPU không cần thiết sau mỗi epoch >>>
            if torch.cuda.is_available():
                 gc.collect()
                 torch.cuda.empty_cache()

        # --- Kết thúc vòng lặp Epoch ---

        trial_duration = time.time() - trial_start_time
        logging.info(f"--- Finished Trial {trial.number} - Best Combined Metric: {best_trial_val_metric:.4f}, Duration: {trial_duration:.2f}s ---")
        return best_trial_val_metric if np.isfinite(best_trial_val_metric) else -1.0

    # <<< Xử lý lỗi tổng quát cho objective >>>
    except optuna.exceptions.TrialPruned as e:
         # Đã log bên trong, chỉ cần raise lại
         raise e
    except Exception as obj_fatal_e:
        logging.error(f"!!!!!!!!!! FATAL ERROR DURING Trial {trial.number} !!!!!!!!!!!")
        logging.error(f"Error type: {type(obj_fatal_e).__name__}")
        logging.error(f"Error message: {obj_fatal_e}", exc_info=True)
        # Trả về giá trị tệ để Optuna biết trial này lỗi
        return -1.0 # Hoặc raise lỗi tùy bạn muốn Optuna xử lý thế nào


# --- Main Script Execution ---
if __name__ == "__main__":
    # 1. Load Data -> loaded_data_list
    loaded_data_list: List[Tuple[str, pd.DataFrame]] = []
    required_gnn_cols = list(set(NODE_FEATURES + [TARGET_FEATURE])) # Cần cả target col
    logging.info(f"Loading data for symbols: {SYMBOLS_TO_TRAIN}")
    for symbol in SYMBOLS_TO_TRAIN:
        load_result = load_and_prepare_gnn_data(symbol, required_gnn_cols)
        if load_result:
            loaded_data_list.append(load_result)
        # Không cần else ở đây vì đã có log lỗi/warning bên trong hàm load

    if not loaded_data_list: # Kiểm tra sau khi lặp xong
        exit("No GNN data could be loaded or prepared for any symbol. Exiting.")

    all_df_list = [df for _, df in loaded_data_list] # Lấy list các DataFrame

    # 2. Fit Scaler -> scaler
    cols_to_scale = NODE_FEATURES # Chỉ scale node features
    logging.info(f"Preparing features for scaler fitting (Columns: {cols_to_scale})...")
    feature_dfs_to_concat = []
    for symbol, df in loaded_data_list: # Lấy cả symbol để log lỗi nếu cần
         if all(col in df.columns for col in cols_to_scale):
              feature_dfs_to_concat.append(df[cols_to_scale])
         else:
              logging.warning(f"Skipping dataframe for symbol '{symbol}' from scaler fitting due to missing columns: {set(cols_to_scale) - set(df.columns)}")
    if not feature_dfs_to_concat:
         exit("No valid dataframes with required features found for scaler fitting.")

    try:
         full_df_combined_features = pd.concat(feature_dfs_to_concat, ignore_index=True)
    except Exception as concat_e:
         logging.error(f"Error concatenating dataframes for scaler fitting: {concat_e}", exc_info=True)
         exit("Exiting due to concatenation error.")

    logging.info(f"Fitting scaler on combined feature data with shape: {full_df_combined_features.shape}")
    scaler = StandardScaler()
    # Kiểm tra NaN/Inf trước khi fit
    if np.any(np.isnan(full_df_combined_features.values)) or np.any(np.isinf(full_df_combined_features.values)):
         logging.error("NaN or Inf values detected in features before scaling. Check data loading/preparation.")
         exit("Cannot fit scaler on data with NaN/Inf.")
    try:
        scaler.fit(full_df_combined_features.values)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        logging.info(f"GNN Scaler (for node features) fitted and saved to {SCALER_SAVE_PATH}")
    except Exception as scaler_e:
         logging.error(f"Error fitting or saving scaler: {scaler_e}", exc_info=True)
         exit("Exiting due to scaler error.")

    # 3. Tạo Dataset -> full_dataset
    logging.info("Creating PyG Dataset...")
    try:
        # <<< Đảm bảo concat dùng đúng list và thứ tự cột >>>
        # Cần df gốc chứa cả feature và target
        original_dfs_to_concat = []
        for symbol, df in loaded_data_list: # Lấy df gốc từ loaded_data_list
            if all(col in df.columns for col in required_gnn_cols):
                original_dfs_to_concat.append(df[required_gnn_cols]) # Giữ đúng thứ tự cột
            else:
                 logging.warning(f"Skipping dataframe for symbol '{symbol}' from dataset creation due to missing columns: {set(required_gnn_cols) - set(df.columns)}")
        if not original_dfs_to_concat:
             exit("No valid dataframes with required columns for dataset creation.")

        full_original_df_combined = pd.concat(original_dfs_to_concat, ignore_index=True)
        logging.info(f"Combined original dataframe shape for dataset: {full_original_df_combined.shape}")

        # Tạo dataset từ df gốc và scaler đã fit
        full_dataset = GraphDatasetPyG(full_original_df_combined, NODE_FEATURES, TARGET_FEATURE, GRAPH_LOOKBACK, VOL_SPIKE_THRESHOLD, scaler)
        dataset_len = len(full_dataset) # Lấy độ dài sau khi tạo
        logging.info(f"PyG Dataset created with {dataset_len} samples.")
        if dataset_len < MIN_SAMPLES_FOR_TRAINING:
             exit(f"Final dataset size {dataset_len} is less than minimum required {MIN_SAMPLES_FOR_TRAINING}. Exiting.")
    except Exception as dataset_init_e:
         logging.error(f"Error creating GraphDatasetPyG: {dataset_init_e}", exc_info=True)
         exit("Exiting due to dataset creation error.")

    # 4. Time-based Split -> train_dataset, val_dataset
    logging.info("Performing time-based data split...")
    try:
        train_dataset, val_dataset = time_based_split(full_dataset, val_ratio=VALIDATION_SPLIT)
        logging.info(f"Time-based split completed: Train={len(train_dataset)}, Validation={len(val_dataset)}")
        # <<< KIỂM TRA SAU KHI SPLIT >>>
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             raise ValueError("Train or validation set is empty after split. Check dataset size and split ratio.")

    except ValueError as split_err:
        logging.error(f"Error splitting dataset: {split_err}", exc_info=True)
        exit("Exiting due to data split error.")
    except Exception as split_e_other: # Bắt các lỗi khác
        logging.error(f"Unexpected error during data split: {split_e_other}", exc_info=True)
        exit("Exiting due to unexpected data split error.")
    logging.info("Starting hyperparameter optimization with Optuna...")
    # <<< LƯU STUDY VÀO DATABASE >>>
    study = optuna.create_study(
        direction="maximize", # Tối đa hóa metric kết hợp
        study_name="gnn_anomaly_optimization_v2", # Tên study mới
        storage=f"sqlite:///{MODEL_SAVE_DIR / OPTUNA_DB_PATH}", # Lưu vào SQLite
        load_if_exists=True, # Tiếp tục nếu study đã tồn tại
        pruner=optuna.pruners.MedianPruner( # <<< CẤU HÌNH PRUNER >>>
            n_startup_trials=PRUNER_N_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_N_WARMUP_STEPS,
            interval_steps=1 # Kiểm tra pruning mỗi epoch
        )
    )
    early_stopping_callback = EarlyStoppingCallback(patience=STUDY_EARLY_STOPPING_PATIENCE, min_delta=0.001)

    try:
        # Truyền dataset/scaler vào objective
        objective_with_data = lambda trial: objective(trial, train_dataset, val_dataset, full_dataset)
        study.optimize(objective_with_data, n_trials=N_TRIALS,callbacks=[early_stopping_callback]) # Tăng timeout nếu cần
    # ... (Xử lý exception và hiển thị kết quả như trước) ...
    except KeyboardInterrupt: logging.warning("Optuna optimization interrupted.")
    except Exception as opt_e: logging.error(f"Optuna optimize error: {opt_e}", exc_info=True)

    if study.best_trial:
        # ... (Log best params/value như trước) ...
        logging.info("\n" + "="*40 + "\n Optuna Finished! \n" + "="*40)
        logging.info(f"Best Trial: {study.best_trial.number}")
        logging.info(f"Best Combined Metric: {study.best_value:.4f}")
        logging.info("Best Hyperparameters:")
        for key, value in study.best_params.items(): logging.info(f"  - {key:<15}: {value}")
        logging.info("="*40 + "\n")
        # Lưu best params
        try:
            with open(MODEL_SAVE_DIR / "best_gnn_hparams.json", 'w') as f: import json; json.dump(study.best_params, f, indent=4)
            logging.info(f"Best hyperparameters saved.")
        except Exception as save_e: logging.error(f"Could not save best hparams: {save_e}")
        # <<< THÊM VISUALIZATION >>>
        if optuna.visualization.is_available():
             try:
                 fig1 = optuna.visualization.plot_optimization_history(study)
                 fig1.write_image(MODEL_SAVE_DIR / "optuna_history.png")
                 fig2 = optuna.visualization.plot_param_importances(study)
                 fig2.write_image(MODEL_SAVE_DIR / "optuna_param_importances.png")
                 # Thêm các slice plot cho các tham số quan trọng
                 important_params = [p[0] for p in optuna.importance.get_param_importances(study).items() if p[1] > 0.05] # Lấy các param quan trọng
                 if important_params:
                      fig3 = optuna.visualization.plot_slice(study, params=important_params[:min(len(important_params), 5)]) # Vẽ tối đa 5 slice
                      fig3.write_image(MODEL_SAVE_DIR / "optuna_slice.png")
                 logging.info("Optuna visualization plots saved.")
             except Exception as plot_e:
                  logging.error(f"Failed to generate or save Optuna plots: {plot_e}")
        else: logging.warning("Plotly not installed. Skipping Optuna visualizations. (pip install plotly)")

    else: logging.warning("Optuna finished without finding a best trial.")
    logging.info("Hyperparameter optimization script finished.")
