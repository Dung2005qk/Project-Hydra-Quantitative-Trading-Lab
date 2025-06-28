import os
import sys
import logging
from pathlib import Path
import traceback
import json
import gc
import time
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

# <<< Logging Setup (Giữ nguyên) >>>
log_file_path = "train_final_gnn_model.log"
logging.basicConfig(
    level=logging.DEBUG, # Có thể đổi thành DEBUG nếu cần log chi tiết hơn
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'), # Thêm encoding
        logging.StreamHandler()
    ]
)
logging.info(f"Logging initialized for STANDALONE FINAL GNN training. Log file: {log_file_path}")

# <<< PyTorch Geometric Import và Fallback (Giữ nguyên) >>>
try:
    from torch_geometric.nn import GATConv as GraphConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool, global_max_pool
    _torch_geometric_available = True
    logging.info("Successfully imported torch_geometric.")
except ImportError:
     logging.error("torch_geometric required for final training. Please install it (pip install torch_geometric). Exiting.")
     exit(1) # Thoát với mã lỗi

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    try:
        torch.cuda.set_device(DEVICE)
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(DEVICE)}")
    except Exception as e:
        logging.error(f"Failed to set CUDA device {DEVICE}: {e}. Falling back to CPU.")
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")
logging.info(f"Using device: {DEVICE}")


# <<< Paths Setup (Giữ nguyên) >>>
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or 'KAGGLE_WORKING_DIR' in os.environ
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
if IS_KAGGLE and KAGGLE_INPUT_DIR.exists():
    # !!! QUAN TRỌNG: Đảm bảo tên dataset trên Kaggle khớp !!!
    DATASET_NAME = "databot" # <<< THAY TÊN DATASET DỮ LIỆU GỐC >>>
    OPTUNA_RESULTS_DATASET = "optunagnn" # <<< THAY TÊN DATASET CHỨA KẾT QUẢ OPTUNA >>>
    DATA_DIR = KAGGLE_INPUT_DIR / DATASET_NAME
    OPTUNA_DIR = KAGGLE_INPUT_DIR / OPTUNA_RESULTS_DATASET
    MODEL_SAVE_DIR = KAGGLE_OUTPUT_DIR / "final_trained_gnn"
    if not DATA_DIR.exists(): exit(f"Data Dataset '{DATA_DIR}' not found on Kaggle.")
    if not OPTUNA_DIR.exists(): exit(f"Optuna Results Dataset '{OPTUNA_DIR}' not found on Kaggle.")
else:
    # Cài đặt cho môi trường local
    try: SCRIPT_DIR = Path(__file__).resolve().parent # Dùng resolve() để đảm bảo đường dẫn tuyệt đối
    except NameError: SCRIPT_DIR = Path.cwd()
    DATA_DIR = SCRIPT_DIR # Dữ liệu cùng cấp với script
    OPTUNA_DIR = SCRIPT_DIR / "optuna_gnn_results" # Kết quả Optuna trong thư mục con
    MODEL_SAVE_DIR = SCRIPT_DIR / "final_trained_gnn" # Lưu model vào thư mục con

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Đảm bảo thư mục tồn tại
BEST_HPARAMS_PATH = OPTUNA_DIR / "best_gnn_hparams.json"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "final_iss_gnn_anomaly.pth"
SCALER_SAVE_PATH = MODEL_SAVE_DIR / "final_iss_gnn_scaler.pkl"
THRESHOLD_SAVE_PATH = MODEL_SAVE_DIR / "best_gnn_threshold.json"

# <<< Logging Paths (Giữ nguyên) >>>
logging.info(f"Using DATA_DIR: {DATA_DIR}")
logging.info(f"Loading best hyperparameters from: {BEST_HPARAMS_PATH}")
logging.info(f"Final model will be saved to: {MODEL_SAVE_PATH}")
logging.info(f"Scaler for final model will be saved to: {SCALER_SAVE_PATH}")
logging.info(f"Best threshold will be saved to: {THRESHOLD_SAVE_PATH}")

# <<< Load Best Hyperparameters (Giữ nguyên) >>>
if not BEST_HPARAMS_PATH.exists(): exit(f"Best hyperparameters file not found at {BEST_HPARAMS_PATH}. Please run Optuna first.")
try:
    with open(BEST_HPARAMS_PATH, 'r') as f: best_hparams = json.load(f)
    logging.info(f"Loaded best hyperparameters: {best_hparams}")
except Exception as e: exit(f"Error loading best hyperparameters from {BEST_HPARAMS_PATH}: {e}")

# --- Parameters ---

# <<< ĐỊNH NGHĨA TRỰC TIẾP THAY VÌ IMPORT TỪ api.py >>>
SYMBOLS_TO_TRAIN = ["BTC/USDT", "ETH/USDT"] # Hoặc đọc từ file config riêng, hoặc tham số dòng lệnh
NODE_FEATURES = ["close", "RSI", "ATR", "volatility", "volume"] # <<< DANH SÁCH FEATURE CỐ ĐỊNH CHO GNN NÀY >>>
NODE_DIM = len(NODE_FEATURES)
# <<< ============================================ >>>

# Các tham số khác của mô hình GNN và dữ liệu
TIMEFRANE = "15m"         # Timeframe sử dụng
GRAPH_LOOKBACK = 30        # Số nến lịch sử cho mỗi đồ thị
TARGET_FEATURE = "volatility" # Feature dùng để xác định target (ví dụ: biến động)
REQUIRED_COLS_LOAD = list(set(NODE_FEATURES + [TARGET_FEATURE])) # Các cột cần tải
VOL_SPIKE_THRESHOLD = 1.4  # Ngưỡng để xác định target (volatility spike)
MIN_SAMPLES_FOR_TRAINING = 1000 # Số mẫu tối thiểu để huấn luyện
VALIDATION_SPLIT = 0.15    # Tỷ lệ dữ liệu cho tập validation
EARLY_STOPPING_PATIENCE = 15 # Số epoch không cải thiện trước khi dừng sớm
EPOCHS = 150               # Số epoch huấn luyện tối đa

# Hyperparameters load từ file JSON (với giá trị mặc định an toàn)
EDGE_DIM = best_hparams.get('edge_dim', 2) # Mặc định là 2 nếu không có trong file
BATCH_SIZE = best_hparams.get('batch_size', 256)
LEARNING_RATE = best_hparams.get('lr', 0.0005)
WEIGHT_DECAY = best_hparams.get('weight_decay', 1e-5)
GRADIENT_CLIP_NORM = best_hparams.get('grad_clip', 1.0)
FOCAL_LOSS_ALPHA = best_hparams.get('focal_alpha', 0.25)
FOCAL_LOSS_GAMMA = best_hparams.get('focal_gamma', 2.0)
GAT_HEADS = best_hparams.get('heads', 4)
HIDDEN_DIM_GNN = best_hparams.get('hidden_dim', 64)
DROPOUT_GNN = best_hparams.get('dropout', 0.4)
GNN_LAYERS = best_hparams.get('gnn_layers', 3)

# <<< Logging Parameters (Cập nhật để rõ ràng hơn) >>>
logging.info("--- Final Training Parameters ---")
logging.info(f"Symbols: {SYMBOLS_TO_TRAIN}")
logging.info(f"Timeframe: {TIMEFRANE}")
logging.info(f"Node Features ({NODE_DIM}): {NODE_FEATURES}")
logging.info(f"Target Feature: {TARGET_FEATURE}")
logging.info(f"Graph Lookback: {GRAPH_LOOKBACK}")
logging.info(f"Volatility Spike Threshold: {VOL_SPIKE_THRESHOLD}")
logging.info(f"Epochs: {EPOCHS}")
logging.info(f"Batch Size: {BATCH_SIZE}")
logging.info(f"Learning Rate: {LEARNING_RATE}")
logging.info(f"Weight Decay: {WEIGHT_DECAY}")
logging.info(f"Gradient Clip Norm: {GRADIENT_CLIP_NORM}")
logging.info(f"Focal Loss Alpha: {FOCAL_LOSS_ALPHA}")
logging.info(f"Focal Loss Gamma: {FOCAL_LOSS_GAMMA}")
logging.info(f"GNN Layers: {GNN_LAYERS}")
logging.info(f"Hidden Dim GNN: {HIDDEN_DIM_GNN}")
logging.info(f"GAT Heads: {GAT_HEADS}")
logging.info(f"Dropout GNN: {DROPOUT_GNN}")
logging.info(f"Edge Dim: {EDGE_DIM}")
logging.info(f"Validation Split: {VALIDATION_SPLIT}")
logging.info(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
logging.info(f"Min Samples for Training: {MIN_SAMPLES_FOR_TRAINING}")
logging.info("-------------------------------")

# --- Helper Functions ---
def load_and_prepare_gnn_data(symbol: str, required_cols: List[str]) -> Optional[Tuple[str, pd.DataFrame]]:
    """Tải và chuẩn bị dữ liệu cho một symbol từ file pkl."""
    symbol_safe = symbol.replace('/', '_').replace(':', '')
    file_path = DATA_DIR / f"{symbol_safe}_data.pkl"
    if not file_path.exists():
        logging.warning(f"Data file not found: {file_path}")
        return None
    try:
        # Sử dụng joblib để tương thích với cách lưu phổ biến
        data_dict = joblib.load(file_path)
        df = data_dict.get(TIMEFRANE) # Lấy DataFrame theo timeframe
        if not isinstance(df, pd.DataFrame) or df.empty:
            logging.warning(f"No valid DataFrame found for timeframe '{TIMEFRANE}' in {file_path}")
            return None
        df = df.copy() # Tạo bản sao để tránh sửa đổi dữ liệu gốc
        # Kiểm tra các cột cần thiết
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns for {symbol}: {missing_cols}")
            return None
        # Chọn các cột cần thiết và xử lý NaN cơ bản
        df_processed = df[required_cols].ffill().bfill().fillna(0)
        if df_processed.isnull().values.any():
            logging.error(f"NaN values remain after processing for {symbol}")
            return None
        return symbol, df_processed
    except ImportError: # Xử lý nếu chưa cài joblib
        logging.error("joblib is required to load data. Please install it (`pip install joblib`).")
        return None
    except Exception as e:
        logging.error(f"Error loading/preparing data for {symbol}: {e}", exc_info=True)
        return None

def build_market_graph(df_slice_scaled: np.ndarray, lookback: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """Xây dựng đồ thị PyG từ slice dữ liệu đã scale."""
    global NODE_DIM, NODE_FEATURES # Sử dụng biến toàn cục đã định nghĩa
    current_node_dim = df_slice_scaled.shape[1]
    # Kiểm tra shape đầu vào
    if len(df_slice_scaled) != lookback or df_slice_scaled.ndim != 2 or current_node_dim != NODE_DIM:
        logging.error(f"build_market_graph: Invalid input shape {df_slice_scaled.shape}, expected ({lookback}, {NODE_DIM})")
        return None, None, None

    node_features = torch.tensor(df_slice_scaled, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long) # Khởi tạo rỗng
    edge_attrs_list = []
    full_edge_list = []
    has_indices = False
    close_idx, vol_idx = -1, -1

    # Tìm index của 'close' và 'volume' để tính edge attributes
    try:
        close_idx = NODE_FEATURES.index('close')
        vol_idx = NODE_FEATURES.index('volume')
        has_indices = True
    except ValueError:
        # Chỉ cảnh báo, edge_attr sẽ là None
        logging.debug("Cannot find 'close'/'volume' in NODE_FEATURES. Edge attributes will be None.")

    # Tạo các cạnh nối tiếp (temporal edges)
    if lookback >= 2:
        for i in range(lookback - 1):
            src, dst = i, i + 1
            full_edge_list.extend([(src, dst), (dst, src)]) # Cạnh hai chiều
            # Tính edge attributes nếu có index
            if has_indices:
                try:
                    # Tính thay đổi giá và volume (dùng abs để là giá trị dương)
                    price_change = abs(node_features[dst, close_idx].item() - node_features[src, close_idx].item())
                    vol_change = abs(node_features[dst, vol_idx].item() - node_features[src, vol_idx].item())
                    attr = [price_change, vol_change]
                    edge_attrs_list.extend([attr, attr]) # Thêm cho cả hai chiều
                except Exception as e:
                    # Nếu lỗi tính toán, tắt cờ và xóa list attrs
                    logging.error(f"Error calculating edge attributes at index {i}: {e}")
                    has_indices = False
                    edge_attrs_list = []

        # Tạo tensor edge_index
        if full_edge_list:
            edge_index = torch.tensor(full_edge_list, dtype=torch.long).t().contiguous()

    # Tạo tensor edge_attr
    edge_attr: Optional[torch.Tensor] = None
    # <<< Sử dụng EDGE_DIM đã load từ hparams >>>
    if has_indices and edge_attrs_list and edge_index.shape[1] > 0:
        edge_attr_tensor_temp = torch.tensor(edge_attrs_list, dtype=torch.float32)
        # <<< Kiểm tra dimension của edge_attr >>>
        if edge_attr_tensor_temp.shape[0] == edge_index.shape[1] and edge_attr_tensor_temp.shape[1] == EDGE_DIM:
            edge_attr = edge_attr_tensor_temp
        elif edge_attr_tensor_temp.shape[0] == edge_index.shape[1]:
            logging.warning(f"Edge attribute dimension mismatch ({edge_attr_tensor_temp.shape[1]} vs expected {EDGE_DIM}). Discarding edge attributes.")
        else:
            logging.error("Edge attribute count mismatch edge index count. Discarding edge attributes.")
    elif edge_index.shape[1] == 0:
        # Nếu không có cạnh, cũng không có thuộc tính cạnh
        edge_attr = None

    return node_features, edge_index, edge_attr

# <<< Model GNN Class (Giữ nguyên, đã sửa logic ở lần trước) >>>
class GraphNeuralNetPyG_Advanced(nn.Module):
    # --- Sử dụng các biến toàn cục hoặc biến từ hparams ---
    def __init__(self, node_dim=NODE_DIM, num_layers=GNN_LAYERS, edge_dim=EDGE_DIM,
                 hidden_dim=HIDDEN_DIM_GNN, out_dim=1, dropout=DROPOUT_GNN, heads=GAT_HEADS):
        super().__init__()
        if not _torch_geometric_available: raise ImportError("torch_geometric required")
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim if isinstance(edge_dim, int) and edge_dim > 0 else None
        self.num_layers = num_layers

        # --- Lớp chiếu đầu vào ---
        self.input_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

        # --- Khởi tạo các tầng GAT, BatchNorm và Skip Connections ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skips = nn.ModuleList()

        in_channels = hidden_dim # Input cho lớp GAT đầu tiên là hidden_dim sau projection
        for i in range(num_layers):
            out_channels_conv = hidden_dim # Hidden dim cho mỗi lớp GAT
            # Lớp cuối cùng dùng 1 head và không concat
            is_last_layer = (i == num_layers - 1)
            current_heads = 1 if is_last_layer else heads
            concat_final = False if is_last_layer else (heads > 1) # Concat nếu nhiều head và không phải lớp cuối

            # Kích thước output thực tế của lớp GAT
            out_channels_actual = out_channels_conv * current_heads if concat_final else out_channels_conv

            # --- Skip Connection ---
            # Kích thước input cho skip connection phải khớp với output của GAT
            if i == 0 : skip_in_dim = hidden_dim # Skip từ output của input_proj
            else: skip_in_dim = in_channels # Skip từ output lớp GAT trước
            # Tạo lớp Linear nếu kích thước không khớp, nếu khớp dùng Identity
            if skip_in_dim != out_channels_actual:
                self.skips.append(nn.Linear(skip_in_dim, out_channels_actual))
            else:
                self.skips.append(nn.Identity())

            # --- GAT Layer ---
            # output_dim là hidden_dim (không cần chia cho heads vì PyG tự xử lý)
            self.convs.append(GraphConv(
                in_channels=in_channels,
                out_channels=out_channels_conv,
                heads=current_heads,
                dropout=dropout,
                edge_dim=self.edge_dim,
                concat=concat_final
            ))

            # --- BatchNorm Layer ---
            self.bns.append(nn.BatchNorm1d(out_channels_actual))

            # Cập nhật input_dim cho lớp tiếp theo
            in_channels = out_channels_actual

        # --- Dropout và Predictor cuối cùng ---
        self.dropout_layer = nn.Dropout(p=dropout)
        # Input cho predictor là concat của mean và max pooling (nên gấp đôi dim của lớp GAT cuối)
        predictor_input_dim = in_channels * 2
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2), # Giảm dropout ở lớp cuối
            nn.Linear(hidden_dim, out_dim)
        )
        logging.info(f"Initialized FINAL GNN (GAT) Layers={num_layers}, Hidden={hidden_dim}, Heads={heads}, Dropout={dropout}, EdgeDim={self.edge_dim}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]=None, batch: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward pass qua mô hình GNN."""
        # Xác định device mục tiêu
        target_device = x.device if x is not None else next(self.parameters()).device
        # Xử lý input rỗng
        if x is None or edge_index is None or x.nelement() == 0:
            batch_size = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 1
            return torch.zeros((batch_size, 1), device=target_device)

        try:
            # Kiểm tra có sử dụng edge_attr không
            use_edge_attr = self.edge_dim is not None and edge_attr is not None
            if use_edge_attr and edge_attr.shape[1] != self.edge_dim:
                logging.warning(f"Forward pass: edge_attr shape mismatch ({edge_attr.shape[1]} vs {self.edge_dim}). Ignoring edge_attr.")
                use_edge_attr = False

            # Áp dụng input projection
            x = self.input_proj(x)
            identity = x # Lưu lại identity sau projection cho skip connection đầu tiên

            # Duyệt qua các tầng GAT
            for i in range(self.num_layers):
                # Chuẩn bị skip connection với dimension phù hợp
                layer_identity = self.skips[i](identity)
                x_in = x # Input cho lớp GAT

                # Áp dụng GATConv
                if use_edge_attr:
                    x = self.convs[i](x_in, edge_index, edge_attr=edge_attr)
                else:
                    x = self.convs[i](x_in, edge_index)

                # BatchNorm, Activation, Dropout
                x = self.bns[i](x)
                x = F.elu(x) # Sử dụng elu thay vì relu
                x = self.dropout_layer(x)

                # Áp dụng Skip Connection (chỉ cộng nếu shape khớp)
                if x.shape == layer_identity.shape:
                    x = x + layer_identity
                else:
                    # Lỗi này không nên xảy ra nếu logic skip connection đúng
                    logging.error(f"Skip connection shape mismatch layer {i+1}: GAT output {x.shape}, Skip input {layer_identity.shape}. Skipping add.")

                # Cập nhật identity cho lớp tiếp theo
                identity = x

            # --- Global Pooling ---
            if batch is None: # Xử lý trường hợp không có batch (chỉ 1 đồ thị)
                mean_pool = torch.mean(x, dim=0, keepdim=True)
                max_pool, _ = torch.max(x, dim=0, keepdim=True)
            else: # Sử dụng global pooling của PyG
                mean_pool = global_mean_pool(x, batch)
                max_pool = global_max_pool(x, batch)

            # Concatenate mean và max pooling
            graph_embedding = torch.cat([mean_pool, max_pool], dim=1)

            # --- Predictor ---
            out_logit = self.predictor(graph_embedding)
            return out_logit

        except Exception as e:
            logging.error(f"GNN forward pass error: {e}", exc_info=True);
            batch_size = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 1;
            return torch.zeros((batch_size, 1), device=target_device) # Trả về tensor 0 nếu lỗi

# <<< Focal Loss Class (Giữ nguyên) >>>
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = FOCAL_LOSS_ALPHA, gamma: float = FOCAL_LOSS_GAMMA, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Sử dụng BCEWithLogitsLoss với reduction='none' để tính toán từng phần tử
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits từ model (chưa qua sigmoid) shape [N, 1] hoặc [N]
            targets: Nhãn thực tế (0 hoặc 1) shape [N, 1] hoặc [N]
        Returns:
            Giá trị loss (scalar nếu reduction là 'mean'/'sum')
        """
        # Đảm bảo target cùng kiểu dữ liệu và device với input
        targets = targets.type_as(inputs)
        # Đảm bảo target có shape giống input (ví dụ: [N, 1] nếu input là [N, 1])
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)

        # Tính BCE loss thô
        BCE_loss = self.bce_with_logits(inputs, targets)
        # Tính pt (xác suất dự đoán đúng lớp)
        pt = torch.exp(-BCE_loss)
        # Tính alpha_t (trọng số cho lớp positive/negative)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Tính Focal loss
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        # Áp dụng reduction
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else: # 'none'
            return F_loss

# <<< GraphDatasetPyG Class (Giữ nguyên, đã sửa lỗi ở lần trước) >>>
class GraphDatasetPyG(Dataset):
    def __init__(self, df, node_features_list, target_col, graph_lookback, vol_spike_threshold, scaler):
        self.df = df
        self.node_features_list = node_features_list
        self.num_node_features = len(node_features_list)
        self.target_col = target_col
        self.graph_lookback = graph_lookback
        self.vol_spike_threshold = vol_spike_threshold
        self.scaler = scaler
        self.targets, self.valid_indices = self._calculate_targets_and_indices()
        self.pos_weight_value = self._calculate_pos_weight() # Dùng cho BCEWithLogitsLoss nếu cần, không dùng trực tiếp với FocalLoss

        if self.valid_indices.size == 0:
            logging.warning("GraphDatasetPyG: No valid indices found after target calculation.")

    def _calculate_targets_and_indices(self):
        targets = []
        valid_indices = []
        # Chuyển cột target sang numpy array để tăng tốc
        vol_values = self.df[self.target_col].values
        num_rows = len(self.df)

        # Lặp qua các vị trí có thể bắt đầu đồ thị
        for i in range(num_rows - self.graph_lookback):
            end_idx = i + self.graph_lookback # Index kết thúc của slice đồ thị
            target_idx = end_idx # Index của nến ngay sau đồ thị (để dự đoán)

            # Đảm bảo target_idx không vượt quá giới hạn DataFrame
            if target_idx >= num_rows:
                continue # Không thể lấy target cho slice cuối cùng

            # Lấy giá trị volatility hiện tại (cuối slice) và tiếp theo (target)
            current_vol = vol_values[end_idx - 1]
            next_vol = vol_values[target_idx]

            # Xác định target = 1 nếu có spike, ngược lại là 0
            # Thêm kiểm tra NaN và giá trị hợp lệ
            is_spike = (
                pd.notna(next_vol) and pd.notna(current_vol) and
                current_vol > 1e-9 and # Tránh chia cho 0 hoặc số rất nhỏ
                next_vol > current_vol * self.vol_spike_threshold
            )
            target = 1.0 if is_spike else 0.0

            targets.append(target)
            valid_indices.append(i) # Lưu index bắt đầu của slice hợp lệ

        return np.array(targets, dtype=np.float32), np.array(valid_indices, dtype=int)

    def _calculate_pos_weight(self) -> Optional[float]:
        """Tính trọng số cho lớp positive (ít dùng với FocalLoss)."""
        if len(self.targets) == 0: return None
        num_pos = np.sum(self.targets == 1)
        num_neg = len(self.targets) - num_pos
        if num_pos == 0 or num_neg == 0:
            logging.warning(f"Only one class found ({num_pos} pos / {num_neg} neg). Cannot calculate meaningful pos_weight.")
            return None # Trả về None nếu chỉ có 1 lớp
        pos_w_value = num_neg / num_pos
        logging.info(f"Calculated pos_weight (for potential BCE): {pos_w_value:.2f} (Positives: {num_pos}/{len(self.targets)})")
        return float(pos_w_value) # Đảm bảo trả về float

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx) -> Optional[Data]:
        if idx >= len(self.valid_indices):
            # Nên raise IndexError hoặc trả về None tùy theo cách DataLoader xử lý
            # logging.debug(f"Index {idx} out of bounds for valid_indices.")
            return None

        actual_start_idx = self.valid_indices[idx]
        end_idx = actual_start_idx + self.graph_lookback

        # Kiểm tra lại phạm vi (an toàn hơn)
        if end_idx > len(self.df):
            logging.error(f"Slice end index {end_idx} exceeds DataFrame length {len(self.df)} for idx {idx}.")
            return None

        try:
            # Lấy slice dữ liệu chưa scale
            df_slice_unscaled = self.df.iloc[actual_start_idx:end_idx][self.node_features_list]

            # Kiểm tra lại kích thước slice
            if len(df_slice_unscaled) != self.graph_lookback:
                logging.error(f"Slice length mismatch for idx {idx}. Expected {self.graph_lookback}, got {len(df_slice_unscaled)}.")
                return None

            # Scale dữ liệu
            # Kiểm tra NaN trước khi scale
            if df_slice_unscaled.isnull().values.any():
                 logging.warning(f"NaN found in unscaled slice for idx {idx}. Trying to fill.")
                 df_slice_unscaled = df_slice_unscaled.ffill().bfill().fillna(0)
                 if df_slice_unscaled.isnull().values.any():
                      logging.error(f"Could not fill NaNs in slice for idx {idx}.")
                      return None

            scaled_slice_values = self.scaler.transform(df_slice_unscaled.values)

            # Kiểm tra NaN/Inf sau khi scale
            if not np.all(np.isfinite(scaled_slice_values)):
                logging.error(f"NaN or Inf found in scaled feature values for idx {idx}. Scaled shape: {scaled_slice_values.shape}")
                # Ghi thêm thông tin unscaled để debug
                # logging.debug(f"Unscaled slice head:\n{df_slice_unscaled.head()}")
                return None

            # Xây dựng đồ thị PyG
            node_features, edge_index, edge_attr = build_market_graph(scaled_slice_values, self.graph_lookback)

            # Kiểm tra kết quả từ build_market_graph
            if node_features is None or edge_index is None:
                logging.error(f"build_market_graph returned None for idx {idx}.")
                return None

            # Lấy target
            target_value = self.targets[idx]
            target = torch.tensor([target_value], dtype=torch.float32) # Target là float cho BCE/FocalLoss

            # Tạo đối tượng Data
            return Data(x=node_features, edge_index=edge_index, y=target, edge_attr=edge_attr)

        except Exception as e:
            logging.error(f"Error in __getitem__ for index {idx} (start_idx={actual_start_idx}): {e}", exc_info=True)
            return None # Trả về None nếu có lỗi

# <<< time_based_split Function (Giữ nguyên) >>>
def time_based_split(dataset: Dataset, val_ratio: float = VALIDATION_SPLIT) -> Tuple[Subset, Subset]:
    """Chia dataset thành train/validation dựa trên thời gian (chỉ số)."""
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot split an empty dataset.")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1.")

    val_size = int(n * val_ratio)
    train_size = n - val_size

    # Đảm bảo cả hai tập đều có dữ liệu
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Cannot split dataset of size {n} with val_ratio {val_ratio}. Train size: {train_size}, Val size: {val_size}")

    # Chỉ số được sắp xếp theo thời gian do cách tạo dataset
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, n))

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# <<< collate_fn Function (Giữ nguyên) >>>
def collate_fn(batch: List[Optional[Data]]) -> Optional[Batch]:
    """Loại bỏ các giá trị None và tạo batch PyG."""
    # Lọc bỏ các giá trị None (do lỗi trong __getitem__)
    batch = [data for data in batch if data is not None]
    if not batch:
        # Trả về None nếu batch rỗng sau khi lọc
        # logging.debug("Collate function received an empty or all-None batch.")
        return None
    try:
        # Sử dụng Batch.from_data_list để tạo batch PyG
        return Batch.from_data_list(batch)
    except Exception as e:
        # Log lỗi nếu không tạo được batch
        logging.error(f"Error in collate_fn during Batch.from_data_list: {e}", exc_info=True)
        return None


# --- Main Script Execution ---
if __name__ == "__main__":
    # 1. Load Data
    loaded_data_list: List[Tuple[str, pd.DataFrame]] = []
    logging.info(f"Loading data for symbols: {SYMBOLS_TO_TRAIN}")
    for symbol in SYMBOLS_TO_TRAIN:
        # <<< Sử dụng REQUIRED_COLS_LOAD đã định nghĩa >>>
        load_result = load_and_prepare_gnn_data(symbol, REQUIRED_COLS_LOAD)
        if load_result:
            loaded_data_list.append(load_result)
            logging.info(f"Successfully loaded and prepared data for {symbol}")
        else:
            logging.warning(f"Failed to load data for {symbol}. It will be excluded.")

    if not loaded_data_list:
        logging.critical("No data could be loaded for any symbol. Exiting.")
        exit(1)

    # 2. Fit Scaler
    scaler = None
    logging.info("Fitting the final scaler on all available training data...")
    # <<< Sử dụng NODE_FEATURES đã định nghĩa >>>
    cols_to_scale = NODE_FEATURES
    # Chỉ lấy các feature cần scale từ các DataFrame đã load thành công
    feature_dfs_to_concat = []
    for _, df in loaded_data_list:
        if all(c in df.columns for c in cols_to_scale):
            feature_dfs_to_concat.append(df[cols_to_scale])
        else:
            missing_in_df = [c for c in cols_to_scale if c not in df.columns]
            logging.warning(f"DataFrame for symbol {_} missing scaler columns: {missing_in_df}. Excluding from scaler fitting.")

    if not feature_dfs_to_concat:
        logging.critical("No valid data available for scaler fitting. Exiting.")
        exit(1)

    # Ghép nối các DataFrame features
    try:
        full_df_combined_features = pd.concat(feature_dfs_to_concat, ignore_index=True)
        logging.info(f"Combined data for scaling shape: {full_df_combined_features.shape}")

        # Kiểm tra NaN/Inf trước khi fit scaler
        if np.any(np.isnan(full_df_combined_features.values)) or np.any(np.isinf(full_df_combined_features.values)):
            nan_inf_rows = np.where(np.isnan(full_df_combined_features.values) | np.isinf(full_df_combined_features.values))[0]
            logging.error(f"NaN/Inf detected in {len(np.unique(nan_inf_rows))} rows before scaling. Cannot fit scaler. Example rows: {np.unique(nan_inf_rows)[:10]}")
            exit(1)

        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(full_df_combined_features.values)
        # Lưu scaler
        joblib.dump(scaler, SCALER_SAVE_PATH)
        logging.info(f"Final scaler fitted and saved to {SCALER_SAVE_PATH}")
    except ValueError as ve:
        logging.error(f"ValueError during scaler fitting (check data consistency): {ve}", exc_info=True)
        exit(1)
    except Exception as scaler_e:
        logging.error(f"Error fitting or saving final scaler: {scaler_e}", exc_info=True)
        exit(1)

    # 3. Create Dataset
    logging.info("Creating PyG Dataset...")
    full_dataset = None
    try:
        # Ghép nối lại các DataFrame gốc (bao gồm cả target)
        original_dfs_to_concat = []
        for _, df in loaded_data_list:
            if all(c in df.columns for c in REQUIRED_COLS_LOAD):
                original_dfs_to_concat.append(df[REQUIRED_COLS_LOAD])
            # else: # Đã log ở bước trước
            #     pass

        if not original_dfs_to_concat:
            logging.critical("No valid dataframes for dataset creation after loading.")
            exit(1)

        full_original_df_combined = pd.concat(original_dfs_to_concat, ignore_index=True)
        logging.info(f"Combined original data for dataset shape: {full_original_df_combined.shape}")

        # Tạo đối tượng Dataset PyG
        # <<< Sử dụng NODE_FEATURES, TARGET_FEATURE, GRAPH_LOOKBACK, VOL_SPIKE_THRESHOLD đã định nghĩa >>>
        full_dataset = GraphDatasetPyG(
            full_original_df_combined,
            NODE_FEATURES,
            TARGET_FEATURE,
            GRAPH_LOOKBACK,
            VOL_SPIKE_THRESHOLD,
            scaler # Truyền scaler đã fit
        )
        dataset_len = len(full_dataset)
        logging.info(f"PyG Dataset created with {dataset_len} samples.")
        # <<< Sử dụng MIN_SAMPLES_FOR_TRAINING đã định nghĩa >>>
        if dataset_len < MIN_SAMPLES_FOR_TRAINING:
            logging.critical(f"Dataset size ({dataset_len}) is less than minimum required ({MIN_SAMPLES_FOR_TRAINING}). Exiting.")
            exit(1)
    except Exception as dataset_init_e:
        logging.critical(f"Error creating GraphDatasetPyG: {dataset_init_e}", exc_info=True)
        exit(1)

    # 4. Time-based Split
    logging.info("Performing time-based data split...")
    try:
        # <<< Sử dụng VALIDATION_SPLIT đã định nghĩa >>>
        train_dataset, val_dataset = time_based_split(full_dataset, val_ratio=VALIDATION_SPLIT)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(f"Empty train ({len(train_dataset)}) or validation ({len(val_dataset)}) set after split.")
        logging.info(f"Time-based split: Train={len(train_dataset)}, Val={len(val_dataset)}")
    except ValueError as split_err:
        logging.critical(f"Error splitting dataset: {split_err}")
        exit(1)
    except Exception as split_e_other:
        logging.critical(f"Unexpected error during data split: {split_e_other}", exc_info=True)
        exit(1)

    # 5. Weighted Sampler
    logging.info("Calculating weights for WeightedRandomSampler...")
    train_sampler = None
    try:
        train_indices = train_dataset.indices # Lấy chỉ số của tập train từ Subset
        # Lấy target của tập train từ dataset gốc sử dụng indices
        train_targets = full_dataset.targets[train_indices]

        if len(train_targets) > 0:
             # Đếm số lượng mỗi lớp (0 và 1)
             class_counts = np.bincount(train_targets.astype(int))
             # Kiểm tra xem có đủ 2 lớp và lớp 1 có mẫu không
             if len(class_counts) >= 2 and class_counts[1] > 0:
                  # Tính trọng số ngược (lớp hiếm có trọng số cao hơn)
                  class_weights = 1. / class_counts
                  # Gán trọng số cho từng mẫu trong tập train
                  sample_weights = np.array([class_weights[int(t)] for t in train_targets])
                  # Tạo Sampler
                  train_sampler = WeightedRandomSampler(
                      weights=torch.from_numpy(sample_weights), # Trọng số cho từng mẫu
                      num_samples=len(sample_weights),         # Số mẫu cần lấy (bằng kích thước tập train)
                      replacement=True                         # Lấy có thay thế
                  )
                  num_pos = class_counts[1]; num_neg = class_counts[0]
                  logging.info(f"WeightedRandomSampler created for training set (Pos: {num_pos}, Neg: {num_neg}).")
             else:
                  pos_count = class_counts[1] if len(class_counts) > 1 else 0
                  logging.warning(f"Cannot create WeightedRandomSampler: Not enough classes or positive samples (Positives: {pos_count}). Training will proceed without sampler.")
        else:
            logging.warning("Training dataset is empty after split. Cannot create sampler.")
    except IndexError as sampler_idx_e:
         logging.error(f"IndexError during sampler creation (check target values/indices): {sampler_idx_e}", exc_info=True)
    except Exception as sampler_e:
        logging.error(f"Error creating WeightedRandomSampler: {sampler_e}", exc_info=True)


    # 6. Create DataLoaders
    logging.info("Creating DataLoaders...")
    try:
        # <<< Sử dụng BATCH_SIZE đã load từ hparams >>>
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler, # sampler=None nếu không tạo được
            num_workers=0, # Đặt là 0 để dễ debug, có thể tăng nếu cần hiệu năng
            collate_fn=collate_fn,
            shuffle=(train_sampler is None), # Chỉ shuffle nếu không dùng sampler
            pin_memory=torch.cuda.is_available(), # Tăng tốc chuyển data sang GPU
            persistent_workers=False # Không cần thiết nếu num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE * 2, # Có thể dùng batch size lớn hơn cho validation
            shuffle=False, # Không cần shuffle validation
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        logging.info(f"DataLoaders created. Train sampler: {'Enabled' if train_sampler else 'Disabled'}")
    except Exception as loader_e:
        logging.critical(f"DataLoader creation error: {loader_e}", exc_info=True)
        exit(1)


    # --- 7. Initialize Model, Optimizer, Loss, Scheduler ---
    logging.info("Initializing model, optimizer, loss, and scheduler with best hyperparameters...")
    try:
        # Khởi tạo model GNN với các tham số đã load/định nghĩa
        # <<< NODE_DIM, GNN_LAYERS, EDGE_DIM, HIDDEN_DIM_GNN, DROPOUT_GNN, GAT_HEADS >>>
        model = GraphNeuralNetPyG_Advanced(
            node_dim=NODE_DIM, num_layers=GNN_LAYERS, edge_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM_GNN, dropout=DROPOUT_GNN, heads=GAT_HEADS
        ).to(DEVICE)

        # Khởi tạo Optimizer
        # <<< LEARNING_RATE, WEIGHT_DECAY >>>
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Khởi tạo Scheduler (giảm LR khi metric ngừng cải thiện)
        # Theo dõi F1-score trên tập validation (mode='max')
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7, verbose=True)

        # Khởi tạo hàm Loss
        # <<< FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA >>>
        # Không cần pos_weight cho FocalLoss
        criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
        logging.info(f"Using FocalLoss with alpha={FOCAL_LOSS_ALPHA}, gamma={FOCAL_LOSS_GAMMA}.")

    except Exception as init_e:
        logging.critical(f"CRITICAL ERROR during model/optimizer/loss initialization: {init_e}", exc_info=True)
        exit(1)

    # --- 8. Training Loop ---
    # <<< Sử dụng EPOCHS, EARLY_STOPPING_PATIENCE đã định nghĩa >>>
    logging.info(f"--- Starting FINAL GNN training for {EPOCHS} epochs ---")
    best_val_f1 = -1.0
    best_model_state = None
    no_improve_epochs = 0
    all_epoch_metrics = [] # Lưu metrics mỗi epoch

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        logging.info(f"--- Epoch {epoch+1}/{EPOCHS} ---")

        # --- Training Phase ---
        model.train() # Đặt model ở chế độ train
        train_loss_sum = 0.0
        processed_train_batches = 0
        all_train_targets_epoch = []
        all_train_preds_probs_epoch = []

        for i, batch in enumerate(train_loader):
            if batch is None: # Bỏ qua batch lỗi từ collate_fn
                logging.warning(f"Skipping None batch at train step {i}")
                continue
            try:
                 # Chuyển batch sang device
                 batch = batch.to(DEVICE)
                 # Kiểm tra dữ liệu batch cơ bản
                 if batch.x is None or batch.edge_index is None or batch.y is None or batch.batch is None or batch.x.nelement() == 0:
                      logging.warning(f"Skipping empty/invalid batch at train step {i}. Batch keys: {batch.keys}")
                      continue

                 optimizer.zero_grad() # Xóa gradient cũ
                 # Lấy edge_attr nếu có
                 edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
                 # Forward pass
                 logits = model(batch.x, batch.edge_index, edge_attr=edge_attr, batch=batch.batch)
                 # Chuẩn bị target (đúng shape và kiểu)
                 target = batch.y.view_as(logits).float()
                 # Tính loss
                 loss = criterion(logits, target)

                 # Kiểm tra loss hợp lệ
                 if torch.isnan(loss) or torch.isinf(loss):
                      logging.warning(f"NaN/Inf detected in training loss at batch {i}. Skipping batch backward.")
                      continue # Không backward loss lỗi

                 # Backward và Optimize
                 loss.backward()
                 # <<< Sử dụng GRADIENT_CLIP_NORM đã load từ hparams >>>
                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                 optimizer.step()

                 # Lưu loss và kết quả dự đoán để tính metrics cuối epoch
                 train_loss_sum += loss.item()
                 processed_train_batches += 1
                 # Lưu target và xác suất sigmoid(logits)
                 all_train_targets_epoch.extend(batch.y.squeeze().cpu().tolist())
                 all_train_preds_probs_epoch.extend(torch.sigmoid(logits).squeeze().detach().cpu().tolist())

            except Exception as e:
                 # Log lỗi chi tiết của batch nhưng không dừng hẳn training
                 logging.error(f"Error processing Training Batch {i} (Epoch {epoch+1}): {e}", exc_info=True)
                 # Optional: Thêm cơ chế dừng nếu quá nhiều lỗi batch liên tiếp
                 continue

        # Tính loss trung bình epoch train
        train_loss_avg = train_loss_sum / processed_train_batches if processed_train_batches > 0 else float('inf')

        # Tính Train Metrics (Precision, Recall, F1, AUC)
        train_precision, train_recall, train_f1, train_auc = 0.0, 0.0, 0.0, 0.0
        if all_train_targets_epoch:
            try:
                # Chuyển xác suất thành nhãn dự đoán (0/1) với ngưỡng 0.5
                train_preds_epoch = (np.array(all_train_preds_probs_epoch) > 0.5).astype(int)
                train_precision = precision_score(all_train_targets_epoch, train_preds_epoch, zero_division=0)
                train_recall = recall_score(all_train_targets_epoch, train_preds_epoch, zero_division=0)
                train_f1 = f1_score(all_train_targets_epoch, train_preds_epoch, zero_division=0)
                # Tính AUC nếu có cả 2 lớp trong target
                if len(np.unique(all_train_targets_epoch)) > 1:
                    train_auc = roc_auc_score(all_train_targets_epoch, all_train_preds_probs_epoch)
                else: train_auc = 0.5 # AUC = 0.5 nếu chỉ có 1 lớp
            except Exception as e:
                logging.error(f"Epoch {epoch+1}: Error calculating Train Metrics: {e}")

        # --- Validation Phase ---
        model.eval() # Đặt model ở chế độ eval
        val_loss_sum = 0.0
        processed_val_batches = 0
        all_val_targets_epoch = []
        all_val_preds_probs_epoch = []

        with torch.no_grad(): # Không cần tính gradient khi validation
             for i, batch in enumerate(val_loader):
                  if batch is None:
                      logging.warning(f"Skipping None batch at validation step {i}")
                      continue
                  try:
                       batch = batch.to(DEVICE)
                       if batch.x is None or batch.edge_index is None or batch.y is None or batch.batch is None or batch.x.nelement() == 0:
                            logging.warning(f"Skipping empty/invalid batch at validation step {i}. Batch keys: {batch.keys}")
                            continue

                       edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
                       logits_val = model(batch.x, batch.edge_index, edge_attr=edge_attr, batch=batch.batch)
                       target_val = batch.y.view_as(logits_val).float()
                       loss_val = criterion(logits_val, target_val)

                       # Chỉ tính loss nếu hợp lệ
                       if not torch.isnan(loss_val) and not torch.isinf(loss_val):
                           val_loss_sum += loss_val.item()
                           processed_val_batches += 1
                           # Lưu target và xác suất sigmoid(logits)
                           all_val_targets_epoch.extend(batch.y.squeeze().cpu().tolist())
                           all_val_preds_probs_epoch.extend(torch.sigmoid(logits_val).squeeze().cpu().tolist())
                       # else: logging.warning(f"NaN/Inf detected in validation loss at batch {i}.") # Ít log hơn cho val

                  except Exception as e:
                       logging.error(f"Error processing Validation Batch {i} (Epoch {epoch+1}): {e}", exc_info=True)
                       continue

        # Tính loss trung bình epoch validation
        val_loss_avg = val_loss_sum / processed_val_batches if processed_val_batches > 0 else float('inf')

        # Tính Val Metrics (Precision, Recall, F1, AUC)
        val_precision, val_recall, val_f1, val_auc = 0.0, 0.0, 0.0, 0.0
        if all_val_targets_epoch:
             try:
                 # Chuyển xác suất thành nhãn dự đoán (0/1) với ngưỡng 0.5
                 val_preds_epoch = (np.array(all_val_preds_probs_epoch) > 0.5).astype(int)
                 val_f1 = f1_score(all_val_targets_epoch, val_preds_epoch, zero_division=0) # Metric chính
                 val_precision = precision_score(all_val_targets_epoch, val_preds_epoch, zero_division=0)
                 val_recall = recall_score(all_val_targets_epoch, val_preds_epoch, zero_division=0)
                 if len(np.unique(all_val_targets_epoch)) > 1:
                     val_auc = roc_auc_score(all_val_targets_epoch, all_val_preds_probs_epoch)
                 else: val_auc = 0.5
             except Exception as e:
                 logging.error(f"Epoch {epoch+1}: Error calculating Validation Metrics: {e}")

        # --- Logging Epoch Results ---
        epoch_duration = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}/{EPOCHS} [{epoch_duration:.2f}s] - Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")
        logging.info(f"  Train P/R/F1/AUC: {train_precision:.4f}/{train_recall:.4f}/{train_f1:.4f}/{train_auc:.4f}")
        logging.info(f"  Val   P/R/F1/AUC: {val_precision:.4f}/{val_recall:.4f}/{val_f1:.4f}/{val_auc:.4f}")

        # Lưu metrics vào list
        all_epoch_metrics.append({
             'epoch': epoch + 1, 'train_loss': train_loss_avg, 'val_loss': val_loss_avg,
             'train_f1': train_f1, 'val_f1': val_f1, 'val_auc': val_auc,
             'train_precision': train_precision, 'train_recall': train_recall,
             'val_precision': val_precision, 'val_recall': val_recall
        })

        # --- Early Stopping & Save Best Model (Based on Val F1) ---
        current_metric_to_save = val_f1 if not np.isnan(val_f1) else -1.0 # Dùng val_f1 để lưu model
        if epoch == 0 or current_metric_to_save > best_val_f1:
            # Nếu là epoch đầu tiên hoặc F1 cải thiện
            if current_metric_to_save > best_val_f1: # Log chỉ khi cải thiện thực sự
                 logging.info(f"*** Validation F1 improved from {best_val_f1:.4f} to {current_metric_to_save:.4f} ***")
            best_val_f1 = current_metric_to_save
            # <<< Quan trọng: Chỉ lưu state_dict vào bộ nhớ, chưa lưu file >>>
            best_model_state = model.state_dict()
            no_improve_epochs = 0 # Reset bộ đếm early stopping
        else:
            no_improve_epochs += 1
            logging.info(f"Validation F1 did not improve for {no_improve_epochs} epoch(s). Best F1: {best_val_f1:.4f}")
            # <<< Sử dụng EARLY_STOPPING_PATIENCE đã định nghĩa >>>
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered at epoch {epoch+1}.")
                break # Dừng vòng lặp training

        # --- Step Scheduler ---
        # Dùng val_f1 cho scheduler (nếu hợp lệ), nếu không dùng (-val_loss)
        metric_for_scheduler = current_metric_to_save if current_metric_to_save >= 0 else (-val_loss_avg if val_loss_avg != float('inf') else 0)
        scheduler.step(metric_for_scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        # logging.debug(f"Epoch {epoch+1} - LR: {current_lr:.7f} - Scheduler metric: {metric_for_scheduler:.4f}") # Có thể bỏ log này nếu verbose=True

        # --- Memory Cleanup (Optional but recommended) ---
        # Giải phóng bộ nhớ GPU không cần thiết
        if DEVICE.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()

    # --- END OF TRAINING LOOP ---
    logging.info("--- FINAL GNN Training Finished ---")

    # --- LOAD BEST MODEL AND FIND BEST THRESHOLD ---
    if best_model_state is not None:
        logging.info(f"Loading best model state (achieved Val F1: {best_val_f1:.4f}) for final evaluation and saving...")
        try:
            # Khởi tạo lại model với cấu trúc giống hệt lúc train
            # Điều này đảm bảo cấu trúc khớp với state_dict
            model = GraphNeuralNetPyG_Advanced(
                node_dim=NODE_DIM, num_layers=GNN_LAYERS, edge_dim=EDGE_DIM,
                hidden_dim=HIDDEN_DIM_GNN, dropout=DROPOUT_GNN, heads=GAT_HEADS
            ).to(DEVICE)
            # Load state_dict tốt nhất vào model
            model.load_state_dict(best_model_state)
            # <<< LƯU MODEL TỐT NHẤT RA FILE >>>
            torch.save(best_model_state, MODEL_SAVE_PATH)
            logging.info(f"Final best model saved successfully to: {MODEL_SAVE_PATH}")
        except Exception as load_save_e:
             logging.error(f"Error reloading/saving best model: {load_save_e}", exc_info=True)
             # Cân nhắc: Dùng model cuối cùng nếu load lỗi?
             logging.warning("Proceeding with the model state from the LAST epoch for evaluation due to load/save error.")
             # model.load_state_dict(model.state_dict()) # Giữ state cuối cùng (đã có sẵn)

        logging.info("Running final validation pass with the best model to collect predictions for threshold finding...")
        model.eval() # Đặt model ở chế độ eval
        all_final_val_targets = []
        all_final_val_preds_probs = []
        with torch.no_grad():
             # Chạy lại trên val_loader
             for i, batch in enumerate(val_loader):
                  if batch is None: continue
                  try:
                       batch = batch.to(DEVICE)
                       if batch.x is None or batch.edge_index is None or batch.y is None or batch.batch is None or batch.x.nelement() == 0: continue
                       edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
                       # Dự đoán bằng model tốt nhất
                       logits_final = model(batch.x, batch.edge_index, edge_attr=edge_attr, batch=batch.batch)
                       # Lưu target và xác suất
                       if batch.y is not None and logits_final is not None:
                           all_final_val_targets.extend(batch.y.squeeze().cpu().tolist())
                           all_final_val_preds_probs.extend(torch.sigmoid(logits_final).squeeze().cpu().tolist())
                  except Exception as e:
                       logging.error(f"Error during final validation pass Batch {i}: {e}", exc_info=True); continue

        # --- FIND BEST THRESHOLD ---
        best_threshold = 0.5 # Mặc định
        best_f1_thresh = -1.0
        if all_final_val_targets and all_final_val_preds_probs:
            logging.info("Finding best decision threshold on validation set based on F1-score...")
            # Tạo các ngưỡng để thử
            thresholds = np.arange(0.05, 1.0, 0.01)
            f1_scores_at_thresholds = []
            for threshold in thresholds:
                # Chuyển xác suất thành nhãn nhị phân theo ngưỡng
                y_pred_binary = (np.array(all_final_val_preds_probs) >= threshold).astype(int)
                # Tính F1 score
                current_f1 = f1_score(all_final_val_targets, y_pred_binary, zero_division=0)
                f1_scores_at_thresholds.append(current_f1)
                # Cập nhật ngưỡng tốt nhất
                if current_f1 > best_f1_thresh:
                    best_f1_thresh = current_f1
                    best_threshold = threshold

            logging.info(f"==> Best Threshold found: {best_threshold:.2f} (yielding Val F1: {best_f1_thresh:.4f})")

            # --- Calculate final metrics AT BEST THRESHOLD ---
            final_preds_binary = (np.array(all_final_val_preds_probs) >= best_threshold).astype(int)
            final_precision = precision_score(all_final_val_targets, final_preds_binary, zero_division=0)
            final_recall = recall_score(all_final_val_targets, final_preds_binary, zero_division=0)
            final_auc = 0.0
            # Tính AUC cuối cùng (nếu có đủ 2 lớp)
            if len(np.unique(all_final_val_targets)) > 1:
                 try:
                      final_auc = roc_auc_score(all_final_val_targets, all_final_val_preds_probs)
                 except ValueError as auc_e:
                      logging.warning(f"Could not calculate final ROC AUC: {auc_e}")
            else: final_auc = 0.5

            logging.info("--- Final Validation Metrics (at Best Threshold) ---")
            logging.info(f"Threshold: {best_threshold:.2f}")
            logging.info(f"Precision: {final_precision:.4f}")
            logging.info(f"Recall:    {final_recall:.4f}")
            logging.info(f"F1-Score:  {best_f1_thresh:.4f}") # F1 tại ngưỡng tốt nhất
            logging.info(f"ROC AUC:   {final_auc:.4f}")
            try:
                logging.info(f"Confusion Matrix:\n{confusion_matrix(all_final_val_targets, final_preds_binary)}")
            except Exception as cm_e: logging.warning(f"Could not display confusion matrix: {cm_e}")
            logging.info("----------------------------------------------------")

            # <<< LƯU NGƯỠNG TỐT NHẤT RA FILE >>>
            try:
                with open(THRESHOLD_SAVE_PATH, 'w') as f: json.dump({"best_threshold": best_threshold}, f, indent=4)
                logging.info(f"Best threshold saved to {THRESHOLD_SAVE_PATH}")
            except Exception as save_thr_e:
                logging.error(f"Could not save best threshold: {save_thr_e}")
        else:
            logging.error("Could not collect final validation predictions. Cannot optimize or save threshold.")
            logging.info("Using default threshold 0.5 if needed elsewhere.")

    else: # Trường hợp không có best_model_state nào được lưu (có thể do training lỗi ngay epoch đầu)
        logging.error("No best model state found (training might have failed early). Cannot evaluate or find threshold.")

    # --- Final Logging ---
    logging.info(f"Final GNN training script finished.")
    logging.info(f"Model saved to: {MODEL_SAVE_PATH}")
    logging.info(f"Scaler saved to: {SCALER_SAVE_PATH}")
    if THRESHOLD_SAVE_PATH.exists(): logging.info(f"Threshold saved to: {THRESHOLD_SAVE_PATH}")
    logging.info(f"Log file saved to: {log_file_path}")

    # Optional: Plot training history
    try:
        import matplotlib.pyplot as plt
        metrics_df = pd.DataFrame(all_epoch_metrics)
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(MODEL_SAVE_DIR / "loss_curve.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_f1'], label='Train F1')
        plt.plot(metrics_df['epoch'], metrics_df['val_f1'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(MODEL_SAVE_DIR / "f1_curve.png")
        plt.close()
        logging.info("Saved loss and F1 curves to PNG files.")
    except ImportError:
        logging.warning("matplotlib not installed. Skipping curve plotting.")
    except Exception as plot_e:
        logging.error(f"Error plotting metrics: {plot_e}")