import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional, List, Any, Callable

# Optional: TA-Lib for advanced indicators
try:
    import talib
except ImportError:
    warnings.warn(
        "TA-Lib library not found. Advanced indicators will not be calculated. "
        "Install using official instructions or 'pip install TA-Lib'. "
        "Feature set will be limited."
    )
    talib = None

print(f"Using PyTorch version: {torch.__version__}")
if talib:
    print(f"TA-Lib version: {talib.__version__}")
else:
    print("TA-Lib not found.")

# --- Định nghĩa số lượng và tên các chế độ thị trường ---
DEFAULT_NUM_REGIMES = 3 # Ví dụ: 0: TREND_UP, 1: SIDEWAYS, 2: TREND_DOWN
REGIME_MAP = {0: "TREND_UP", 1: "SIDEWAYS", 2: "TREND_DOWN", -1: "UNKNOWN"} # Map index sang tên

# ==============================================================================
# 1. FEATURE ENGINEERING FUNCTIONS 
# ==============================================================================
def apply_basic_feature_engineering(x, input_dim=5):
    """Applies basic features: Log Returns, Volume Z-Score."""
    if x.shape[-1] != input_dim:
        warnings.warn(f"Basic FE: Expected input_dim {input_dim} but got {x.shape[-1]}. Skipping.")
        return x, x.shape[-1]
    if x.shape[1] < 2:
        warnings.warn("Basic FE: Sequence length < 2. Skipping.")
        return x, x.shape[-1]

    engineered_features = [x.clone()]
    close_prices = x[..., 3].clone()
    volume = x[..., 4].clone()

    # Log Returns
    close_shifted = torch.roll(close_prices, shifts=1, dims=1)
    close_shifted[:, 0] = close_prices[:, 0]
    log_returns = torch.log(close_prices / (close_shifted + 1e-9) + 1e-9)
    log_returns[:, 0] = 0.0
    engineered_features.append(log_returns.unsqueeze(-1))

    # Volume Z-Score
    vol_mean = volume.mean(dim=1, keepdim=True)
    vol_std = volume.std(dim=1, keepdim=True, unbiased=False)
    volume_zscore = (volume - vol_mean) / (vol_std + 1e-9)
    engineered_features.append(volume_zscore.unsqueeze(-1))

    final_tensor = torch.cat(engineered_features, dim=-1)
    return final_tensor, final_tensor.shape[-1]

def apply_mta_advanced_feature_engineering(
    x_dict,
    primary_tf='15m',
    required_tfs=('15m', '1h', '4h'),
    indicator_timeperiods={'ATR': 14, 'RSI': 14, 'MACD': (12, 26, 9), 'BBANDS': 20, 'STOCH': (14, 3, 3), 'ADX': 14}
):
    """Applies basic and advanced (TA-Lib) features across multiple timeframes."""
    if primary_tf not in x_dict or primary_tf not in required_tfs:
        raise ValueError(f"Primary timeframe '{primary_tf}' data is missing or not required.")
    for tf in required_tfs:
        if tf not in x_dict: raise ValueError(f"Required timeframe '{tf}' data not found.")
        if x_dict[tf].shape[-1] < 5: raise ValueError(f"Timeframe '{tf}' needs at least 5 features (OHLCV).")

    x_primary_data = x_dict[primary_tf].clone()
    batch_size, primary_seq_len, _ = x_primary_data.shape
    device = x_primary_data.device
    all_features_primary_seq = [] # Bắt đầu lại từ đầu để đảm bảo thứ tự

    # Thêm dữ liệu OHLCV gốc của primary TF trước
    all_features_primary_seq.append(x_primary_data)

    # Áp dụng basic features cho primary TF
    x_primary_basic, _ = apply_basic_feature_engineering(x_primary_data, 5)
    if x_primary_basic.shape[-1] > x_primary_data.shape[-1]:
         all_features_primary_seq.append(x_primary_basic[..., x_primary_data.shape[-1]:])

    # --- Tính toán và thêm các chỉ báo nâng cao (TA-Lib) ---
    if not talib:
        warnings.warn("TA-Lib not found. Skipping advanced indicators.")
        final_basic_tensor = torch.cat(all_features_primary_seq, dim=-1)
        return final_basic_tensor, final_basic_tensor.shape[-1]

    indicator_features = {} # Lưu trữ các tensor chỉ báo cuối cùng
    for tf in required_tfs:
        x_tf_data = x_dict[tf].clone()
        tf_seq_len = x_tf_data.shape[1]

        # Chuyển sang numpy trên CPU một lần
        open_np = x_tf_data[..., 0].cpu().numpy()
        high_np = x_tf_data[..., 1].cpu().numpy()
        low_np = x_tf_data[..., 2].cpu().numpy()
        close_np = x_tf_data[..., 3].cpu().numpy()
        volume_np = x_tf_data[..., 4].cpu().numpy()

        batch_indicators = { # Khởi tạo keys cho các chỉ báo tiềm năng
            f'ATR_{tf}': [], f'RSI_{tf}': [], f'MACD_{tf}': [], f'MACDhist_{tf}': [],
            f'BBands%B_{tf}': [], f'STOCHk_{tf}': [], f'STOCHd_{tf}': [], f'ADX_{tf}': []
        }

        for i in range(batch_size):
            # Sử dụng dict để lưu kết quả cho batch hiện tại
            current_batch_results = {}
            # Wrap TA-Lib calls trong try-except
            try: period = indicator_timeperiods.get('ATR', 14); current_batch_results[f'ATR_{tf}'] = talib.ATR(high_np[i], low_np[i], close_np[i], timeperiod=period)
            except Exception: current_batch_results[f'ATR_{tf}'] = np.full(tf_seq_len, np.nan)
            try: period = indicator_timeperiods.get('RSI', 14); current_batch_results[f'RSI_{tf}'] = talib.RSI(close_np[i], timeperiod=period)
            except Exception: current_batch_results[f'RSI_{tf}'] = np.full(tf_seq_len, np.nan)
            try: fp, sp, sigp = indicator_timeperiods.get('MACD', (12, 26, 9)); macd, macdsignal, macdhist = talib.MACD(close_np[i], fastperiod=fp, slowperiod=sp, signalperiod=sigp); current_batch_results[f'MACD_{tf}'] = macd; current_batch_results[f'MACDhist_{tf}'] = macdhist
            except Exception: current_batch_results[f'MACD_{tf}'] = np.full(tf_seq_len, np.nan); current_batch_results[f'MACDhist_{tf}'] = np.full(tf_seq_len, np.nan)
            try: period = indicator_timeperiods.get('BBANDS', 20); upper, _, lower = talib.BBANDS(close_np[i], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0); percent_b = (close_np[i] - lower) / (upper - lower + 1e-9); current_batch_results[f'BBands%B_{tf}'] = percent_b
            except Exception: current_batch_results[f'BBands%B_{tf}'] = np.full(tf_seq_len, np.nan)
            try: fp, sp_k, sp_d = indicator_timeperiods.get('STOCH', (14, 3, 3)); slowk, slowd = talib.STOCH(high_np[i], low_np[i], close_np[i], fastk_period=fp, slowk_period=sp_k, slowk_matype=0, slowd_period=sp_d, slowd_matype=0); current_batch_results[f'STOCHk_{tf}'] = slowk; current_batch_results[f'STOCHd_{tf}'] = slowd
            except Exception: current_batch_results[f'STOCHk_{tf}'] = np.full(tf_seq_len, np.nan); current_batch_results[f'STOCHd_{tf}'] = np.full(tf_seq_len, np.nan)
            try: period = indicator_timeperiods.get('ADX', 14); current_batch_results[f'ADX_{tf}'] = talib.ADX(high_np[i], low_np[i], close_np[i], timeperiod=period)
            except Exception: current_batch_results[f'ADX_{tf}'] = np.full(tf_seq_len, np.nan)

            # Thêm kết quả vào list tương ứng
            for key in batch_indicators.keys():
                 if key in current_batch_results:
                     batch_indicators[key].append(current_batch_results[key])
                 else: # Đảm bảo mọi list đều có cùng số lượng phần tử (batch_size)
                     warnings.warn(f"Indicator key {key} not found in batch results. Appending NaNs.")
                     batch_indicators[key].append(np.full(tf_seq_len, np.nan))

        # Xử lý từng chỉ báo sau khi tính xong cho cả batch
        for key, data_list in batch_indicators.items():
            if not data_list or len(data_list) != batch_size:
                 warnings.warn(f"Skipping indicator {key} due to empty data or batch size mismatch.")
                 continue

            try:
                batch_np = np.stack(data_list) # Shape: (batch_size, tf_seq_len)
                if tf_seq_len > primary_seq_len:
                    batch_np_aligned = batch_np[:, -primary_seq_len:]
                elif tf_seq_len < primary_seq_len:
                    pad_width = primary_seq_len - tf_seq_len
                    batch_np_aligned = np.pad(batch_np, ((0, 0), (pad_width, 0)), 'constant', constant_values=np.nan)
                else:
                    batch_np_aligned = batch_np
                tensor = torch.from_numpy(batch_np_aligned).float().to(device).unsqueeze(-1) # Shape: (batch_size, primary_seq_len, 1)
                indicator_features[key] = tensor
            except Exception as e:
                 warnings.warn(f"Error processing/aligning indicator {key}: {e}")

    # Thêm các tensor chỉ báo đã xử lý vào list features cuối cùng
    for key in sorted(indicator_features.keys()):
        if key in indicator_features and indicator_features[key].shape[1] == primary_seq_len:
            all_features_primary_seq.append(indicator_features[key])
        else:
             warnings.warn(f"Skipping indicator {key} due to previous error or final shape mismatch (Shape: {indicator_features.get(key, None)}).")

    # Ghép tất cả features lại
    try:
        final_features_tensor = torch.cat(all_features_primary_seq, dim=-1)
    except Exception as e:
        warnings.warn(f"Error concatenating final features: {e}. Returning basic features only.")
        final_features_tensor = torch.cat(all_features_primary_seq[:2], dim=-1) if len(all_features_primary_seq) >= 2 else all_features_primary_seq[0]

    # Fill NaNs cuối cùng
    final_features_tensor = torch.nan_to_num(final_features_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

    final_feature_dim = final_features_tensor.shape[-1]
    return final_features_tensor, final_feature_dim

# ==============================================================================
# 2. POSITIONAL ENCODING (Giữ nguyên)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Shape (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer

    def forward(self, x):
        """ Args: x: Tensor, shape [batch_size, seq_len, embedding_dim] """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ==============================================================================
# 3. MODELS (Giữ nguyên từ phiên bản trước)
# ==============================================================================
# --- Hybrid CNN-LSTM Model ---
class CryptoTradingModel(nn.Module):
    # ... (code __init__, _estimate_feature_dim, _init_output_layers, forward giữ nguyên) ...
    def __init__(self, config: Dict[str, Any], example_input_dict: Optional[Dict[str, torch.Tensor]] = None, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.input_dim = config['input_dim'] # Base input dim (e.g., 5 for OHLCV)
        self.hidden_dim = config['hidden_dim']
        self.num_regimes = config.get('num_regimes', DEFAULT_NUM_REGIMES)
        self.use_attention = config.get('use_attention', False)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.primary_tf = config.get('primary_tf', '15m')
        self.required_tfs = config.get('required_tfs', ('15m', '1h', '4h')) # <<< Cập nhật TFs
        self.device = device
        self.engineered_feature_dim = None

        # Dynamically determine engineered feature dimension
        if example_input_dict:
            try:
                example_input_dict_device = {
                    tf: data.to(self.device)
                    for tf, data in example_input_dict.items() if tf in self.required_tfs
                }
                if len(example_input_dict_device) == len(self.required_tfs):
                    _, self.engineered_feature_dim = apply_mta_advanced_feature_engineering(
                        example_input_dict_device,
                        primary_tf=self.primary_tf,
                        required_tfs=self.required_tfs
                    )
                    print(f"Hybrid Model: Dynamically determined engineered_feature_dim: {self.engineered_feature_dim}")
                else:
                    warnings.warn("Hybrid Model: Example input dict missing required timeframes for dynamic dim calculation. Estimating.")
                    self.engineered_feature_dim = self._estimate_feature_dim()
            except Exception as e:
                warnings.warn(f"Hybrid Model: Error determining feature dim dynamically: {e}. Falling back to estimation.")
                self.engineered_feature_dim = self._estimate_feature_dim()
        else:
            warnings.warn("Hybrid Model: No example_input_dict provided. Estimating feature dimension.")
            self.engineered_feature_dim = self._estimate_feature_dim()

        if not self.engineered_feature_dim or self.engineered_feature_dim <= 0:
             raise ValueError(f"Could not determine a valid engineered feature dimension (got {self.engineered_feature_dim}). Check feature engineering function or estimation.")

        # Define layers using the determined engineered_feature_dim
        self.cnn = nn.Conv1d(
            in_channels=self.engineered_feature_dim,
            out_channels=self.hidden_dim,
            kernel_size=config['cnn_kernel_size'],
            padding='same' # Keep sequence length the same
        )
        self.cnn_activation = nn.ReLU()
        self.cnn_dropout = nn.Dropout(self.dropout_rate)

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim, # Input to LSTM is output of CNN
            hidden_size=self.hidden_dim,
            num_layers=config['lstm_layers'],
            batch_first=True,
            dropout=self.dropout_rate if config['lstm_layers'] > 1 else 0
        )

        if self.use_attention:
            num_heads = config.get('num_heads', 4)
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=self.dropout_rate,
                batch_first=True
            )
        else:
            self.attention = None

        self.final_dropout = nn.Dropout(self.dropout_rate)
        self._init_output_layers()

    def _estimate_feature_dim(self) -> int:
        """Estimates feature dim if dynamic calculation fails."""
        num_base_primary = 5 + 2
        num_adv_per_tf = 8 if talib else 0
        est_dim = 5 + 2 + num_adv_per_tf * len(self.required_tfs)
        print(f"Hybrid Model: Estimated engineered_feature_dim: {est_dim}")
        return est_dim

    def _init_output_layers(self):
        """Initializes the output heads."""
        if self.num_regimes > 0:
            self.regime_head = nn.Linear(self.hidden_dim, self.num_regimes)
            print(f"Hybrid Model: Initialized regime head with output dim {self.num_regimes}")
        else:
            self.regime_head = None
            warnings.warn("Hybrid Model: num_regimes is 0 or not specified. Regime classification head disabled.")

        self.multi_step_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 4 * 5) # Predict 4 steps, 5 features (OHLCV) each
        )
        print(f"Hybrid Model: Initialized multi-step head for 4 steps.")


    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Forward pass for the Hybrid CNN-LSTM model. """
        x_eng, _ = apply_mta_advanced_feature_engineering(
            x_dict, primary_tf=self.primary_tf, required_tfs=self.required_tfs
        )
        cnn_input = x_eng.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_input)
        cnn_out = self.cnn_activation(cnn_out)
        cnn_out = self.cnn_dropout(cnn_out)

        lstm_input = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)

        attn_weights = None
        if self.use_attention and self.attention:
            attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            final_features = attn_output[:, -1, :]
        else:
            final_features = lstm_out[:, -1, :]

        final_features = self.final_dropout(final_features)
        outputs = {}

        if self.regime_head:
            regime_logits = self.regime_head(final_features)
            outputs['regime_logits'] = regime_logits
            outputs['regime_probabilities'] = F.softmax(regime_logits, dim=-1)

        multi_step_output = self.multi_step_head(final_features)
        outputs['multi_step_ohlcv'] = multi_step_output.view(-1, 4, 5)

        if attn_weights is not None:
            outputs['attention_weights'] = attn_weights

        return outputs

# --- Transformer Model for Decision Support (SL/RR) ---
class CryptoDecisionSupportModel(nn.Module):
    # ... (code __init__, forward giữ nguyên) ...
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.input_dim = config['input_dim'] # Base input dim (e.g., 5 for OHLCV)
        self.d_model = config['d_model'] # Transformer embedding dimension
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.sequence_length = config['sequence_length']
        self.num_rr_levels = config['num_rr_levels'] # Number of RR levels to predict probability for
        self.device = device

        self.engineered_feature_dim = self.input_dim + 2
        print(f"Decision Model: Using basic features. Expected input feature dim: {self.engineered_feature_dim}")

        self.input_embed = nn.Linear(self.engineered_feature_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate, self.sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate, batch_first=True, activation=F.gelu
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_encoder_layers, norm=encoder_norm
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        self.sl_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.GELU(),
            nn.Linear(self.d_model // 2, 1)
        )
        self.rr_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.GELU(),
            nn.Linear(self.d_model // 2, self.num_rr_levels)
        )
        print(f"Decision Model: Initialized SL head and RR head for {self.num_rr_levels} levels.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Forward pass for the Decision Support Transformer model. """
        x_eng, _ = apply_basic_feature_engineering(x, self.input_dim)
        x_emb = self.input_embed(x_eng)
        x_pos = self.pos_encoder(x_emb)
        transformer_output = self.transformer_encoder(x_pos)
        embedding = transformer_output[:, -1, :]
        embedding_dropped = self.dropout(embedding)
        optimal_sl_raw = self.sl_head(embedding_dropped)
        rr_logits = self.rr_head(embedding_dropped)
        rr_probabilities = torch.sigmoid(rr_logits)

        return {
            'embedding': embedding,
            'optimal_sl_raw': optimal_sl_raw,
            'rr_probabilities': rr_probabilities
        }

# ==============================================================================
# 4. MARKET CLASSIFICATION FUNCTION (Giữ nguyên từ phiên bản trước)
# ==============================================================================
# ... (classify_market_regime giữ nguyên) ...
def classify_market_regime(
    regime_probabilities: Optional[torch.Tensor], # Now optional
    current_data: Dict[str, pd.DataFrame],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Classifies the market into trend (up/down) or sideways with strength,
    using regime probabilities (if available) and technical indicators.
    """
    # --- Determine Base Regime and Confidence ---
    base_regime_type = "UNKNOWN"
    confidence = 0.5 # Default confidence if no probabilities

    if regime_probabilities is not None:
        if regime_probabilities.ndim > 1: regime_probabilities = regime_probabilities.squeeze(0)
        if regime_probabilities.ndim == 1 and len(regime_probabilities) == config.get('num_regimes', DEFAULT_NUM_REGIMES):
            regime_idx = torch.argmax(regime_probabilities).item()
            base_regime_type = REGIME_MAP.get(regime_idx, "UNKNOWN")
            confidence = regime_probabilities[regime_idx].item()
        else:
             warnings.warn(f"Invalid regime_probabilities shape: {regime_probabilities.shape}. Relying on indicators only.")
             base_regime_type = "UNKNOWN"
             confidence = 0.0

    # --- Get Data from Higher Timeframe ---
    higher_tf = '4h' if '4h' in current_data and not current_data['4h'].empty else \
                '1h' if '1h' in current_data and not current_data['1h'].empty else \
                '15m' # Fallback
    higher_tf_data = current_data.get(higher_tf)

    result = {
        'type': base_regime_type, 'strength': 'NORMAL',
        'confidence': confidence, 'indicators': {}
    }

    if higher_tf_data is None or higher_tf_data.empty or len(higher_tf_data) < 10:
        warnings.warn(f"Insufficient data on timeframe '{higher_tf}' for detailed market classification.")
        if base_regime_type == "UNKNOWN": result['strength'] = 'UNKNOWN'
        return result

    required_cols = ['close', 'volume', 'ADX', 'EMA_50', 'EMA_200', 'BB_upper', 'BB_lower', 'BB_middle']
    if not all(col in higher_tf_data.columns for col in required_cols):
         warnings.warn(f"Missing required columns in '{higher_tf}' data for classification.")
         if 'ADX' not in higher_tf_data.columns or 'BB_middle' not in higher_tf_data.columns:
              if base_regime_type == "UNKNOWN": result['strength'] = 'UNKNOWN'
              return result

    last_row = higher_tf_data.iloc[-1]; lookback = 5
    current_adx = last_row.get('ADX', 0); result['indicators']['adx'] = current_adx
    bb_middle = last_row.get('BB_middle', last_row.get('close', 0))
    bb_upper = last_row.get('BB_upper', bb_middle * 1.02)
    bb_lower = last_row.get('BB_lower', bb_middle * 0.98)
    bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-9); result['indicators']['bb_width'] = bb_width
    ma_slope = 0.0
    if 'EMA_50' in last_row and 'EMA_200' in last_row and len(higher_tf_data) > lookback:
        try:
            ma_diff = last_row['EMA_50'] - last_row['EMA_200']
            prev_ma_diff = higher_tf_data['EMA_50'].iloc[-lookback] - higher_tf_data['EMA_200'].iloc[-lookback]
            ma_slope = (ma_diff - prev_ma_diff) / (last_row['close'] + 1e-9)
        except Exception as e: warnings.warn(f"Error calculating MA slope on {higher_tf}: {e}")
    result['indicators']['ma_slope'] = ma_slope
    volume_change = 1.0; volume_cv = 0.5
    if 'volume' in last_row and len(higher_tf_data) > lookback:
        try:
            recent_volume_mean = higher_tf_data['volume'].iloc[-lookback:].mean()
            prev_volume_mean = higher_tf_data['volume'].iloc[-lookback*2:-lookback].mean()
            volume_change = recent_volume_mean / (prev_volume_mean + 1e-9)
            volume_std_roll = higher_tf_data['volume'].rolling(10).std().iloc[-1]
            volume_mean_roll = higher_tf_data['volume'].rolling(10).mean().iloc[-1]
            volume_cv = volume_std_roll / (volume_mean_roll + 1e-9)
        except Exception as e: warnings.warn(f"Error calculating Volume change/CV on {higher_tf}: {e}")
    result['indicators']['volume_change'] = volume_change; result['indicators']['volume_cv'] = volume_cv

    indicator_based_type = "UNKNOWN"; strength_score = 0
    adx_strong = config.get('adx_strong_thresh', 25); adx_very_strong = config.get('adx_very_strong_thresh', 35)
    sideway_adx = config.get('sideway_adx_thresh', 20); ma_slope_thresh_strong = config.get('ma_slope_thresh_strong', 0.005)
    ma_slope_thresh_normal = config.get('ma_slope_thresh_normal', 0.002); trend_vol_change = config.get('trend_volume_change_thresh', 1.15)
    sideway_vol_cv = config.get('sideway_volume_cv_thresh', 0.35); sideway_bbw = config.get('sideway_bb_width_thresh', 0.04)
    trend_bbw_expand = config.get('trend_bb_width_expand_thresh', 0.06)

    if current_adx >= adx_strong:
        if ma_slope > ma_slope_thresh_normal: indicator_based_type = "TREND_UP"
        elif ma_slope < -ma_slope_thresh_normal: indicator_based_type = "TREND_DOWN"
        else: indicator_based_type = base_regime_type if "TREND" in base_regime_type else "UNKNOWN"; strength_score -= 0.5
    elif current_adx < sideway_adx: indicator_based_type = "SIDEWAYS"
    else:
        if ma_slope > ma_slope_thresh_normal: indicator_based_type = "TREND_UP"
        elif ma_slope < -ma_slope_thresh_normal: indicator_based_type = "TREND_DOWN"
        else: indicator_based_type = "SIDEWAYS" if base_regime_type == "SIDEWAYS" else "UNKNOWN"

    if indicator_based_type != "UNKNOWN": result['type'] = indicator_based_type

    if result['type'] == "TREND_UP": trend_direction = 1
    elif result['type'] == "TREND_DOWN": trend_direction = -1
    else: trend_direction = 0 # Sideways

    if trend_direction != 0: # TREND
        if current_adx > adx_very_strong: strength_score += 2.0
        elif current_adx > adx_strong: strength_score += 1.5
        if abs(ma_slope) > ma_slope_thresh_strong: strength_score += 1.5 # Check absolute slope for strength
        elif abs(ma_slope) > ma_slope_thresh_normal: strength_score += 1.0
        if volume_change > trend_vol_change : strength_score += 1.0
        if bb_width > trend_bbw_expand: strength_score += 0.5
        very_strong_thresh = config.get('trend_very_strong_thresh', 4.0); strong_thresh = config.get('trend_strong_thresh', 2.5)
        if strength_score >= very_strong_thresh: result['strength'] = "VERY_STRONG"
        elif strength_score >= strong_thresh: result['strength'] = "STRONG"
        else: result['strength'] = "NORMAL"
    elif result['type'] == "SIDEWAYS":
        if current_adx < sideway_adx: strength_score += 1.5
        if bb_width < sideway_bbw: strength_score += 1.5
        if volume_cv < sideway_vol_cv: strength_score += 1.0
        strong_thresh = config.get('sideway_strong_thresh', 3.0); normal_thresh = config.get('sideway_normal_thresh', 1.5)
        if strength_score >= strong_thresh: result['strength'] = "STRONG"
        elif strength_score >= normal_thresh: result['strength'] = "NORMAL"
        else: result['strength'] = "WEAK"
    else: result['strength'] = "UNKNOWN"

    return result

# ==============================================================================
# 5. COMBINED TRADING AGENT (Cập nhật toàn diện với logic mới)
# ==============================================================================
class CombinedTradingAgent:
    """
    Agent combining models, generating entry signals with SL, and managing positions
    using enhanced regime-aware trailing TP logic including conflict handling,
    confidence usage, regime change exits, and dynamic R:R checks.
    """
    original_classify_market_regime = staticmethod(classify_market_regime)

    def __init__(self,
                 decision_model: CryptoDecisionSupportModel,
                 hybrid_model: CryptoTradingModel,
                 bot_config: Dict[str, Any],
                 entry_optimizer_model: Optional[Any],
                 xgboost_models: Dict[str, Any],
                 xgboost_scalers: Dict[str, Any],
                 get_sac_state_func: Callable,
                 scale_sac_state_func: Callable,
                 get_dynamic_adx_func: Callable,
                 temporal_features: List[str],
                 min_rr_level_for_check: int = 1,
                 max_rr_level_for_check: int = 10,
                 min_sl_atr_multiplier: float = 0.5,
                 max_sl_atr_multiplier: float = 5.0,
                 regime_map: Dict[int, str] = REGIME_MAP):
        """ Initializes agent with models, bot components, and enhanced regime-aware TP parameters. """
        self.decision_model = decision_model
        self.hybrid_model = hybrid_model
        self.config = bot_config
        self.entry_optimizer = entry_optimizer_model
        self.xgboost_models = xgboost_models
        self.xgboost_scalers = xgboost_scalers
        self._get_sac_state_unscaled_external = get_sac_state_func
        self._scale_sac_state_external = scale_sac_state_func
        self._get_dynamic_adx_threshold_external = get_dynamic_adx_func
        self.temporal_features = temporal_features
        self.regime_map = regime_map

        # --- Load Agent Parameters (Allowing Overrides) ---
        self.min_sl_atr_multiplier = self.config.get('min_sl_atr_multiplier', min_sl_atr_multiplier)
        self.max_sl_atr_multiplier = self.config.get('max_sl_atr_multiplier', max_sl_atr_multiplier)

        num_predicted_levels = getattr(decision_model, 'num_rr_levels', 0)
        if num_predicted_levels <= 0: raise ValueError("Decision model needs 'num_rr_levels' attribute > 0.")
        self.min_rr_level_for_check = max(1, self.config.get('min_rr_level_for_check', min_rr_level_for_check))
        self.max_rr_level_for_check = min(num_predicted_levels, self.config.get('max_rr_level_for_check', max_rr_level_for_check))
        if self.min_rr_level_for_check > self.max_rr_level_for_check:
             raise ValueError(f"min_rr_level ({self.min_rr_level_for_check}) cannot be greater than max_rr_level ({self.max_rr_level_for_check})")

        # --- Load Enhanced TP Thresholds & Configs ---
        # 1. RR Prob Thresholds (Regime & Confidence Aware)
        self.regime_aware_rr_prob_thresholds = self.config.get('regime_aware_rr_prob_thresholds', {
            'TREND_VERY_STRONG': 0.55, 'TREND_STRONG': 0.58, 'TREND_NORMAL': 0.60,
            'SIDEWAYS_STRONG': 0.65, 'SIDEWAYS_NORMAL': 0.65, 'SIDEWAYS_WEAK': 0.70,
            'UNKNOWN': 0.75 # <<< More conservative for UNKNOWN
        })
        self.conservative_rr_prob_threshold = self.config.get('conservative_rr_prob_threshold', 0.75) # <<< For Conflict/Unknown override
        self.low_confidence_threshold = self.config.get('low_confidence_threshold', 0.60)
        self.confidence_penalty_factor = self.config.get('confidence_penalty_factor', 1.15) # <<< Increase RR prob requirement by 15% if confidence is low

        # 2. Hybrid Pass Configs
        self.hybrid_trend_strong_pass_config = self.config.get('hybrid_trend_strong_pass', {'min_steps_confirm': 3, 'max_pullback_ratio': 0.3, 'min_target_cross': 1})
        self.hybrid_trend_normal_pass_config = self.config.get('hybrid_trend_normal_pass', {'min_steps_confirm': 2, 'max_pullback_ratio': 0.5, 'min_target_cross': 1})
        self.hybrid_sideways_pass_config = self.config.get('hybrid_sideways_pass', {'min_target_cross': 1}) # Simplest check
        self.conservative_hybrid_pass_config = self.hybrid_sideways_pass_config # <<< Use simplest check for Conflict/Unknown

        # 3. Floor Lock Thresholds
        self.lock_floor_trend_very_strong_rr_thresh = self.config.get('lock_floor_trend_very_strong_rr_thresh', 0.80)
        self.lock_floor_trend_strong_rr_thresh = self.config.get('lock_floor_trend_strong_rr_thresh', 0.85)
        self.lock_floor_trend_normal_rr_thresh = self.config.get('lock_floor_trend_normal_rr_thresh', 0.90)
        self.lock_floor_sideways_rr_thresh = self.config.get('lock_floor_sideways_rr_thresh', 0.98)

        # 4. Regime Change Exit Config
        # Defines pairs that trigger exit, e.g., ("TREND", "SIDEWAYS") means exit if entry was Trend and current is Sideways
        self.significant_regime_change_def = self.config.get('significant_regime_change_def', [
            ("TREND", "SIDEWAYS"), # Trend -> Sideways
            ("SIDEWAYS", "TREND"), # Sideways -> Trend 
            ("TREND_UP", "TREND_DOWN"), # Trend reversal
            ("TREND_DOWN", "TREND_UP")  # Trend reversal
        ])

        # 5. Dynamic R:R Exit Config
        self.enable_dynamic_rr_exit = self.config.get('enable_dynamic_rr_exit', True) # <<< Enable/Disable feature
        self.exit_threshold_dynamic_RR = self.config.get('exit_threshold_dynamic_RR', 1.0) # <<< Exit if Potential Reward < 1.0 * Risk_to_Floor
        self.min_risk_to_floor_ratio = self.config.get('min_risk_to_floor_ratio', 0.1) # <<< Min risk (vs 1R) needed to activate check

        print("--- CombinedTradingAgent Initialized (Enhanced Logic) ---")
        print(f"TP Check Range: RR {self.min_rr_level_for_check} to {self.max_rr_level_for_check}")
        print(f"SL ATR Multiplier Range: [{self.min_sl_atr_multiplier}, {self.max_sl_atr_multiplier}]")
        print(f"Regime-Aware RR Prob Thresholds Base: {self.regime_aware_rr_prob_thresholds}")
        print(f"Low Confidence Threshold: {self.low_confidence_threshold}, Penalty Factor: {self.confidence_penalty_factor}")
        print(f"Conservative RR Prob Thresh (Conflict/Unknown): {self.conservative_rr_prob_threshold}")
        print(f"Hybrid Pass Configs: Loaded (Conservative uses Sideways config)")
        print(f"TP Floor Lock RR Thresholds: Loaded")
        print(f"Significant Regime Change Exit Defs: {self.significant_regime_change_def}")
        print(f"Dynamic R:R Exit Enabled: {self.enable_dynamic_rr_exit}, Threshold: {self.exit_threshold_dynamic_RR}, Min Risk Ratio: {self.min_risk_to_floor_ratio}")
        print("-" * 60)

        if self.decision_model: self.decision_model.eval()
        if self.hybrid_model: self.hybrid_model.eval()


    @staticmethod
    def _calculate_rr_price(entry: float, sl: float, level: int, p_type: str) -> Optional[float]:
        # ... (code giữ nguyên) ...
        risk = 0.0
        if p_type == 'long': risk = entry - sl
        elif p_type == 'short': risk = sl - entry
        else: warnings.warn(f"Unknown position type: {p_type}"); return None
        if risk <= 1e-9: return None
        target = entry + level * risk if p_type == 'long' else entry - level * risk
        return max(0.0, target) if target is not None else None


    def _check_hybrid_passes(self, multi_step_pred: Optional[torch.Tensor], rr_price_target: Optional[float],
                             position_type: str, market_classification: Dict[str, Any],
                             use_conservative_mode: bool = False) -> bool: # <<< Thêm cờ conservative
        """ Checks hybrid prediction support, uses conservative mode if specified. """
        if rr_price_target is None or multi_step_pred is None or multi_step_pred.nelement() < 5 or multi_step_pred.dim() != 2 or multi_step_pred.shape[0] < 2:
             return False

        regime_type = market_classification.get('type', 'UNKNOWN')
        strength = market_classification.get('strength', 'NORMAL')

        # <<< Chọn config phù hợp, ưu tiên conservative nếu cờ được bật >>>
        if use_conservative_mode:
             cfg = self.conservative_hybrid_pass_config
             # print("--- Hybrid Check (Conservative Mode Active) ---")
        elif "TREND" in regime_type:
            cfg = self.hybrid_trend_strong_pass_config if strength in ["VERY_STRONG", "STRONG"] else self.hybrid_trend_normal_pass_config
        elif regime_type == "SIDEWAYS":
            cfg = self.hybrid_sideways_pass_config
        else: # UNKNOWN
            cfg = self.conservative_hybrid_pass_config # Dùng conservative cho UNKNOWN

        try:
            # ... (logic tính toán prices_to_check, target_comparison, confirm_direction, pullback_check giữ nguyên) ...
            if position_type == 'long':
                prices_to_check = multi_step_pred[:, 1]; initial_price = multi_step_pred[0, 3]
                target_comparison = lambda p: p > rr_price_target; confirm_direction = lambda diff: diff > 0
                pullback_check = lambda prices: (initial_price - torch.min(multi_step_pred[:, 2])) / (rr_price_target - initial_price + 1e-9)
            elif position_type == 'short':
                prices_to_check = multi_step_pred[:, 2]; initial_price = multi_step_pred[0, 3]
                target_comparison = lambda p: p < rr_price_target; confirm_direction = lambda diff: diff < 0
                pullback_check = lambda prices: (torch.max(multi_step_pred[:, 1]) - initial_price) / (initial_price - rr_price_target + 1e-9)
            else: return False

            crossed_target_mask = target_comparison(prices_to_check)
            crossed_target_count = torch.sum(crossed_target_mask).item()
            if crossed_target_count == 0: return False

            # --- Áp dụng kiểm tra dựa trên config đã chọn (cfg) ---
            if 'min_steps_confirm' in cfg: # Nếu là kiểm tra Trend (có steps và pullback)
                diffs = prices_to_check[1:] - prices_to_check[:-1]
                steps_confirmed = torch.sum(confirm_direction(diffs)).item()
                target_range = abs(rr_price_target - initial_price)
                max_pullback_ratio = pullback_check(prices_to_check).item() if target_range > 1e-9 else 0.0
                passes = (crossed_target_count >= cfg['min_target_cross'] and
                          steps_confirmed >= cfg['min_steps_confirm'] and
                          max_pullback_ratio >= 0 and
                          max_pullback_ratio <= cfg['max_pullback_ratio'])
            else: # Nếu là kiểm tra Sideways hoặc Conservative (chỉ cần target cross)
                passes = (crossed_target_count >= cfg['min_target_cross'])

            return passes

        except Exception as e:
            warnings.warn(f"_check_hybrid_passes: Error during check - {e}")
            return False


    def _check_very_strong_signal(self, rr_probs: Optional[torch.Tensor], multi_step_pred: Optional[torch.Tensor],
                                  k_raised_to: int, position_type: str, market_classification: Dict[str, Any]) -> bool:
        """ Checks for very strong signal, considering market indicators (ADX). """
        # <<< Logic bên trong giữ nguyên như phiên bản trước (đã có check ADX) >>>
        if rr_probs is None or not (0 <= k_raised_to < len(rr_probs)): return False
        regime_type = market_classification.get('type', 'UNKNOWN')
        strength = market_classification.get('strength', 'NORMAL')
        prob_next_rr = rr_probs[k_raised_to].item()
        current_adx = market_classification.get('indicators', {}).get('adx', 0)
        hybrid_monotonic = False
        if multi_step_pred is not None and multi_step_pred.dim()==2 and multi_step_pred.shape[0] > 1:
            try:
                if position_type == 'long': prices = multi_step_pred[:, 1]; diffs = prices[1:] - prices[:-1]; hybrid_monotonic = torch.all(diffs > 0).item()
                elif position_type == 'short': prices = multi_step_pred[:, 2]; diffs = prices[1:] - prices[:-1]; hybrid_monotonic = torch.all(diffs < 0).item()
            except Exception as e: warnings.warn(f"Error checking hybrid monotonicity: {e}")

        try:
            if "TREND" in regime_type:
                adx_strong_confirm_thresh = self.config.get('adx_strong_thresh', 25)
                adx_basic_confirm_thresh = self.config.get('adx_thresholds', {}).get(self.config.get('primary_tf','15m'), 20)
                if strength == "VERY_STRONG":
                    required_rr_prob = self.lock_floor_trend_very_strong_rr_thresh
                    condition_met = (prob_next_rr >= required_rr_prob or hybrid_monotonic) and (current_adx >= adx_strong_confirm_thresh)
                    if condition_met: print(f"--- Lock Floor Check (TREND_VERY_STRONG): PASSED (Prob={prob_next_rr:.2f}/{required_rr_prob:.2f} or HybridMono={hybrid_monotonic}) AND (ADX={current_adx:.1f}/{adx_strong_confirm_thresh:.1f})")
                    return condition_met
                elif strength == "STRONG":
                    required_rr_prob = self.lock_floor_trend_strong_rr_thresh
                    condition_met = (prob_next_rr >= required_rr_prob or hybrid_monotonic) and (current_adx >= adx_strong_confirm_thresh)
                    if condition_met: print(f"--- Lock Floor Check (TREND_STRONG): PASSED (Prob={prob_next_rr:.2f}/{required_rr_prob:.2f} or HybridMono={hybrid_monotonic}) AND (ADX={current_adx:.1f}/{adx_strong_confirm_thresh:.1f})")
                    return condition_met
                elif strength == "NORMAL":
                    required_rr_prob = self.lock_floor_trend_normal_rr_thresh
                    condition_met = (prob_next_rr >= required_rr_prob) and (current_adx >= adx_basic_confirm_thresh)
                    if condition_met: print(f"--- Lock Floor Check (TREND_NORMAL): PASSED (Prob={prob_next_rr:.2f}/{required_rr_prob:.2f}) AND (ADX={current_adx:.1f}/{adx_basic_confirm_thresh:.1f})")
                    return condition_met
            elif regime_type == "SIDEWAYS":
                required_rr_prob = self.lock_floor_sideways_rr_thresh
                adx_sideways_confirm_thresh = self.config.get('sideway_adx_thresh', 20)
                condition_met = (prob_next_rr >= required_rr_prob) and (current_adx < adx_sideways_confirm_thresh)
                if condition_met: print(f"--- Lock Floor Check (SIDEWAYS): PASSED (Prob={prob_next_rr:.2f}/{required_rr_prob:.2f}) AND (ADX={current_adx:.1f} < {adx_sideways_confirm_thresh:.1f})")
                return condition_met
            # print(f"--- Lock Floor Check ({regime_type}/{strength}): FAILED (Prob={prob_next_rr:.2f}, HybridMono={hybrid_monotonic}, ADX={current_adx:.1f})")
            return False
        except Exception as e: warnings.warn(f"_check_very_strong_signal: Error during check - {e}"); return False


    def _generate_entry_signal(self, symbol: str, current_data: Dict[str, pd.DataFrame],
                              embedding: Optional[torch.Tensor], optimal_sl_raw: Optional[float],
                              market_classification: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ Generates entry signal, including market classification at entry. """
        # ... (logic lấy tín hiệu SAC, XGBoost, tính score, final_direction như trước) ...
        # ... (logic tính SL dựa trên optimal_sl_raw và current_atr như trước) ...

        primary_tf = self.config.get('primary_tf', '15m'); df_primary = current_data.get(primary_tf)
        if df_primary is None or df_primary.empty or len(df_primary) < 2: return None
        try:
            last_row = df_primary.iloc[-1]; required_cols = ["close", "ATR", "ADX"]
            if any(col not in last_row or pd.isna(last_row[col]) for col in required_cols): return None
            entry_price = last_row["close"]; current_atr = last_row["ATR"]; current_adx = last_row["ADX"]; timestamp = last_row.name
            if entry_price <= 0 or current_atr <= 0: return None
        except Exception as e: warnings.warn(f"_generate_entry_signal ({symbol}): Error data access: {e}"); return None

        # --- Get SAC Signal ---
        sac_direction = None
        if self.entry_optimizer and self._get_sac_state_unscaled_external:
            try:
                state_unscaled = self._get_sac_state_unscaled_external(symbol, current_data=current_data, embedding=embedding)
                if state_unscaled is not None and self._scale_sac_state_external:
                    state_scaled = self._scale_sac_state_external(state_unscaled)
                    if state_scaled is not None:
                        action_tuple = self.entry_optimizer.predict(state_scaled, deterministic=True); action_value = action_tuple[0].item()
                        long_thresh = self.config.get("sac_long_threshold", 0.1); short_thresh = self.config.get("sac_short_threshold", -0.1)
                        if action_value > long_thresh: sac_direction = "LONG"
                        elif action_value < short_thresh: sac_direction = "SHORT"
            except Exception as sac_e: warnings.warn(f"Error SAC {symbol}: {sac_e}")
        # --- Get XGBoost Signal ---
        xgb_win_prob = 0.5; xgb_direction = None
        if symbol in self.xgboost_models and symbol in self.xgboost_scalers:
            # ... (logic lấy feature, scale, predict XGB như trước) ...
             model = self.xgboost_models[symbol]; scaler = self.xgboost_scalers[symbol]
             try:
                 features_base_xgb = self.config.get('xgboost_feature_columns', []); temporal_features_exist = [f for f in self.temporal_features if f in last_row]; required_xgb_features = features_base_xgb + temporal_features_exist
                 if any(f not in last_row or pd.isna(last_row[f]) for f in required_xgb_features): raise ValueError("Missing XGB features")
                 current_features_no_emb_s = last_row[required_xgb_features].fillna(0); current_features_no_emb = current_features_no_emb_s.values.astype(float)
                 if embedding is not None:
                     embedding_np = embedding.detach().cpu().numpy().flatten()
                     if current_features_no_emb.ndim == 1 and embedding_np.ndim == 1: current_features_with_emb = np.concatenate((current_features_no_emb, embedding_np))
                     else: current_features_with_emb = current_features_no_emb
                 else: current_features_with_emb = current_features_no_emb
                 current_features_reshaped = current_features_with_emb.reshape(1, -1)
                 if hasattr(scaler, 'n_features_in_') and current_features_reshaped.shape[1] == scaler.n_features_in_:
                     scaled_features = scaler.transform(current_features_reshaped); xgb_pred_proba = model.predict_proba(scaled_features)
                     xgb_win_prob_raw = xgb_pred_proba[0][1]; long_threshold = self.config.get("xgb_long_threshold", 0.60); short_threshold = self.config.get("xgb_short_threshold", 0.40)
                     if xgb_win_prob_raw >= long_threshold: xgb_direction = "LONG"; xgb_win_prob = xgb_win_prob_raw
                     elif xgb_win_prob_raw <= short_threshold: xgb_direction = "SHORT"; xgb_win_prob = 1.0 - xgb_win_prob_raw
                 else: warnings.warn(f"Feature mismatch XGB {symbol}")
             except Exception as xgb_e: warnings.warn(f"Error XGB {symbol}: {xgb_e}")

        # --- Combine Signals (using Market Classification) ---
        final_direction = None; signal_quality = 'NONE'; final_win_prob = 0.5
        adaptive_weights = self.config.get('ADAPTIVE_WEIGHTS', {'default': {'sac': 0.5, 'xgb': 0.5}})
        regime_type = market_classification.get('type', 'UNKNOWN'); strength = market_classification.get('strength', 'NORMAL')
        bb_width = market_classification.get('indicators', {}).get('bb_width', 0); bb_vol_thresh = self.config.get('bb_width_vol_threshold', 0.05)
        weight_key = 'default'
        if "TREND" in regime_type: weight_key = 'trending_strong' if strength in ["VERY_STRONG", "STRONG"] and 'trending_strong' in adaptive_weights else 'trending'
        elif regime_type == "SIDEWAYS": weight_key = 'sideways_high_vol' if bb_width > bb_vol_thresh and 'sideways_high_vol' in adaptive_weights else ('sideways_low_vol' if 'sideways_low_vol' in adaptive_weights else 'default')
        if weight_key not in adaptive_weights: weight_key = 'default'
        w_sac = adaptive_weights[weight_key].get('sac', 0.5); w_xgb = adaptive_weights[weight_key].get('xgb', 0.5)
        # print(f"[{symbol}] Entry Weights: Key='{weight_key}', SAC={w_sac:.2f}, XGB={w_xgb:.2f}")
        sac_score = 1.0 if sac_direction == "LONG" else -1.0 if sac_direction == "SHORT" else 0.0; xgb_score = 0.0
        if xgb_direction == "LONG": xgb_score = (xgb_win_prob - 0.5) * 2; 
        elif xgb_direction == "SHORT": xgb_score = -(xgb_win_prob - 0.5) * 2
        score_long = max(0, sac_score * w_sac) + max(0, xgb_score * w_xgb); score_short = max(0, -sac_score * w_sac) + max(0, -xgb_score * w_xgb)
        strong_threshold = self.config.get('signal_strong_threshold', 0.75); medium_threshold = self.config.get('signal_medium_threshold', 0.65); weak_threshold = self.config.get('signal_weak_threshold', 0.55)
        if score_long >= strong_threshold: final_direction = "LONG"; signal_quality = 'STRONG'; final_win_prob = 0.5 + score_long / 2
        elif score_short >= strong_threshold: final_direction = "SHORT"; signal_quality = 'STRONG'; final_win_prob = 0.5 + score_short / 2
        elif score_long >= medium_threshold: final_direction = "LONG"; signal_quality = 'MEDIUM'; final_win_prob = 0.5 + score_long / 2
        elif score_short >= medium_threshold: final_direction = "SHORT"; signal_quality = 'MEDIUM'; final_win_prob = 0.5 + score_short / 2
        elif score_long >= weak_threshold: final_direction = "LONG"; signal_quality = 'WEAK'; final_win_prob = 0.5 + score_long / 2
        elif score_short >= weak_threshold: final_direction = "SHORT"; signal_quality = 'WEAK'; final_win_prob = 0.5 + score_short / 2
        elif sac_direction and xgb_direction and sac_direction != xgb_direction: signal_quality = 'CONFLICT'
        else: signal_quality = 'NONE'

        # --- Calculate SL & Finalize ---
        if final_direction:
            potential_sl_price = None; sl_distance = None; calculated_atr_multiplier = None
            if optimal_sl_raw is not None and current_atr > 0:
                 sl_atr_multiplier = np.clip(abs(optimal_sl_raw), self.min_sl_atr_multiplier, self.max_sl_atr_multiplier)
                 calculated_atr_multiplier = sl_atr_multiplier; sl_distance = sl_atr_multiplier * current_atr
                 if final_direction == "LONG": potential_sl_price = entry_price - sl_distance
                 elif final_direction == "SHORT": potential_sl_price = entry_price + sl_distance
                 if potential_sl_price is not None and potential_sl_price <= 0 and final_direction == "LONG": potential_sl_price = None
            if potential_sl_price is not None:
                return {
                    "direction": final_direction, "win_prob": np.clip(final_win_prob, 0.01, 0.99),
                    "signal_quality": signal_quality, "entry_price": entry_price,
                    "potential_sl_price": potential_sl_price, "sl_distance": sl_distance,
                    "sl_atr_multiplier_used": calculated_atr_multiplier,
                    "atr": current_atr, "adx": current_adx, "timestamp": timestamp, "symbol": symbol,
                    "market_classification_at_entry": copy.deepcopy(market_classification), # <<< Store deep copy
                }
        return None


    @torch.no_grad()
    def get_trade_signals(self,
                          symbol: str,
                          x_decision: torch.Tensor,
                          x_hybrid_dict: Dict[str, torch.Tensor],
                          current_price: float,
                          current_data: Dict[str, pd.DataFrame],
                          open_position: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
        """
        Generates entry signals and manages open positions with ENHANCED regime-aware logic.
        (Sửa lỗi logic tính ngưỡng, kiểm tra regime change, và khởi tạo debug dict)
        """
        # --- 1. Get Model Outputs & Basic Info ---
        # ... (lấy output model như trước) ...
        support_outputs = self.decision_model(x_decision) if self.decision_model else {}
        hybrid_outputs = self.hybrid_model(x_hybrid_dict) if self.hybrid_model else {}
        embedding = support_outputs.get('embedding').squeeze(0) if support_outputs.get('embedding') is not None else None
        optimal_sl_raw = support_outputs.get('optimal_sl_raw').squeeze(0).item() if support_outputs.get('optimal_sl_raw') is not None else None
        rr_probs = support_outputs.get('rr_probabilities').squeeze(0) if support_outputs.get('rr_probabilities') is not None else None
        multi_step_pred = hybrid_outputs.get('multi_step_ohlcv').squeeze(0) if hybrid_outputs.get('multi_step_ohlcv') is not None else None
        if multi_step_pred is not None and multi_step_pred.dim() != 2: multi_step_pred = None
        regime_probabilities = hybrid_outputs.get('regime_probabilities')
        if regime_probabilities is not None and regime_probabilities.ndim > 1: regime_probabilities = regime_probabilities.squeeze(0)

        # --- 2. Detailed Market Classification ---
        # <<< Quan trọng: Nếu test case có market_classification_override, sử dụng nó >>>
        global sim_market_classification_override # Access the global variable set by set_sim_outputs
        if sim_market_classification_override is not None:
            market_classification = sim_market_classification_override
            # Lấy lại các biến từ override để đảm bảo nhất quán
            market_confidence = market_classification.get('confidence', 0.5)
            market_type = market_classification.get('type', 'UNKNOWN')
            market_strength = market_classification.get('strength', 'NORMAL')
            print(f"[{symbol}] Using OVERRIDDEN Market Classification: {market_type}/{market_strength} (Conf: {market_confidence:.2f})")
        else:
            # Nếu không có override, tính toán như bình thường
            market_classification = self.original_classify_market_regime(
                 regime_probabilities, current_data, self.config
            )
            market_confidence = market_classification.get('confidence', 0.5)
            market_type = market_classification.get('type', 'UNKNOWN')
            market_strength = market_classification.get('strength', 'NORMAL')

        # --- 2.5 Check for Regime Conflict (Hybrid vs Indicator) ---
        is_conflicted = False
        predicted_regime_from_hybrid = "UNKNOWN"
        if regime_probabilities is not None and regime_probabilities.ndim == 1:
            try: # Thêm try-except phòng trường hợp lỗi tensor
                predicted_regime_index = torch.argmax(regime_probabilities).item()
                predicted_regime_from_hybrid = self.regime_map.get(predicted_regime_index, "UNKNOWN")
            except Exception as e:
                warnings.warn(f"Error determining predicted_regime_from_hybrid: {e}")

        # <<< Sửa lỗi logic so sánh: Chỉ conflict nếu một bên là TREND và một bên là SIDEWAYS >>>
        if predicted_regime_from_hybrid != "UNKNOWN" and market_type != "UNKNOWN":
            is_trend_hybrid = "TREND" in predicted_regime_from_hybrid
            is_sideways_hybrid = "SIDEWAYS" in predicted_regime_from_hybrid
            is_trend_indicator = "TREND" in market_type
            is_sideways_indicator = "SIDEWAYS" in market_type
            # Chỉ coi là conflict nếu một bên là Trend và một bên là Sideways
            if (is_trend_hybrid and is_sideways_indicator) or (is_sideways_hybrid and is_trend_indicator):
                is_conflicted = True
                warnings.warn(f"[{symbol}] Regime Conflict Detected: Hybrid='{predicted_regime_from_hybrid}', Indicator='{market_type}/{market_strength}'. Applying conservative TP logic.")


        # --- 3. Initialize Signals Dictionary ---
        signals = {
            'embedding': embedding, 'optimal_sl_raw': optimal_sl_raw, 'rr_probabilities': rr_probs,
            'multi_step_ohlcv': multi_step_pred, 'predicted_regime': predicted_regime_from_hybrid,
            'regime_probabilities': regime_probabilities, 'market_classification': market_classification,
            'entry_signal': None, 'exit_signal': 'hold',
            'trailing_tp_level': 0, 'min_locked_tp_level': 0,
            'potential_target_rr': 0, 'current_rr': 0,
            # <<< Sửa lỗi: Khởi tạo keys với giá trị mặc định None hoặc 0.0 >>>
            'debug_tp_logic': {
                'rr_prob_threshold_used': None,
                'dynamic_RR': None,
                'risk_to_floor': None,
                'potential_reward': None
            }
        }

        # --- 4. Manage Open Position ---
        if open_position and isinstance(open_position, dict) and \
           all(k in open_position for k in ['type', 'entry_price', 'stop_loss_price']) and \
           rr_probs is not None:

            entry_price: float = open_position['entry_price']
            sl_price: float = open_position['stop_loss_price']
            position_type: str = open_position['type']
            current_trailing_tp_level: int = open_position.get('trailing_tp_level', 0)
            min_locked_tp_level: int = open_position.get('min_locked_tp_level', 0)
            market_classification_at_entry = open_position.get('market_classification_at_entry')
            new_trailing_tp_level = current_trailing_tp_level
            new_min_locked_tp_level = min_locked_tp_level
            current_rr = 0
            initial_risk_amount = abs(entry_price - sl_price) if sl_price is not None else 0


            # --- 4a. Check for Significant Regime Change Exit ---
            # <<< Sửa lỗi logic kiểm tra và xử lý UNKNOWN >>>
            if market_classification_at_entry and initial_risk_amount > 1e-9 :
                entry_type = market_classification_at_entry.get('type', 'UNKNOWN')
                current_type = market_type # Đã lấy ở đầu hàm, đã xử lý override
                regime_changed_significantly = False

                # Không coi là thay đổi nếu trạng thái hiện tại là UNKNOWN hoặc entry là UNKNOWN
                if entry_type != "UNKNOWN" and current_type != "UNKNOWN":
                    entry_base = "TREND" if "TREND" in entry_type else "SIDEWAYS" if "SIDEWAYS" in entry_type else "OTHER"
                    current_base = "TREND" if "TREND" in current_type else "SIDEWAYS" if "SIDEWAYS" in current_type else "OTHER"

                    for type1_def, type2_def in self.significant_regime_change_def:
                        # Check xem cặp định nghĩa có khớp với cặp base hiện tại không (theo cả 2 chiều)
                        if (type1_def in entry_base and type2_def in current_base) or \
                           (type2_def in entry_base and type1_def in current_base):
                           # Thêm kiểm tra chi tiết hơn nếu là TREND -> TREND (UP -> DOWN hoặc ngược lại)
                           if "TREND" in type1_def and "TREND" in type2_def:
                               if entry_type != current_type: # Chỉ trigger nếu là UP->DOWN hoặc DOWN->UP
                                   regime_changed_significantly = True
                                   break
                           else: # Nếu là TREND <-> SIDEWAYS
                               regime_changed_significantly = True
                               break

                if regime_changed_significantly:
                    exit_reason = f'close_due_to_regime_change_{entry_type}_to_{current_type}'
                    signals['exit_signal'] = exit_reason
                    print(f"[{symbol}] EXIT Signal: Significant regime change detected ({entry_type} -> {current_type}).")
                    signals['trailing_tp_level'] = new_trailing_tp_level
                    signals['min_locked_tp_level'] = new_min_locked_tp_level
                    return signals # Return sớm


            # --- 4b. Determine Current RR Level Achieved ---
            for k_check in range(self.max_rr_level_for_check, 0, -1):
                rr_k_price = self._calculate_rr_price(entry_price, sl_price, k_check, position_type)
                if rr_k_price is None: continue
                price_passed_rrk = (position_type == 'long' and current_price >= rr_k_price) or \
                                   (position_type == 'short' and current_price <= rr_k_price)
                if price_passed_rrk: current_rr = k_check; break
            signals['current_rr'] = current_rr


            # --- 4c. Apply TP Logic (Only if >= RR1) ---
            if current_rr >= 1 and initial_risk_amount > 1e-9:
                potential_target_rr = current_rr
                is_unknown = (market_type == "UNKNOWN")
                use_conservative_tp_mode = is_conflicted or is_unknown

                # --- Determine Regime-Aware RR Prob Threshold ---
                if not use_conservative_tp_mode:
                    regime_key = f"{market_type}_{market_strength}"
                    # Tạo key chuẩn hơn cho fallback
                    base_market_type = "TREND" if "TREND" in market_type else "SIDEWAYS" if "SIDEWAYS" in market_type else "UNKNOWN"
                    fallback_key_normal = f"{base_market_type}_NORMAL" if base_market_type != "UNKNOWN" else "UNKNOWN"

                    # Ưu tiên key chi tiết -> key normal -> key UNKNOWN -> giá trị conservative mặc định
                    if regime_key in self.regime_aware_rr_prob_thresholds:
                         base_rr_prob_threshold = self.regime_aware_rr_prob_thresholds[regime_key]
                    elif fallback_key_normal in self.regime_aware_rr_prob_thresholds:
                         base_rr_prob_threshold = self.regime_aware_rr_prob_thresholds[fallback_key_normal]
                    elif "UNKNOWN" in self.regime_aware_rr_prob_thresholds:
                         base_rr_prob_threshold = self.regime_aware_rr_prob_thresholds["UNKNOWN"]
                    else: # Fallback cuối cùng nếu ngay cả UNKNOWN cũng không có
                        base_rr_prob_threshold = self.conservative_rr_prob_threshold
                        warnings.warn(f"Could not find base RR prob threshold for key '{regime_key}' or fallbacks. Using conservative value: {base_rr_prob_threshold}")
                else: # Nếu đang ở conservative mode
                    base_rr_prob_threshold = self.conservative_rr_prob_threshold

                # --- Adjust RR Prob Threshold by Confidence ---
                current_rr_prob_threshold = base_rr_prob_threshold # Bắt đầu với base đã xác định
                if not use_conservative_tp_mode and market_confidence < self.low_confidence_threshold:
                    original_threshold = current_rr_prob_threshold # Lưu lại ngưỡng trước khi điều chỉnh
                    current_rr_prob_threshold = min(0.99, current_rr_prob_threshold * self.confidence_penalty_factor)
                    print(f"[{symbol}] Low Confidence ({market_confidence:.2f}), adjusting RR Prob Thresh: {original_threshold:.2f} -> {current_rr_prob_threshold:.2f}")
                elif use_conservative_tp_mode:
                     # Không cần print lại ở đây nếu đã print ở bước conflict/unknown
                     pass

                # <<< Gán giá trị ngưỡng đã tính vào debug dict >>>
                signals['debug_tp_logic']['rr_prob_threshold_used'] = current_rr_prob_threshold


                # --- Determine Potential Target RR ---
                # <<< Đảm bảo truyền đúng use_conservative_tp_mode >>>
                for k in range(self.max_rr_level_for_check, current_rr, -1):
                    if not (0 <= k - 1 < len(rr_probs)): continue
                    prob_rr_k = rr_probs[k - 1].item()
                    rr_k_price_target = self._calculate_rr_price(entry_price, sl_price, k, position_type)
                    hybrid_passes = self._check_hybrid_passes(
                        multi_step_pred, rr_k_price_target, position_type, market_classification,
                        use_conservative_mode=use_conservative_tp_mode # << Truyền cờ
                    )
                    if prob_rr_k >= current_rr_prob_threshold and hybrid_passes:
                        potential_target_rr = k
                        break # Tìm thấy mức cao nhất
                signals['potential_target_rr'] = potential_target_rr


                # --- Decide Whether to Raise Trailing TP ---
                just_raised_tp = False
                target_tp_before_strength_check = current_trailing_tp_level
                if current_rr > current_trailing_tp_level and current_rr >= new_min_locked_tp_level:
                    target_tp_before_strength_check = current_rr # Mức TP mới tiềm năng
                    just_raised_tp = True


                # --- Decide Whether to Lock TP Floor ---
                # <<< Đảm bảo check đúng use_conservative_tp_mode >>>
                if just_raised_tp and not use_conservative_tp_mode:
                    k_raised_to = target_tp_before_strength_check
                    very_strong_signal = self._check_very_strong_signal(
                        rr_probs, multi_step_pred, k_raised_to, position_type, market_classification
                    )
                    if very_strong_signal:
                        floor_level = max(1, k_raised_to - 1)
                        if floor_level > new_min_locked_tp_level:
                            print(f"[{symbol}] Locking TP floor at RR{floor_level} (Reason: Very Strong Signal - {market_type}/{market_strength})")
                            new_min_locked_tp_level = floor_level
                elif just_raised_tp and use_conservative_tp_mode:
                    print(f"[{symbol}] TP Raised to RR{target_tp_before_strength_check}, but Floor Lock disabled due to Conflict/Unknown state.")


                # --- Update TP Level ---
                if just_raised_tp: new_trailing_tp_level = target_tp_before_strength_check
                new_trailing_tp_level = max(new_trailing_tp_level, new_min_locked_tp_level)


                # --- Dynamic R:R Exit Check ---
                # <<< Gán giá trị vào debug dict trước khi kiểm tra >>>
                if self.enable_dynamic_rr_exit and signals['exit_signal'] == 'hold' and new_min_locked_tp_level > 0:
                    tp_floor_price = self._calculate_rr_price(entry_price, sl_price, new_min_locked_tp_level, position_type)

                    # Only proceed if floor price is valid
                    if tp_floor_price is not None:
                        # Calculate risk remaining to the floor
                        risk_to_floor = abs(current_price - tp_floor_price)

                        # Calculate minimum risk required to perform the check (fraction of initial 1R)
                        # Ensure initial_risk_amount is valid before calculation
                        min_risk_value = 0.0
                        if initial_risk_amount > 1e-9: # Avoid issues if initial risk is zero
                            min_risk_value = initial_risk_amount * self.min_risk_to_floor_ratio

                        # Only proceed if risk to floor is meaningful relative to initial risk
                        if risk_to_floor >= min_risk_value:
                            # Calculate potential target price
                            potential_target_price = self._calculate_rr_price(entry_price, sl_price, potential_target_rr, position_type)

                            potential_reward_amount = 0 # Khởi tạo bằng 0
                            if potential_target_price is not None:
                                # Chỉ tính phần thưởng dương còn lại
                                if position_type == 'long' and potential_target_price > current_price:
                                    potential_reward_amount = potential_target_price - current_price
                                elif position_type == 'short' and potential_target_price < current_price:
                                    potential_reward_amount = current_price - potential_target_price
                                # Nếu potential_target_price không có lợi hơn current_price, reward_amount sẽ là 0

                            # Calculate dynamic R:R ratio, handle division by zero
                            current_dynamic_RR = potential_reward_amount / risk_to_floor if risk_to_floor > 1e-9 else float('inf')

                            # Gán vào debug dict (ngay cả khi không trigger exit)
                            signals['debug_tp_logic']['dynamic_RR'] = current_dynamic_RR
                            signals['debug_tp_logic']['risk_to_floor'] = risk_to_floor
                            signals['debug_tp_logic']['potential_reward'] = potential_reward_amount

                            # Check if dynamic R:R is below threshold
                            if current_dynamic_RR < self.exit_threshold_dynamic_RR:
                                exit_reason = f'close_dynamic_RR_below_threshold_{current_dynamic_RR:.2f}'
                                signals['exit_signal'] = exit_reason
                                print(f"[{symbol}] EXIT Signal: Dynamic R:R ({current_dynamic_RR:.2f}) below threshold ({self.exit_threshold_dynamic_RR:.2f}). Floor RR{new_min_locked_tp_level}, Potential RR{potential_target_rr}.")


            # --- Final Check: Hit Trailing TP ---
            if signals['exit_signal'] == 'hold' and new_trailing_tp_level > 0:
                tp_price = self._calculate_rr_price(entry_price, sl_price, new_trailing_tp_level, position_type)
                if tp_price is not None: price_hit_tp = (position_type == 'long' and current_price <= tp_price) or (position_type == 'short' and current_price >= tp_price)
                if price_hit_tp: signals['exit_signal'] = f'close_at_trailing_tp_{new_trailing_tp_level}'


            # Update final TP state
            signals['trailing_tp_level'] = new_trailing_tp_level
            signals['min_locked_tp_level'] = new_min_locked_tp_level

        # --- 5. Generate Entry Signal ---
        elif not open_position and signals['exit_signal'] == 'hold':
             # <<< Đảm bảo dùng deepcopy khi gán market_classification_at_entry >>>
            entry_signal_dict = self._generate_entry_signal(
                symbol, current_data, embedding, optimal_sl_raw, market_classification
            )
            signals['entry_signal'] = entry_signal_dict # Logic ngoài cần lấy và lưu deepcopy nếu cần

        # <<< Reset override để không ảnh hưởng lần gọi sau (quan trọng khi test nhiều lần) >>>
        sim_market_classification_override = None
        return signals

# ==============================================================================
# 6. EXAMPLE USAGE
# ==============================================================================
if __name__ == '__main__':

    print("\n--- Initializing Models and Agent with ENHANCED Regime-Aware Logic ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    DECISION_SEQ_LEN = 60; HYBRID_PRIMARY_TF = '15min'
    # <<< Sửa lỗi Pandas Warning >>>
    HYBRID_REQUIRED_TFS = ('15min', '1h', '4h') # Use 'min' if Pandas > 2.2
    HYBRID_SEQ_LEN_MAP = {'15min': 60, '1h': 15, '4h': 4 }

    dummy_config = { 
        'input_dim': 5, 'sequence_length': DECISION_SEQ_LEN, 'd_model': 64, 'nhead': 4,
        'num_encoder_layers': 3, 'dim_feedforward': 128, 'num_rr_levels': 10, 'dropout_rate': 0.1,
        'num_regimes': DEFAULT_NUM_REGIMES, 'hidden_dim': 128, 'cnn_kernel_size': 3, 'lstm_layers': 2,
        'use_attention': False, 'primary_tf': HYBRID_PRIMARY_TF, 'required_tfs': HYBRID_REQUIRED_TFS,
        'sac_long_threshold': 0.1, 'sac_short_threshold': -0.1, 'xgb_long_threshold': 0.65,
        'xgb_short_threshold': 0.35, 'signal_strong_threshold': 0.75, 'signal_medium_threshold': 0.65,
        'signal_weak_threshold': 0.55,
        'ADAPTIVE_WEIGHTS': { 'trending_strong': {'sac': 0.3, 'xgb': 0.7}, 'trending': {'sac': 0.4, 'xgb': 0.6},
                              'sideways_high_vol': {'sac': 0.7, 'xgb': 0.3}, 'sideways_low_vol': {'sac': 0.5, 'xgb': 0.5},
                              'default': {'sac': 0.5, 'xgb': 0.5} },
        'min_sl_atr_multiplier': 0.5, 'max_sl_atr_multiplier': 5.0,
        'adx_strong_thresh': 25, 'adx_very_strong_thresh': 35, 'sideway_adx_thresh': 20,
        'ma_slope_thresh_strong': 0.005, 'ma_slope_thresh_normal': 0.002, 'trend_volume_change_thresh': 1.15,
        'sideway_volume_cv_thresh': 0.35, 'sideway_bb_width_thresh': 0.04, 'trend_bb_width_expand_thresh': 0.06,
        'trend_very_strong_thresh': 4.0, 'trend_strong_thresh': 2.5, 'sideway_strong_thresh': 3.0,
        'sideway_normal_thresh': 1.5,
        'min_rr_level_for_check': 1, 'max_rr_level_for_check': 10,
        'regime_aware_rr_prob_thresholds': { 'TREND_VERY_STRONG': 0.55, 'TREND_STRONG': 0.58, 'TREND_NORMAL': 0.60, 'SIDEWAYS_STRONG': 0.65, 'SIDEWAYS_NORMAL': 0.65, 'SIDEWAYS_WEAK': 0.70, 'UNKNOWN': 0.75 },
        'conservative_rr_prob_threshold': 0.75, 'low_confidence_threshold': 0.60, 'confidence_penalty_factor': 1.15,
        'hybrid_trend_strong_pass': {'min_steps_confirm': 3, 'max_pullback_ratio': 0.3, 'min_target_cross': 1},
        'hybrid_trend_normal_pass': {'min_steps_confirm': 2, 'max_pullback_ratio': 0.5, 'min_target_cross': 1},
        'hybrid_sideways_pass': {'min_target_cross': 1},
        'lock_floor_trend_very_strong_rr_thresh': 0.80, 'lock_floor_trend_strong_rr_thresh': 0.85,
        'lock_floor_trend_normal_rr_thresh': 0.90, 'lock_floor_sideways_rr_thresh': 0.98,
        'significant_regime_change_def': [ ("TREND_UP", "TREND_DOWN") ],
        'enable_dynamic_rr_exit': True, 'exit_threshold_dynamic_RR': 1.0, 'min_risk_to_floor_ratio': 0.1,
        'adx_thresholds': {'15min': 22, '1h': 22, '4h': 25, 'default': 20}, 'bb_width_vol_threshold': 0.05,
        'xgboost_feature_columns': ["ATR", "ADX", "EMA_50", "EMA_200", "BB_upper", "BB_lower", "BB_middle", "RSI", "MACD", "MACD_signal", "OBV", "volatility"],
        'temporal_features': ['hour', 'day_of_week', 'is_weekend']
    }

    # --- Instantiate Models, Agent, Data ---
    example_hybrid_input_dict = {tf: torch.randn(2, HYBRID_SEQ_LEN_MAP[tf], dummy_config['input_dim']) for tf in dummy_config['required_tfs']}
    decision_model = CryptoDecisionSupportModel(dummy_config, device=DEVICE).to(DEVICE)
    hybrid_model = CryptoTradingModel(dummy_config, example_hybrid_input_dict, device=DEVICE).to(DEVICE)
    print(f"Decision Params: {sum(p.numel() for p in decision_model.parameters()):,}, Hybrid Params: {sum(p.numel() for p in hybrid_model.parameters()):,}")
    print(f"Hybrid Engineered Dim: {hybrid_model.engineered_feature_dim}")
    class DummySACPredictor:
        def predict(self, obs, deterministic=True): return (np.array([0.0]), None)
    agent = CombinedTradingAgent(decision_model=decision_model, hybrid_model=hybrid_model, bot_config=dummy_config, entry_optimizer_model=DummySACPredictor(), xgboost_models={}, xgboost_scalers={}, get_sac_state_func=lambda *a, **k: None, scale_sac_state_func=lambda s: s, get_dynamic_adx_func=lambda *a, **k: 22, temporal_features=dummy_config['temporal_features'])
    BATCH_SIZE = 1; dummy_symbol = "BTC/USDT"
    dummy_input_decision = torch.randn(BATCH_SIZE, DECISION_SEQ_LEN, dummy_config['input_dim']).to(DEVICE)
    dummy_input_hybrid_dict = {tf: torch.randn(BATCH_SIZE, HYBRID_SEQ_LEN_MAP[tf], dummy_config['input_dim']).to(DEVICE) for tf in dummy_config['required_tfs']}
    dummy_current_data = {}
    for tf_key in dummy_config['required_tfs']:
        # <<< Sửa lỗi Pandas Warning freq='T' -> freq=tf_key >>>
        # Giả sử tf_key là '15min', '1h', '4h'
        freq = tf_key
        if 'min' not in freq and 'h' not in freq: # Handle potential non-standard keys
            freq = '15min' # Default fallback
        seq_len = 100; df_index = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=seq_len, freq=freq))
        df = pd.DataFrame(index=df_index); start_price = 10000
        df['open'] = np.random.uniform(-1, 1, size=seq_len).cumsum() + start_price
        df['high'] = df['open'] + np.random.uniform(0, 2, size=seq_len)
        df['low'] = df['open'] - np.random.uniform(0, 2, size=seq_len)
        df['close'] = df['open'] + np.random.uniform(-0.5, 0.5, size=seq_len)
        df['volume'] = np.random.uniform(500, 1500, size=seq_len)
        if talib: # ... (tính indicator như cũ) ...
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14); df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['EMA_50'] = talib.EMA(df['close'], timeperiod=50); df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20); df['BB_upper'] = bb_upper; df['BB_middle'] = bb_middle; df['BB_lower'] = bb_lower
            df['RSI'] = talib.RSI(df['close'], timeperiod=14); macd, macdsignal, macdhist = talib.MACD(df['close']); df['MACD'] = macd; df['MACD_signal'] = macdsignal
            df['OBV'] = talib.OBV(df['close'], df['volume']); df['volatility'] = df['close'].rolling(10).std()
        else: # ... (tạo dummy indicator như cũ) ...
            for col in ['ATR', 'ADX', 'EMA_50', 'EMA_200', 'BB_upper', 'BB_middle', 'BB_lower', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'volatility']: df[col]=np.random.rand(seq_len)*10
        df['regime'] = 0; df['hour'] = df.index.hour; df['day_of_week'] = df.index.dayofweek; df['is_weekend'] = df.index.dayofweek.isin([5,6]).astype(int)
        # <<< Sửa lỗi Pandas Warning fillna >>>
        dummy_current_data[tf_key] = df.bfill().fillna(0)

    # --- Simulation Setup ---
    print("\n" + "="*10 + " Simulating Agent Decisions (Corrected) " + "="*10)
    original_decision_forward = decision_model.forward; original_hybrid_forward = hybrid_model.forward
    sim_market_classification_override = None # <<< Make sure this is global or passed correctly
    def set_sim_outputs(prob_rr_values=None, sl_raw_override=None, regime_logits_values=None, hybrid_predictions=None, market_classification_override=None):
        global sim_market_classification_override # <<< Declare global to modify
        sim_market_classification_override = market_classification_override
        # ... (patched_decision_forward giữ nguyên) ...
        def patched_decision_forward(self_model, x):
            out = original_decision_forward.__get__(self_model, type(self_model))(x)
            if prob_rr_values is not None and 'rr_probabilities' in out:
                num_levels = min(len(prob_rr_values), out['rr_probabilities'].shape[1]); new_rr = out['rr_probabilities'].clone()
                # <<< Sửa lỗi PyTorch Warning >>>
                prob_tensor = torch.tensor(prob_rr_values[:num_levels], device=new_rr.device, dtype=new_rr.dtype)
                new_rr[0, :num_levels] = prob_tensor; out['rr_probabilities'] = new_rr
            if sl_raw_override is not None and 'optimal_sl_raw' in out:
                out['optimal_sl_raw'] = torch.full_like(out['optimal_sl_raw'], sl_raw_override)
            return out
        # ... (patched_hybrid_forward giữ nguyên) ...
        def patched_hybrid_forward(self_model, x_dict):
            out = original_hybrid_forward.__get__(self_model, type(self_model))(x_dict)
            if regime_logits_values is not None and 'regime_logits' in out:
                 num_regimes = min(len(regime_logits_values), out['regime_logits'].shape[1]); new_logits = out['regime_logits'].clone()
                 logit_tensor = torch.tensor(regime_logits_values[:num_regimes], device=new_logits.device, dtype=new_logits.dtype)
                 new_logits[0, :num_regimes] = logit_tensor; out['regime_logits'] = new_logits
                 if 'regime_probabilities' in out: out['regime_probabilities'] = F.softmax(out['regime_logits'], dim=-1)
            if hybrid_predictions is not None and 'multi_step_ohlcv' in out:
                 preds = out['multi_step_ohlcv'];
                 # <<< Sửa lỗi PyTorch Warning & đảm bảo input là tensor >>>
                 if not isinstance(hybrid_predictions, torch.Tensor):
                     pred_tensor_np = np.array(hybrid_predictions) # Convert list/np.array to numpy first
                     pred_tensor = torch.from_numpy(pred_tensor_np).to(device=preds.device, dtype=preds.dtype)
                 else:
                      pred_tensor = hybrid_predictions.clone().detach().to(device=preds.device, dtype=preds.dtype)

                 if pred_tensor.dim() == 2: pred_tensor = pred_tensor.unsqueeze(0) # Add batch dim
                 b, s, f = min(preds.shape[0], pred_tensor.shape[0]), min(preds.shape[1], pred_tensor.shape[1]), min(preds.shape[2], pred_tensor.shape[2])
                 new_preds = preds.clone(); new_preds[:b, :s, :f] = pred_tensor[:b, :s, :f]; out['multi_step_ohlcv'] = new_preds
            return out
        decision_model.forward = patched_decision_forward.__get__(decision_model, CryptoDecisionSupportModel)
        hybrid_model.forward = patched_hybrid_forward.__get__(hybrid_model, CryptoTradingModel)

    # --- Helper Function to Safely Print Debug Floats ---
    def print_debug_float(value, format_spec=".2f"):
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value) # Return as string if None or other type
    # --- Test Case 1: Regime Conflict -> Conservative TP ---
    print("\n--- Test Case 1: Regime Conflict -> Conservative TP ---")
    sim_open_position_1 = {'type': 'long', 'entry_price': 100.0, 'stop_loss_price': 95.0, 'trailing_tp_level': 0, 'min_locked_tp_level': 0, 'market_classification_at_entry': {'type': 'TREND_UP', 'strength': 'NORMAL'}}
    sim_current_price_1 = 105.5
    # <<< Sửa lỗi: hybrid_predictions phải là list/numpy hoặc tensor >>>
    set_sim_outputs(
        prob_rr_values=[0.95, 0.9, 0.8], regime_logits_values=[2.5, -1.0, -1.0], # Hybrid: TREND_UP
        hybrid_predictions=[[105,106,104,105.5,0],[106,107,105,106.5,0],[107,111,106,110.5,0]], # <<< Dự đoán vượt RR2 (110)
        market_classification_override={'type': 'SIDEWAYS', 'strength': 'STRONG', 'confidence': 0.8, 'indicators': {'adx': 15}} # Indicator: SIDEWAYS
    )
    signals_1 = agent.get_trade_signals(dummy_symbol, dummy_input_decision, dummy_input_hybrid_dict, sim_current_price_1, dummy_current_data, sim_open_position_1)
    # <<< Sửa lỗi print TC1 >>>
    print(f"Hybrid Regime: {signals_1['predicted_regime']}, Indicator Class: {signals_1['market_classification']['type']}")
    print(f"RR Prob Thresh Used: {print_debug_float(signals_1['debug_tp_logic'].get('rr_prob_threshold_used'), '.2f')} (Expected Conservative: {agent.conservative_rr_prob_threshold:.2f})")
    print(f"Current RR: {signals_1['current_rr']}, Potential RR: {signals_1['potential_target_rr']}")
    print(f"Exit Signal: {signals_1['exit_signal']}")
    print(f"--> RESULT: TP={signals_1['trailing_tp_level']}, Floor={signals_1['min_locked_tp_level']}")
    # <<< Sửa lỗi Kỳ vọng TC1: Conservative hybrid check (sideways) chỉ cần target cross. Dự đoán đạt RR2 -> Potential RR=2 >>>
    print(f"--> EXPECTED: TP=1, Floor=0 (Conflict -> Conservative mode -> Floor lock disabled). Potential RR should be 2 (Prob OK, Conservative Hybrid Check OK for RR2).")


    # --- Test Case 2: Low Confidence -> Higher RR Prob Threshold ---
    print("\n--- Test Case 2: Low Confidence -> Higher RR Prob Threshold ---")
    sim_open_position_2 = {'type': 'long', 'entry_price': 100.0, 'stop_loss_price': 95.0, 'trailing_tp_level': 0, 'min_locked_tp_level': 0, 'market_classification_at_entry': {'type': 'TREND_UP', 'strength': 'NORMAL'}}
    sim_current_price_2 = 105.5
    set_sim_outputs(
        prob_rr_values=[0.95, 0.68, 0.5], regime_logits_values=[1.0, 0.8, -1.0], # <<< Confidence = 0.45 >>>
        hybrid_predictions=[[105,106,104,105.5,0],[106,108,105,107.5,0],[107,110,106,109.5,0]],
        market_classification_override={'type': 'TREND_UP', 'strength': 'NORMAL', 'confidence': 0.45, 'indicators': {'adx': 22}}
    )
    signals_2 = agent.get_trade_signals(dummy_symbol, dummy_input_decision, dummy_input_hybrid_dict, sim_current_price_2, dummy_current_data, sim_open_position_2)
    # <<< Sửa lỗi print TC2 >>>
    print(f"Market Class: {signals_2['market_classification']['type']} / Conf: {signals_2['market_classification']['confidence']:.2f}")
    expected_thresh = agent.regime_aware_rr_prob_thresholds.get('TREND_NORMAL', agent.conservative_rr_prob_threshold) * agent.confidence_penalty_factor # <<< Lấy đúng base trend normal >>>
    print(f"RR Prob Thresh Used: {print_debug_float(signals_2['debug_tp_logic'].get('rr_prob_threshold_used'), '.4f')} (Expected ~ {expected_thresh:.4f})") # In 4 chữ số
    print(f"Current RR: {signals_2['current_rr']}, Potential RR: {signals_2['potential_target_rr']}")
    print(f"Exit Signal: {signals_2['exit_signal']}")
    print(f"--> RESULT: TP={signals_2['trailing_tp_level']}, Floor={signals_2['min_locked_tp_level']}")
    # <<< Sửa lỗi Kỳ vọng TC2: RR2 prob (0.68) < expected thresh (0.69) -> Potential RR=1 >>>
    print(f"--> EXPECTED: Potential RR=1 (RR2 prob 0.68 < adjusted thresh ~0.69), TP=1, Floor=0")


    # --- Test Case 3: UNKNOWN Regime -> Conservative TP ---
    print("\n--- Test Case 3: UNKNOWN Regime -> Conservative TP ---")
    sim_open_position_3 = {'type': 'long', 'entry_price': 100.0, 'stop_loss_price': 95.0, 'trailing_tp_level': 0, 'min_locked_tp_level': 0, 'market_classification_at_entry': {'type': 'TREND_UP', 'strength': 'NORMAL'}}
    sim_current_price_3 = 105.5
    set_sim_outputs(
        prob_rr_values=[0.95, 0.9, 0.8], regime_logits_values=[0.1, 0.2, 0.1], # Hybrid UNKNOWN
        hybrid_predictions=[[105,106,104,105.5,0],[106,107,105,106.5,0],[107,111,106,110.5,0]], # <<< Dự đoán vượt RR2 >>>
        market_classification_override={'type': 'UNKNOWN', 'strength': 'UNKNOWN', 'confidence': 0.35, 'indicators': {'adx': 21}} # Force UNKNOWN
    )
    signals_3 = agent.get_trade_signals(dummy_symbol, dummy_input_decision, dummy_input_hybrid_dict, sim_current_price_3, dummy_current_data, sim_open_position_3)
    print(f"Market Class: {signals_3['market_classification']['type']}")
    # <<< Sửa lỗi print TC3: Dùng helper >>>
    print(f"RR Prob Thresh Used: {print_debug_float(signals_3['debug_tp_logic'].get('rr_prob_threshold_used'), '.2f')} (Expected Conservative: {agent.conservative_rr_prob_threshold:.2f})")
    print(f"Current RR: {signals_3['current_rr']}, Potential RR: {signals_3['potential_target_rr']}")
    print(f"Exit Signal: {signals_3['exit_signal']}") # <<< Kỳ vọng không có exit do regime change >>>
    print(f"--> RESULT: TP={signals_3['trailing_tp_level']}, Floor={signals_3['min_locked_tp_level']}")
    # <<< Sửa lỗi Kỳ vọng TC3: UNKNOWN dùng conservative -> Potential RR=2 (như TC1) >>>
    print(f"--> EXPECTED: Exit=hold. TP=1, Floor=0 (Conservative mode). Potential RR=2 (Prob OK, Conservative Hybrid Check OK for RR2).")


    # --- Test Case 4: Regime Change Exit ---
    print("\n--- Test Case 4: Regime Change Exit ---")
    sim_open_position_4 = {'type': 'long', 'entry_price': 100.0, 'stop_loss_price': 95.0, 'trailing_tp_level': 2, 'min_locked_tp_level': 1, 'market_classification_at_entry': {'type': 'TREND_UP', 'strength': 'STRONG', 'confidence': 0.9, 'indicators': {'adx': 30}} }
    sim_current_price_4 = 112.0
    set_sim_outputs(
        prob_rr_values=[0.9, 0.8, 0.7, 0.6], regime_logits_values=[-1.0, 2.5, -1.0], # Hybrid SIDEWAYS
        hybrid_predictions=None, # Không cần cho test này
        market_classification_override={'type': 'SIDEWAYS', 'strength': 'NORMAL', 'confidence': 0.8, 'indicators': {'adx': 18}} # Indicator SIDEWAYS
    )
    signals_4 = agent.get_trade_signals(dummy_symbol, dummy_input_decision, dummy_input_hybrid_dict, sim_current_price_4, dummy_current_data, sim_open_position_4)
    print(f"Entry Regime: {sim_open_position_4['market_classification_at_entry']['type']}, Current Regime: {signals_4['market_classification']['type']}")
    print(f"Current RR: {signals_4['current_rr']}, Potential RR: {signals_4['potential_target_rr']}") # TP logic không chạy do exit sớm
    print(f"Exit Signal: {signals_4['exit_signal']}")
    print(f"--> RESULT: TP={signals_4['trailing_tp_level']}, Floor={signals_4['min_locked_tp_level']}")
    print(f"--> EXPECTED: Exit Signal contains TP=2, Floor=1")


    # --- Test Case 5: Dynamic R:R Exit ---
    print("\n--- Test Case 5: Dynamic R:R Exit ---")
    sim_open_position_5 = {'type': 'long', 'entry_price': 100.0, 'stop_loss_price': 90.0, 'trailing_tp_level': 4, 'min_locked_tp_level': 3, 'market_classification_at_entry': {'type': 'TREND_UP', 'strength': 'VERY_STRONG'}}
    sim_current_price_5 = 132.0
    set_sim_outputs(
        prob_rr_values=[0.9]*3 + [0.6, 0.4, 0.2], regime_logits_values=[2.0, 0.5, -1.0], # TREND_UP NORMAL
        hybrid_predictions=[[131,133,130,132,0],[132,135,131,134,0],[134,138,133,137,0], [137, 141, 136, 140, 0]],
        market_classification_override={'type': 'TREND_UP', 'strength': 'NORMAL', 'confidence': 0.7, 'indicators': {'adx': 24}}
    )
    signals_5 = agent.get_trade_signals(dummy_symbol, dummy_input_decision, dummy_input_hybrid_dict, sim_current_price_5, dummy_current_data, sim_open_position_5)
    print(f"Market Class: {signals_5['market_classification']['type']} / {signals_5['market_classification']['strength']}")
    print(f"Current RR: {signals_5['current_rr']}, Potential RR: {signals_5['potential_target_rr']} (Expected 4)")
    print(f"Debug Info: {signals_5['debug_tp_logic']}")
    print(f"Exit Signal: {signals_5['exit_signal']}")
    print(f"--> RESULT: TP={signals_5['trailing_tp_level']}, Floor={signals_5['min_locked_tp_level']}")
    print(f"--> EXPECTED: Exit=hold, TP=4, Floor=3 (Dynamic R:R ~4.0 > Threshold 1.0)")


    # --- Test Case 5b: Dynamic R:R Exit - Triggered ---
    print("\n--- Test Case 5b: Dynamic R:R Exit - Triggered ---")
    sim_current_price_5b = 138.0
    tp_floor_price_5 = 130.0 # Giữ nguyên từ case 5
    set_sim_outputs(
        prob_rr_values=[0.9]*3 + [0.55, 0.3, 0.1], # RR4 prob = 0.55 < TREND_NORMAL base 0.60 -> Potential RR=3
        regime_logits_values=[2.0, 0.5, -1.0], # TREND_UP NORMAL
        hybrid_predictions=[[137,139,136,138,0],[138,140,137,139,0],[139,141,138,140,0],[137, 141, 136, 140, 0]],
        market_classification_override={'type': 'TREND_UP', 'strength': 'NORMAL', 'confidence': 0.7, 'indicators': {'adx': 24}}
    )
    signals_5b = agent.get_trade_signals(dummy_symbol, dummy_input_decision, dummy_input_hybrid_dict, sim_current_price_5b, dummy_current_data, sim_open_position_5) # Dùng lại sim_open_position_5
    print(f"Market Class: {signals_5b['market_classification']['type']} / {signals_5b['market_classification']['strength']}")
    print(f"Current RR: {signals_5b['current_rr']} (Expected 3), Potential RR: {signals_5b['potential_target_rr']} (Expected 3)")
    print(f"Debug Info: {signals_5b['debug_tp_logic']}")
    # Expected Calculation: Potential RR=3. Reward = abs(RR3_price(130) - current(138)) = 0 (vì đã qua). Risk = abs(current(138) - floor_price(130)) = 8. Ratio = 0 / 8 = 0.
    # Since 0 < exit_threshold (1.0). EXIT expected.
    print(f"Exit Signal: {signals_5b['exit_signal']}")
    print(f"--> RESULT: TP={signals_5b['trailing_tp_level']}, Floor={signals_5b['min_locked_tp_level']}")
    print(f"--> EXPECTED: Exit Signal contains 'close_dynamic_RR_below_threshold_0.00', TP=3, Floor=3") # TP được nâng lên current RR (3)


    # Restore original forward methods if necessary
    decision_model.forward = original_decision_forward
    hybrid_model.forward = original_hybrid_forward
    # <<< Reset override sau khi test xong >>>
    sim_market_classification_override = None

    print("\n--- Simulation Complete ---")
