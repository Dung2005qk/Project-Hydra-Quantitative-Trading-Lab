import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import time
import glob
import gc
from typing import List, Optional, Tuple

# ==============================================================================
# Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        # Log riêng cho lần chạy này
        logging.FileHandler("prepare_lob_final_combined.log", encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)

# --- Paths ---
BASE_DATA_DIR = Path("D:/")
OUTPUT_DIR = BASE_DATA_DIR / "processed_lob_data"
CSV_250MS_PATH = BASE_DATA_DIR / "1-09-1-20.csv"
XBTUSD_DIR = BASE_DATA_DIR

OUTPUT_DIR.mkdir(exist_ok=True)

# Output file names
FINAL_OUTPUT_PATH_X = OUTPUT_DIR / "combined_lob_training_X_250ms_XBTUSD.npy"
FINAL_OUTPUT_PATH_Y = OUTPUT_DIR / "combined_lob_training_y_250ms_XBTUSD.npy"

# --- LOB Parameters ---
LEVELS_TO_EXTRACT = 5
INPUT_DIM_EXPECTED = LEVELS_TO_EXTRACT * 4

# --- Target Calculation Parameters ---
PREDICTION_HORIZON_MS = 5000
STATIONARY_THRESHOLD_PCT = 0.0001

# --- Processing Parameters ---
CSV_CHUNKSIZE = 1_000_000

# --- Missing XBTUSD Dates ---
MISSING_XBTUSD_DATES = ["2020-01-14", "2020-02-09"]

# --- Column Configurations ---

# 250ms CSV (No Header) - *** Cấu hình đã sửa theo hình ảnh cuối ***
TIMESTAMP_COL_INDEX_250MS = 1 # Index 1 là timestamp micro giây
LEVELS_IN_250MS_FILE = 10
# Indices trong file gốc
BID_PRICE_INDICES_FILE_250MS = [3 + i*2 for i in range(LEVELS_IN_250MS_FILE)]
BID_VOLUME_INDICES_FILE_250MS = [4 + i*2 for i in range(LEVELS_IN_250MS_FILE)]
ASK_PRICE_INDICES_FILE_250MS = [23 + i*2 for i in range(LEVELS_IN_250MS_FILE)]
ASK_VOLUME_INDICES_FILE_250MS = [24 + i*2 for i in range(LEVELS_IN_250MS_FILE)]
# Indices cần thiết (5 levels đầu)
BID_PRICE_INDICES_NEEDED_250MS = BID_PRICE_INDICES_FILE_250MS[:LEVELS_TO_EXTRACT]
BID_VOLUME_INDICES_NEEDED_250MS = BID_VOLUME_INDICES_FILE_250MS[:LEVELS_TO_EXTRACT]
ASK_PRICE_INDICES_NEEDED_250MS = ASK_PRICE_INDICES_FILE_250MS[:LEVELS_TO_EXTRACT]
ASK_VOLUME_INDICES_NEEDED_250MS = ASK_VOLUME_INDICES_FILE_250MS[:LEVELS_TO_EXTRACT]
# Tập hợp index cần đọc
ALL_REQUIRED_INDICES_250MS = sorted(list(set(
    [TIMESTAMP_COL_INDEX_250MS] + BID_PRICE_INDICES_NEEDED_250MS + BID_VOLUME_INDICES_NEEDED_250MS +
    ASK_PRICE_INDICES_NEEDED_250MS + ASK_VOLUME_INDICES_NEEDED_250MS
)))
# Mapping index -> tên cột
INDEX_TO_NAME_MAP_250MS = {TIMESTAMP_COL_INDEX_250MS: 'timestamp_orig'}
for i in range(LEVELS_TO_EXTRACT):
    INDEX_TO_NAME_MAP_250MS[BID_PRICE_INDICES_NEEDED_250MS[i]] = f'bid_p{i+1}'
    INDEX_TO_NAME_MAP_250MS[BID_VOLUME_INDICES_NEEDED_250MS[i]] = f'bid_v{i+1}'
    INDEX_TO_NAME_MAP_250MS[ASK_PRICE_INDICES_NEEDED_250MS[i]] = f'ask_p{i+1}'
    INDEX_TO_NAME_MAP_250MS[ASK_VOLUME_INDICES_NEEDED_250MS[i]] = f'ask_v{i+1}'

# XBTUSD CSV (Has header)
TIMESTAMP_COL_INDEX_XBTUSD = 0
TIMESTAMP_COL_NAME_INTERNAL = 'datetime_str' # Tên tạm
MIDPOINT_COL_XBTUSD = 'midpoint'
# Sử dụng tên cột chuẩn (đã được xác nhận từ log header)
BID_DIST_COLS_XBTUSD = [f'bids_distance_{i}' for i in range(LEVELS_TO_EXTRACT)]
BID_NOTIONAL_COLS_XBTUSD = [f'bids_limit_notional_{i}' for i in range(LEVELS_TO_EXTRACT)] # Dùng limit_notional
ASK_DIST_COLS_XBTUSD = [f'asks_distance_{i}' for i in range(LEVELS_TO_EXTRACT)]
ASK_NOTIONAL_COLS_XBTUSD = [f'asks_limit_notional_{i}' for i in range(LEVELS_TO_EXTRACT)] # Dùng limit_notional
# Các cột có tên cần thiết
REQUIRED_NAMED_COLS_XBTUSD = [MIDPOINT_COL_XBTUSD] + BID_DIST_COLS_XBTUSD + BID_NOTIONAL_COLS_XBTUSD + ASK_DIST_COLS_XBTUSD + ASK_NOTIONAL_COLS_XBTUSD

# Feature order expected by the model
FEATURE_COLS_ORDER = [f'bid_p{i+1}' for i in range(LEVELS_TO_EXTRACT)] + \
                     [f'bid_v{i+1}' for i in range(LEVELS_TO_EXTRACT)] + \
                     [f'ask_p{i+1}' for i in range(LEVELS_TO_EXTRACT)] + \
                     [f'ask_v{i+1}' for i in range(LEVELS_TO_EXTRACT)]

# ==============================================================================
# Helper Functions
# ==============================================================================
def parse_microseconds_timestamp(timestamp_series: pd.Series) -> pd.Series:
    try:
        numeric_ts = pd.to_numeric(timestamp_series, errors='coerce')
        ms_ts = (numeric_ts // 1000).astype('Int64')
        return ms_ts
    except Exception as e: logging.error(f"Error parsing microseconds: {e}"); return pd.Series(pd.NA, index=timestamp_series.index, dtype='Int64')

def parse_datetime_timestamp(timestamp_series: pd.Series) -> pd.Series:
    try:
        dt_series = pd.to_datetime(timestamp_series, errors='coerce')
        if dt_series.dt.tz is None: dt_series = dt_series.dt.tz_localize('UTC')
        else: dt_series = dt_series.dt.tz_convert('UTC')
        ms_ts = dt_series.astype(np.int64) // 1_000_000
        return ms_ts.astype('Int64')
    except Exception as e: logging.error(f"Error parsing datetime: {e}"); return pd.Series(pd.NA, index=timestamp_series.index, dtype='Int64')

def calculate_quantity_from_notional(notional: float, price: float) -> Optional[float]:
    if pd.notna(notional) and pd.notna(price) and np.isfinite(notional) and np.isfinite(price) and \
       price > 1e-12 and notional >= 0:
        try:
            qty = notional / price
            if np.isfinite(qty) and qty >= 0: return qty
            else: return None
        except: return None
    return None

# ==============================================================================
# Chunk Processing Functions
# ==============================================================================

def process_chunk_250ms(chunk: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Processes a chunk of data from the 250ms CSV file."""
    start_rows = len(chunk)
    logging.debug(f"Processing 250ms chunk with {start_rows} rows.")
    try:
        rename_dict = {idx: name for idx, name in INDEX_TO_NAME_MAP_250MS.items() if idx in chunk.columns}
        df_processed = chunk.rename(columns=rename_dict)
    except Exception as e: logging.error(f"Error renaming 250ms chunk: {e}"); return None

    df_processed['timestamp_ms'] = parse_microseconds_timestamp(df_processed['timestamp_orig'])
    df_processed = df_processed.dropna(subset=['timestamp_ms'])
    if df_processed.empty: return None # Bỏ qua chunk nếu không có timestamp hợp lệ
    df_processed['timestamp_ms'] = df_processed['timestamp_ms'].astype(np.int64)

    lob_cols = FEATURE_COLS_ORDER
    missing_lob_cols = [c for c in lob_cols if c not in df_processed.columns]
    if missing_lob_cols: logging.error(f"Missing LOB cols after rename (250ms): {missing_lob_cols}"); return None

    for col in lob_cols: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed = df_processed.dropna(subset=lob_cols)
    if df_processed.empty: return None # Bỏ qua nếu có NaN trong LOB
    for col in lob_cols: df_processed[col] = df_processed[col].astype(np.float32)

    try:
        valid_mask = (df_processed['ask_p1'] > df_processed['bid_p1']) & \
                     (df_processed['bid_p1'] > 1e-9) & \
                     (df_processed['ask_p1'] > 1e-9) & \
                     (df_processed[[f'bid_v{i+1}' for i in range(LEVELS_TO_EXTRACT)]] >= 0).all(axis=1) & \
                     (df_processed[[f'ask_v{i+1}' for i in range(LEVELS_TO_EXTRACT)]] >= 0).all(axis=1)
    except KeyError as e: logging.error(f"KeyError during validation mask (250ms): {e}"); return None

    df_processed = df_processed[valid_mask]
    if df_processed.empty: return None # Bỏ qua nếu không qua kiểm tra logic

    try:
        timestamps_np = df_processed['timestamp_ms'].values
        midpoints_np = ((df_processed['bid_p1'] + df_processed['ask_p1']) / 2.0).values.astype(np.float32)
        features_np = df_processed[FEATURE_COLS_ORDER].values.astype(np.float32)
        if features_np.shape[1] != INPUT_DIM_EXPECTED: logging.error("Feature shape mismatch (250ms)"); return None
    except Exception as e: logging.error(f"Error extracting final arrays (250ms): {e}"); return None

    logging.debug(f"Successfully processed 250ms chunk, generated {len(timestamps_np)} valid entries.")
    return timestamps_np, features_np, midpoints_np

def process_chunk_xbtusd_iterative(chunk: pd.DataFrame, required_named_cols: List[str], timestamp_col_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Processes a chunk of XBTUSD data using row-by-row iteration."""
    start_rows = len(chunk)
    logging.debug(f"Processing XBTUSD chunk iteratively with {start_rows} rows.")

    # 1. Parse Timestamp
    chunk['timestamp_ms'] = parse_datetime_timestamp(chunk[timestamp_col_name])
    chunk = chunk.dropna(subset=['timestamp_ms'])
    if chunk.empty: return None
    chunk['timestamp_ms'] = chunk['timestamp_ms'].astype(np.int64)

    # 2. Convert numeric columns, handle NaN
    for col in required_named_cols: chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk = chunk.dropna(subset=required_named_cols)
    if chunk.empty: return None
    for col in required_named_cols: chunk[col] = chunk[col].astype(np.float32)

    # 3. Iterate and process rows
    processed_data_list = []
    for _, row in chunk.iterrows():
        try:
            midpoint = row[MIDPOINT_COL_XBTUSD]
            if midpoint <= 1e-9: continue
            bid_prices, bid_qtys, ask_prices, ask_qtys = [], [], [], []
            valid_entry = True
            for i in range(LEVELS_TO_EXTRACT): # Bids
                price = midpoint + row[BID_DIST_COLS_XBTUSD[i]]
                qty = calculate_quantity_from_notional(row[BID_NOTIONAL_COLS_XBTUSD[i]], price)
                if price <= 1e-12 or qty is None or qty < 0: valid_entry = False; break
                bid_prices.append(price); bid_qtys.append(qty)
            if not valid_entry: continue
            for i in range(LEVELS_TO_EXTRACT): # Asks
                price = midpoint + row[ASK_DIST_COLS_XBTUSD[i]]
                qty = calculate_quantity_from_notional(row[ASK_NOTIONAL_COLS_XBTUSD[i]], price)
                if price <= 1e-12 or qty is None or qty < 0: valid_entry = False; break
                ask_prices.append(price); ask_qtys.append(qty)
            if not valid_entry: continue
            if ask_prices[0] <= bid_prices[0]: continue
            # Combine features
            features = np.concatenate([bid_prices, bid_qtys, ask_prices, ask_qtys]).astype(np.float32)
            if features.shape[0] != INPUT_DIM_EXPECTED: continue
            processed_data_list.append({'timestamp_ms': row['timestamp_ms'], 'features': features, 'midpoint': midpoint})
        except Exception as row_err: logging.debug(f"Skipping row error (XBTUSD iter): {row_err}"); continue

    if not processed_data_list: return None

    # 4. Convert list to NumPy arrays
    try:
        timestamps_np = np.array([d['timestamp_ms'] for d in processed_data_list], dtype=np.int64)
        features_np = np.stack([d['features'] for d in processed_data_list]).astype(np.float32)
        midpoints_np = np.array([d['midpoint'] for d in processed_data_list], dtype=np.float32)
    except Exception as final_conv_err: logging.error(f"Error converting XBTUSD list to arrays: {final_conv_err}"); return None

    logging.debug(f"Finished XBTUSD iterative chunk, {len(timestamps_np)} valid rows.")
    return timestamps_np, features_np, midpoints_np

# ==============================================================================
# Target Calculation Function (Giữ nguyên)
# ==============================================================================
import pandas as pd
import numpy as np
import logging
import time
import gc

# Giả định các hằng số đã được định nghĩa ở đâu đó
PREDICTION_HORIZON_MS = 5000
STATIONARY_THRESHOLD_PCT = 0.0001

def calculate_target_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculating targets using vectorized approach...")
    target_start_time = time.time()
    # Đảm bảo df không bị thay đổi ngoài ý muốn
    df = df.copy()
    # Sắp xếp và reset index để đảm bảo tính liên tục cho np.arange
    df = df.sort_values('timestamp_ms').reset_index(drop=True)
    timestamps_ms = df['timestamp_ms'].values
    mid_prices = df['midpoint'].values
    n_samples = len(df)
     # *** THÊM BƯỚC LỌC MIDPOINT ***
    min_reasonable_price = 1000 # Đặt ngưỡng giá tối thiểu hợp lý
    max_reasonable_price = 100000 # Đặt ngưỡng giá tối đa hợp lý
    reasonable_midpoint_mask = (mid_prices > min_reasonable_price) & (mid_prices < max_reasonable_price) & np.isfinite(mid_prices)
    num_filtered_unreasonable = (~reasonable_midpoint_mask).sum()
    if num_filtered_unreasonable > 0:
        logging.warning(f"Filtering out {num_filtered_unreasonable} rows due to unreasonable midpoints (outside {min_reasonable_price}-{max_reasonable_price}).")
        # Lọc DataFrame và các mảng tương ứng
        df = df[reasonable_midpoint_mask].reset_index(drop=True) # Lọc df trước
        if df.empty:
            logging.error("No data remaining after filtering unreasonable midpoints.")
            return None
        # Cập nhật lại các biến sau khi lọc df
        timestamps_ms = df['timestamp_ms'].values
        mid_prices = df['midpoint'].values
        n_samples = len(df)
        logging.info(f"Shape after filtering unreasonable midpoints: {df.shape}")
    # *** KẾT THÚC LỌC MIDPOINT ***

    logging.info(f"Input shape for target calc (after midpoint filter): {df.shape}")
    targets = np.full(n_samples, -1, dtype=np.int8)

    future_timestamps = timestamps_ms + PREDICTION_HORIZON_MS
    # Tìm index của snapshot tương lai gần nhất (hoặc ngay sau thời điểm 5s tới)
    future_indices = np.searchsorted(timestamps_ms, future_timestamps, side='left')

    # Mask cho các hàng có index tương lai hợp lệ (trong phạm vi và > index hiện tại)
    valid_future_mask = (future_indices < n_samples) & (future_indices > np.arange(n_samples))
    num_potential_future = valid_future_mask.sum()
    logging.info(f"Number of rows with potential future index: {num_potential_future}")

    if not valid_future_mask.any():
        logging.error("No rows found with a valid future index within the dataset range.")
        return None # Trả về None nếu không có điểm nào có tương lai

    # Lấy ra các index hiện tại và tương lai hợp lệ
    current_indices = np.arange(n_samples)[valid_future_mask]
    valid_future_indices = future_indices[valid_future_mask]

    # Lấy giá midpoint tương ứng
    current_mids = mid_prices[current_indices]
    future_mids = mid_prices[valid_future_indices]

    # Mask cho các cặp midpoint hợp lệ để tính toán (không NaN/Inf, > 0)
    valid_calc_mask = np.isfinite(current_mids) & np.isfinite(future_mids) & (current_mids > 1e-9)
    num_valid_pairs = valid_calc_mask.sum()
    logging.info(f"Number of rows with valid current/future midpoints for calculation: {num_valid_pairs}")

    if not valid_calc_mask.any():
        logging.error("No valid midpoint pairs found for target calculation (check for NaNs, Infs, non-positive values).")
        # In ra ví dụ nếu có lỗi (phần debug giữ nguyên)
        invalid_mid_indices = current_indices[~valid_calc_mask]
        if len(invalid_mid_indices) > 0:
             logging.debug(f"Example invalid current midpoints: {mid_prices[invalid_mid_indices[:5]]}")
             corresponding_future_indices = future_indices[invalid_mid_indices[:5]]
             valid_future_lookups = corresponding_future_indices[corresponding_future_indices < n_samples]
             if len(valid_future_lookups) > 0:
                  logging.debug(f"Example corresponding future midpoints: {mid_prices[valid_future_lookups]}")
        return None

    # Chỉ lấy ra các index và giá trị thực sự dùng để tính toán
    final_valid_indices = current_indices[valid_calc_mask]
    valid_current_mids = current_mids[valid_calc_mask].astype(np.float64)
    valid_future_mids = future_mids[valid_calc_mask].astype(np.float64)

    # Khởi tạo mảng pct change với NaN
    price_change_pct = np.full(n_samples, np.nan, dtype=np.float64)

    # Tính pct change an toàn, kết quả lưu vào vị trí tương ứng trong price_change_pct
    diff = valid_future_mids - valid_current_mids
    np.divide(diff, valid_current_mids,
              out=price_change_pct[final_valid_indices], # Gán kết quả vào đúng vị trí
              where=np.abs(valid_current_mids) > 1e-12)

    # --- Xử lý NaN/Inf SAU phép chia và xác định index/giá trị để gán target ---
    # Lấy tất cả pct đã tính được (tại các vị trí final_valid_indices)
    calculated_pcts_at_valid_indices = price_change_pct[final_valid_indices]

    # Mask cho những giá trị pct hợp lệ (không NaN/Inf) trong số những cái đã tính
    finite_pct_mask = np.isfinite(calculated_pcts_at_valid_indices)
    num_finite_pct = finite_pct_mask.sum()
    logging.info(f"Number of finite price change percentages calculated: {num_finite_pct}")

    if num_finite_pct == 0:
        logging.error("Could not calculate any finite price change percentages.")
        # In ra một vài ví dụ về tử số (diff) và mẫu số (current_mids) gây lỗi
        logging.debug(f"Example diff values leading to non-finite pct: {diff[:10]}")
        logging.debug(f"Example current_mids values leading to non-finite pct: {valid_current_mids[:10]}")
        # In ra một vài kết quả pct không hữu hạn
        logging.debug(f"Example non-finite pct values: {calculated_pcts_at_valid_indices[:10]}")
        return None

    # Lấy ra các index và giá trị pct thực sự dùng để gán target 0, 1, 2
    indices_for_comparison = final_valid_indices[finite_pct_mask]
    pct_for_comparison = calculated_pcts_at_valid_indices[finite_pct_mask]

    # Gán target 0, 1, 2 dựa trên ngưỡng
    mask_up = pct_for_comparison > STATIONARY_THRESHOLD_PCT
    mask_down = pct_for_comparison < -STATIONARY_THRESHOLD_PCT
    mask_stationary = np.abs(pct_for_comparison) <= STATIONARY_THRESHOLD_PCT

    logging.info(f"Target counts (for comparable values): UP={mask_up.sum()}, DOWN={mask_down.sum()}, STATIONARY={mask_stationary.sum()}")

    targets[indices_for_comparison[mask_up]] = 1
    targets[indices_for_comparison[mask_down]] = 0
    targets[indices_for_comparison[mask_stationary]] = 2

    # Gán cột target vào DataFrame
    df['target'] = targets
    # Lọc bỏ tất cả các hàng có target = -1 (bao gồm cả những hàng không tính được pct ban đầu)
    df_filtered = df[df['target'] != -1].copy()
    gc.collect()

    if df_filtered.empty:
        logging.error("DataFrame is empty after filtering out rows with target = -1.")
        return None

    logging.info(f"Target calculation finished in {time.time() - target_start_time:.2f} seconds.")
    logging.info(f"Shape after calculating and filtering targets: {df_filtered.shape}")
    target_counts = df_filtered['target'].value_counts(normalize=True).sort_index()
    logging.info(f"Target distribution: {target_counts.to_dict()}")
    return df_filtered

# ==============================================================================
# Main Execution Logic
# ==============================================================================
if __name__ == "__main__":
    logging.info("=== Starting Combined 250ms + XBTUSD LOB Data Processing ===")
    overall_start_time = time.time()

    all_timestamps_list = []
    all_features_list = []
    all_midpoints_list = []
    total_rows_combined = 0

    # --- 1. Process 250ms File ---
    if CSV_250MS_PATH.exists():
        logging.info(f"Processing 250ms file: {CSV_250MS_PATH.name}")
        try:
            chunk_iterator = pd.read_csv(
                CSV_250MS_PATH, chunksize=CSV_CHUNKSIZE, header=None,
                usecols=ALL_REQUIRED_INDICES_250MS, # Chỉ đọc cột cần thiết
                low_memory=False
            )
            file_rows_processed_250ms = 0
            for chunk_num, chunk in enumerate(chunk_iterator):
                logging.info(f"Processing 250ms chunk {chunk_num+1}...")
                result = process_chunk_250ms(chunk) # Sử dụng hàm đã sửa index
                if result:
                    ts_np, f_np, m_np = result
                    all_timestamps_list.append(ts_np)
                    all_features_list.append(f_np)
                    all_midpoints_list.append(m_np)
                    rows_in_chunk = len(ts_np)
                    total_rows_combined += rows_in_chunk
                    file_rows_processed_250ms += rows_in_chunk
                del chunk; gc.collect()
            logging.info(f"Finished processing 250ms file. Total valid rows from this file: {file_rows_processed_250ms}")
        except Exception as e:
            logging.error(f"Error processing 250ms file {CSV_250MS_PATH}: {e}", exc_info=True)
    else:
        logging.warning(f"250ms file not found: {CSV_250MS_PATH}. Skipping.")

    # --- 2. Process XBTUSD Files (Using Iterative Logic) ---
    logging.info(f"Scanning for XBTUSD files in: {XBTUSD_DIR}")
    xbtusd_files = sorted(list(XBTUSD_DIR.glob("XBTUSD_????-??-??.csv")))
    logging.info(f"Found {len(xbtusd_files)} potential XBTUSD files.")

    # Đọc header và xác định cột
    actual_xbtusd_cols = None
    cols_to_read_indices = None
    col_names_standard_xbtusd = None # Tên cột chuẩn sẽ dùng
    if xbtusd_files:
        try:
            header_df = pd.read_csv(xbtusd_files[0], nrows=0)
            actual_xbtusd_cols = header_df.columns.tolist()
            logging.info(f"Detected XBTUSD header: {actual_xbtusd_cols}")
            # Kiểm tra các cột cần thiết
            if not all(col in actual_xbtusd_cols for col in REQUIRED_NAMED_COLS_XBTUSD):
                 missing = [c for c in REQUIRED_NAMED_COLS_XBTUSD if c not in actual_xbtusd_cols]
                 logging.error(f"XBTUSD files missing required columns: {missing}. Skipping XBTUSD.")
                 xbtusd_files = []
            else:
                # Xác định index cần đọc
                cols_to_read_indices = [TIMESTAMP_COL_INDEX_XBTUSD] + [actual_xbtusd_cols.index(c) for c in REQUIRED_NAMED_COLS_XBTUSD]
                # Xác định tên cột chuẩn sẽ dùng sau khi đọc
                col_names_standard_xbtusd = [TIMESTAMP_COL_NAME_INTERNAL] + REQUIRED_NAMED_COLS_XBTUSD
                logging.info(f"Will read XBTUSD indices: {cols_to_read_indices}")
                logging.info(f"Will assign names: {col_names_standard_xbtusd}")
        except Exception as e:
            logging.error(f"Could not read header/validate columns from {xbtusd_files[0].name}: {e}. Skipping XBTUSD.")
            xbtusd_files = []

    initial_row_count_xbtusd = total_rows_combined
    for filepath in xbtusd_files:
        file_date_str = filepath.stem.replace("XBTUSD_", "")
        if file_date_str in MISSING_XBTUSD_DATES:
            logging.info(f"Skipping specified missing file: {filepath.name}")
            continue

        logging.info(f"Processing XBTUSD file: {filepath.name}")
        file_rows_processed_xbt = 0
        try:
            chunk_iterator = pd.read_csv(
                filepath, chunksize=CSV_CHUNKSIZE, header=0,
                usecols=cols_to_read_indices, # Đọc theo index
                low_memory=False
            )
            for chunk_num, chunk in enumerate(chunk_iterator):
                 # Gán tên cột chuẩn
                 chunk.columns = col_names_standard_xbtusd
                 # Gọi hàm xử lý iterative
                 result = process_chunk_xbtusd_iterative(chunk, REQUIRED_NAMED_COLS_XBTUSD, TIMESTAMP_COL_NAME_INTERNAL)
                 if result:
                    ts_np, f_np, m_np = result
                    all_timestamps_list.append(ts_np)
                    all_features_list.append(f_np)
                    all_midpoints_list.append(m_np)
                    rows_in_chunk = len(ts_np)
                    total_rows_combined += rows_in_chunk
                    file_rows_processed_xbt += rows_in_chunk
                 del chunk; gc.collect()
            logging.info(f"Finished processing {filepath.name}. Added {file_rows_processed_xbt} rows. Total valid rows now: {total_rows_combined}")
        except Exception as e:
            logging.error(f"Error processing XBTUSD file {filepath}: {e}", exc_info=True)

    processed_xbtusd_count = total_rows_combined - initial_row_count_xbtusd
    logging.info(f"Processed {processed_xbtusd_count} total rows from XBTUSD files.")

    # --- 3. Combine, Sort, Deduplicate, Calculate Target ---
    if not all_timestamps_list:
        logging.error("No valid data processed from any source. Exiting.")
        exit()

    logging.info("Concatenating all processed data...")
    final_timestamps = np.concatenate(all_timestamps_list)
    final_features = np.concatenate(all_features_list, axis=0)
    final_midpoints = np.concatenate(all_midpoints_list)
    del all_timestamps_list, all_features_list, all_midpoints_list; gc.collect()
    logging.info(f"Initial combined shape before sort/dedup: ts={final_timestamps.shape}, f={final_features.shape}, m={final_midpoints.shape}")

    combined_df = pd.DataFrame({
        'timestamp_ms': final_timestamps,
        'features': [arr for arr in final_features],
        'midpoint': final_midpoints
    })
    del final_timestamps, final_features, final_midpoints; gc.collect()

    logging.info("Sorting combined data by timestamp...")
    combined_df = combined_df.sort_values('timestamp_ms').reset_index(drop=True)

    logging.info("Deduplicating combined timestamps (keeping last)...")
    initial_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['timestamp_ms'], keep='last')
    logging.info(f"Shape after deduplicating (removed {initial_rows - len(combined_df)} duplicates): {combined_df.shape}")
    if combined_df.empty: logging.error("No data left after deduplication"); exit()


    # --- 4. Calculate Target ---
    final_df_with_targets = calculate_target_vectorized(combined_df)
    del combined_df; gc.collect()

    # --- 5. Extract Final X, y and Save ---
    if final_df_with_targets is not None and not final_df_with_targets.empty:
        logging.info("Extracting final X and y arrays...")
        try:
            if 'features' not in final_df_with_targets.columns: logging.error("Column 'features' not found."); exit()
            is_array_mask = final_df_with_targets['features'].apply(lambda x: isinstance(x, np.ndarray))
            if not is_array_mask.all():
                 logging.warning(f"Filtering {sum(~is_array_mask)} non-array elements in 'features'.")
                 final_df_with_targets = final_df_with_targets[is_array_mask]
            if final_df_with_targets.empty: logging.error("No rows left after filtering features."); exit()
            feature_shapes = final_df_with_targets['features'].apply(lambda x: x.shape)
            if not (feature_shapes == (INPUT_DIM_EXPECTED,)).all():
                 logging.error(f"Inconsistent feature shapes. Expected ({INPUT_DIM_EXPECTED},)."); exit()

            X_final = np.stack(final_df_with_targets['features'].values).astype(np.float32)
            y_final = final_df_with_targets['target'].values.astype(np.int64)
            del final_df_with_targets; gc.collect()

            logging.info(f"Final Data Shapes - X: {X_final.shape}, y: {y_final.shape}")

            if X_final.shape[0] != y_final.shape[0]: logging.error("FATAL: X/y row count mismatch."); exit()
            if X_final.ndim != 2 or X_final.shape[1] != INPUT_DIM_EXPECTED: logging.error("FATAL: Final X dimension incorrect."); exit()
            if np.isnan(X_final).any() or np.isinf(X_final).any(): logging.warning("NaNs/Infs found in final X!")
            if np.isnan(y_final).any(): logging.error("FATAL: NaNs found in final y!"); exit()

            logging.info(f"Saving final combined data...")
            np.save(FINAL_OUTPUT_PATH_X, X_final)
            np.save(FINAL_OUTPUT_PATH_Y, y_final)
            logging.info(f"Saved final X to {FINAL_OUTPUT_PATH_X}")
            logging.info(f"Saved final y to {FINAL_OUTPUT_PATH_Y}")

        except Exception as e:
            logging.error(f"Error during final extraction or saving: {e}", exc_info=True)
    else:
        logging.error("Final data processing failed. No output files generated.")

    overall_end_time = time.time()
    logging.info(f"=== Combined Data Processing Finished ===")
    logging.info(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
    logging.info(f"Total valid rows in final output: {len(X_final) if 'X_final' in locals() else 0}")