import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import time
from typing import List, Dict, Optional, Tuple, Any
import glob
import sqlite3

# ==============================================================================
# Configuration
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("prepare_lob_data.log", encoding='utf-8', mode='w'), # Ghi đè log mỗi lần chạy
        logging.StreamHandler()
    ]
)

# --- File Paths (All on D: Drive) ---
BASE_DATA_DIR = Path("D:/")
DB_FILE_PATH = BASE_DATA_DIR / "BTC 2025-01-01-to-2025-03-01-Seconds.db"
CSV_250MS_PATH = BASE_DATA_DIR / "1-09-1-20.csv" # File 250ms BTC (12 ngày)
CSV_1SEC_DIR = BASE_DATA_DIR # Thư mục chứa file _1sec.csv (BTC & ETH) - Cùng là D:\

OUTPUT_DIR = BASE_DATA_DIR / "processed_lob_data"
OUTPUT_FILE_PREFIX = "combined_lob_training" # Tiền tố cho file output

OUTPUT_DIR.mkdir(exist_ok=True)

# Symbols to process (Script sẽ tự kiểm tra file tồn tại)
SYMBOLS_TO_PROCESS = ["BTC/USDT", "ETH/USDT"]

# LOB Parameters
LOB_LEVELS_NEEDED = 5
# Output Feature Order: [bid_p1..5, bid_q1..5, ask_p1..5, ask_q1..5]
INPUT_DIM_EXPECTED = LOB_LEVELS_NEEDED * 4 # 20 features

# Target Calculation Parameters
PREDICTION_HORIZON_MS = 5000 # Dự đoán 5 giây tới
STATIONARY_THRESHOLD_PCT = 0.0001 # Ngưỡng 0.01% thay đổi để coi là đứng yên

# Processing Parameters
CSV_CHUNKSIZE = 1_000_000 # Điều chỉnh nếu cần dựa trên RAM
DB_ITEM_CHUNKSIZE = 10000 # Số lượng snapshot xử lý mỗi lần đọc từ DB

# --- CSV Column Configurations ---
# 250ms CSV ('1-09-1-20.csv') - NO HEADER, Indices from 0
TIMESTAMP_COL_INDEX = 0
BID_PRICE_INDICES = [i*2 + 2 for i in range(LOB_LEVELS_NEEDED)]   # [2, 4, 6, 8, 10]
BID_VOLUME_INDICES = [i*2 + 3 for i in range(LOB_LEVELS_NEEDED)]  # [3, 5, 7, 9, 11]
ASK_PRICE_INDICES = [i*2 + 22 for i in range(LOB_LEVELS_NEEDED)] # [22, 24, 26, 28, 30]
ASK_VOLUME_INDICES = [i*2 + 23 for i in range(LOB_LEVELS_NEEDED)] # [23, 25, 27, 29, 31]

# Tập hợp tất cả các index cột cần thiết từ file gốc
ALL_REQUIRED_INDICES = [TIMESTAMP_COL_INDEX] + BID_PRICE_INDICES + BID_VOLUME_INDICES + ASK_PRICE_INDICES + ASK_VOLUME_INDICES

# 1sec CSV ('*_1sec.csv') - HAS HEADER
TIMESTAMP_COL_1SEC = 'system_time'
MIDPOINT_COL_1SEC = 'midpoint'
BID_DIST_COLS_1SEC = [f'bids_distance_{i}' for i in range(LOB_LEVELS_NEEDED)]
BID_NOTIONAL_COLS_1SEC = [f'bids_limit_notional_{i}' for i in range(LOB_LEVELS_NEEDED)]
ASK_DIST_COLS_1SEC = [f'asks_distance_{i}' for i in range(LOB_LEVELS_NEEDED)]
ASK_NOTIONAL_COLS_1SEC = [f'asks_limit_notional_{i}' for i in range(LOB_LEVELS_NEEDED)]
REQUIRED_COLS_1SEC = [TIMESTAMP_COL_1SEC, MIDPOINT_COL_1SEC] + BID_DIST_COLS_1SEC + BID_NOTIONAL_COLS_1SEC + ASK_DIST_COLS_1SEC + ASK_NOTIONAL_COLS_1SEC

# ==============================================================================
# Helper Functions
# ==============================================================================

def calculate_mid_price(bid1: float, ask1: float) -> Optional[float]:
    """Calculates mid-price from best bid/ask."""
    # Kiểm tra NaN và ask > bid
    if pd.notna(bid1) and pd.notna(ask1) and ask1 > bid1 and bid1 > 1e-12:
        return (bid1 + ask1) / 2.0
    return None

def calculate_quantity_from_notional(notional: float, price: float) -> Optional[float]:
    """Calculates quantity from notional value and price."""
    if pd.notna(notional) and pd.notna(price) and price > 1e-12: # Use a small positive threshold
        try:
            qty = notional / price
            return max(0.0, qty) # Quantity cannot be negative
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
    return None

def parse_timestamps(timestamp_series: pd.Series, source_hint: str = "") -> pd.Series:
    """
    Robustly parses a Series of timestamps into int64 milliseconds Unix Epoch UTC.
    Handles ISO strings, numeric seconds, and numeric milliseconds.
    Returns Series with dtype Int64 (to handle NA).
    """
    logging.debug(f"Attempting to parse timestamps from {source_hint} (dtype: {timestamp_series.dtype})...")

    # 1. Try direct ISO 8601 / already datetime format first
    try:
        dt_series = pd.to_datetime(timestamp_series, errors='coerce', utc=True)
        if dt_series.notna().sum() > 0: # Check if at least one value was parsed
             # Check if MOST values were parsed this way (heuristic)
            if dt_series.notna().mean() > 0.8: # If >80% parsed, assume this is the correct format
                logging.debug(f"Parsed timestamps as datetime/ISO format from {source_hint}.")
                # Convert valid datetimes to int64 milliseconds, others remain NaT
                ms_series = dt_series.astype(np.int64) // 1_000_000
                return ms_series.astype('Int64') # Use Pandas nullable integer type
    except Exception as e:
        logging.debug(f"Direct datetime parsing failed for {source_hint}: {e}")
        pass # Continue to numeric parsing

    # 2. Try numeric parsing
    try:
        numeric_ts = pd.to_numeric(timestamp_series, errors='coerce')
        # Create a mask for finite values to avoid warnings with NA comparison
        finite_mask = np.isfinite(numeric_ts)
        if not finite_mask.any():
            logging.warning(f"Could not convert any timestamps to numeric for {source_hint}.")
            return pd.Series(pd.NA, index=timestamp_series.index, dtype='Int64')

        # Analyze only the finite values
        finite_numeric_ts = numeric_ts[finite_mask]
        median_val = finite_numeric_ts.median()
        logging.debug(f"Median numeric timestamp value for {source_hint}: {median_val}")

        # Convert based on magnitude, apply only to numeric values
        result_ms = pd.Series(pd.NA, index=timestamp_series.index, dtype='Int64')
        if median_val > 1e11: # Likely milliseconds
            logging.debug(f"Assuming milliseconds for {source_hint}.")
            # Directly use the numeric value if it's already ms
            result_ms.loc[finite_mask] = finite_numeric_ts.astype(np.int64)
        elif median_val > 1e8: # Likely seconds (most common Unix timestamp)
            logging.debug(f"Assuming seconds for {source_hint}.")
            # Convert seconds to milliseconds
            result_ms.loc[finite_mask] = (finite_numeric_ts * 1000).astype(np.int64)
        else:
            logging.warning(f"Numeric timestamps for {source_hint} are outside expected Unix range ({median_val}). Cannot reliably determine unit.")
            # Keep as NA

        return result_ms

    except Exception as e:
        logging.error(f"Error during numeric timestamp parsing for {source_hint}: {e}", exc_info=True)
        return pd.Series(pd.NA, index=timestamp_series.index, dtype='Int64')

# ==============================================================================
# Data Reading and Processing Functions
# ==============================================================================

def read_db_file(filepath: Path, levels_to_fetch: int = LOB_LEVELS_NEEDED) -> Optional[pd.DataFrame]:
    """Reads LOB data from SQLite DB IN CHUNKS OF ITEMS, querying bids/asks per chunk."""
    logging.info(f"Reading SQLite DB file (Chunked Query Mode): {filepath}...")
    if not filepath.exists(): logging.warning(f"DB file not found: {filepath}"); return None

    all_processed_data = []
    conn = None
    total_processed_count = 0
    try:
        conn = sqlite3.connect(f'file:{filepath}?mode=ro', uri=True, timeout=60.0) # Increased timeout just in case
        cursor = conn.cursor() # Use cursor for potentially large item table reading

        logging.info("Fetching item IDs and timestamps (chunked)...")
        items_data = []
        offset = 0
        item_fetch_chunksize = 100000 # Read item IDs in chunks too
        while True:
            # Use cursor.execute for potentially better memory handling on large tables
            cursor.execute(f"SELECT id, time_exchange FROM items ORDER BY id LIMIT {item_fetch_chunksize} OFFSET {offset}")
            rows = cursor.fetchall()
            if not rows: break
            items_data.extend(rows)
            offset += len(rows)
            logging.debug(f"Fetched {offset} item headers...")
        if not items_data: logging.error("No items found in DB."); return None
        items_df = pd.DataFrame(items_data, columns=['item_id', 'time_exchange'])
        del items_data # Free memory

        logging.info(f"Fetched {len(items_df)} items total.")

        # Parse timestamps robustly
        items_df['timestamp_ms'] = parse_timestamps(items_df['time_exchange'], source_hint="DB time_exchange")
        items_df = items_df.dropna(subset=['timestamp_ms'])
        if items_df.empty: logging.error("Could not parse any timestamps from DB."); return None
        items_df['timestamp_ms'] = items_df['timestamp_ms'].astype(np.int64)
        items_df = items_df.sort_values('timestamp_ms').reset_index(drop=True)
        logging.info(f"Successfully parsed {len(items_df)} valid timestamps.")

        num_items = len(items_df)
        start_process_time = time.time()
        log_interval = max(1000, num_items // 100)

        logging.info("Processing snapshots chunk by chunk (querying DB per chunk)...")
        for i in range(0, num_items, DB_ITEM_CHUNKSIZE):
            chunk_start_time = time.time()
            item_chunk = items_df.iloc[i : i + DB_ITEM_CHUNKSIZE]
            chunk_item_ids = tuple(item_chunk['item_id'].unique().tolist()) # Get unique IDs for IN clause
            logging.info(f"Processing item chunk {i // DB_ITEM_CHUNKSIZE + 1}/{ (num_items + DB_ITEM_CHUNKSIZE - 1) // DB_ITEM_CHUNKSIZE } (Items {i+1} to {min(i+DB_ITEM_CHUNKSIZE, num_items)})...")

            if not chunk_item_ids: continue

            # --- Query Bids/Asks for the current chunk of item_ids ---
            placeholders = ','.join('?' * len(chunk_item_ids))
            bids_query = f"SELECT item_id, price, size FROM bids WHERE item_id IN ({placeholders})"
            asks_query = f"SELECT item_id, price, size FROM asks WHERE item_id IN ({placeholders})"
            try:
                # Use Pandas directly with parameters for better type handling potentially
                bids_chunk_df = pd.read_sql_query(bids_query, conn, params=chunk_item_ids)
                asks_chunk_df = pd.read_sql_query(asks_query, conn, params=chunk_item_ids)
                logging.debug(f"Fetched {len(bids_chunk_df)} bids and {len(asks_chunk_df)} asks for chunk.")
            except sqlite3.Error as sql_e: logging.error(f"SQL error fetching chunk bids/asks: {sql_e}"); continue
            except Exception as fetch_e: logging.error(f"Error fetching chunk bids/asks: {fetch_e}"); continue

            if bids_chunk_df.empty or asks_chunk_df.empty: continue

            # Convert types and handle NaNs
            for df in [bids_chunk_df, asks_chunk_df]:
                df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(np.float32)
                df['size'] = pd.to_numeric(df['size'], errors='coerce').astype(np.float32)
                df.dropna(inplace=True)

            # Group and get top N levels (more efficient approach)
            bids_grouped = bids_chunk_df.sort_values('price', ascending=False).groupby('item_id').head(levels_to_fetch)
            asks_grouped = asks_chunk_df.sort_values('price', ascending=True).groupby('item_id').head(levels_to_fetch)
            del bids_chunk_df, asks_chunk_df # Free memory

            # Merge and process valid snapshots
            chunk_processed_data = []
            # Create dicts for faster lookup within the chunk
            bids_dict = bids_grouped.groupby('item_id').agg(list).to_dict('index')
            asks_dict = asks_grouped.groupby('item_id').agg(list).to_dict('index')
            item_chunk_map = item_chunk.set_index('item_id')['timestamp_ms'].to_dict()


            for item_id, timestamp_ms in item_chunk_map.items():
                lob_entry = {'timestamp_ms': timestamp_ms}
                try:
                    bid_data = bids_dict.get(item_id)
                    ask_data = asks_dict.get(item_id)

                    # Check if we have enough levels after grouping
                    if bid_data is None or ask_data is None or \
                       len(bid_data.get('price', [])) < levels_to_fetch or \
                       len(ask_data.get('price', [])) < levels_to_fetch:
                        continue

                    bid_prices = np.array(bid_data['price'], dtype=np.float32)
                    bid_volumes = np.array(bid_data['size'], dtype=np.float32)
                    ask_prices = np.array(ask_data['price'], dtype=np.float32)
                    ask_volumes = np.array(ask_data['size'], dtype=np.float32)

                    # Validity checks
                    if np.any(bid_prices <= 1e-12) or np.any(ask_prices <= 1e-12) or \
                       np.any(bid_volumes < 0) or np.any(ask_volumes < 0): continue
                    if ask_prices[0] <= bid_prices[0]: continue

                    midpoint = (bid_prices[0] + ask_prices[0]) / 2.0
                    # Ensure correct order: bid_p, bid_v, ask_p, ask_v
                    features = np.concatenate([bid_prices, bid_volumes, ask_prices, ask_volumes])
                    if features.shape[0] != INPUT_DIM_EXPECTED: continue

                    lob_entry['features'] = features
                    lob_entry['midpoint'] = midpoint
                    chunk_processed_data.append(lob_entry)
                    total_processed_count += 1

                except KeyError: continue # Should not happen with .get()
                except Exception as row_e: logging.warning(f"Err processing DB item {item_id}: {row_e}"); continue

            if chunk_processed_data:
                all_processed_data.extend(chunk_processed_data)
            chunk_end_time = time.time()
            logging.info(f"Finished DB item chunk in {chunk_end_time - chunk_start_time:.2f}s. Total valid rows so far: {total_processed_count}")
            # Optional: Free memory explicitly after each chunk
            del item_chunk, bids_grouped, asks_grouped, chunk_processed_data
            import gc
            gc.collect()


        logging.info(f"Finished processing {total_processed_count} valid LOB snapshots from DB in {time.time() - start_process_time:.2f} seconds.")
        if not all_processed_data: logging.error("No valid LOB data extracted from the DB."); return None
        return pd.DataFrame(all_processed_data)

    except sqlite3.Error as e: logging.error(f"SQLite error: {e}", exc_info=True); return None
    except Exception as e: logging.error(f"Unexpected error reading DB: {e}", exc_info=True); return None
    finally:
        if conn: conn.close(); logging.debug("Closed DB connection.")


def process_csv_chunk_1sec(chunk: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Processes a chunk of data from a 1-second CSV file."""
    processed_data = []
    if not all(col in chunk.columns for col in REQUIRED_COLS_1SEC):
        logging.error(f"Missing required columns in 1sec chunk. Required: {REQUIRED_COLS_1SEC}, Found: {chunk.columns.tolist()}")
        return None
    try:
        # Parse Timestamp robustly
        chunk['timestamp_ms'] = parse_timestamps(chunk[TIMESTAMP_COL_1SEC], source_hint="CSV 1sec")
        chunk = chunk.dropna(subset=['timestamp_ms'])
        if chunk.empty: return None
        chunk['timestamp_ms'] = chunk['timestamp_ms'].astype(np.int64)

        # Convert numeric columns, coercing errors
        cols_to_convert = [MIDPOINT_COL_1SEC] + BID_DIST_COLS_1SEC + BID_NOTIONAL_COLS_1SEC + ASK_DIST_COLS_1SEC + ASK_NOTIONAL_COLS_1SEC
        for col in cols_to_convert:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce') # Keep as float for now

        # Drop rows with NaN in ANY of the required columns for calculation
        chunk = chunk.dropna(subset=REQUIRED_COLS_1SEC)
        if chunk.empty: return None

        # Convert to float32 after dropping NaNs
        for col in cols_to_convert:
             chunk[col] = chunk[col].astype(np.float32)

    except Exception as e: logging.error(f"Error converting columns in 1sec chunk: {e}"); return None

    # Iterate and calculate features
    for _, row in chunk.iterrows():
        lob_entry = {'timestamp_ms': row['timestamp_ms']}
        midpoint = row[MIDPOINT_COL_1SEC] # Already float32 and not NaN

        # Skip if midpoint is invalid
        if midpoint <= 1e-9: continue

        valid_entry = True
        bid_prices, bid_qtys, ask_prices, ask_qtys = [], [], [], []
        try:
            # Bids: price = midpoint + distance
            for i in range(LOB_LEVELS_NEEDED):
                distance = row[BID_DIST_COLS_1SEC[i]] # Already float32
                notional = row[BID_NOTIONAL_COLS_1SEC[i]] # Already float32
                price = midpoint + distance
                qty = calculate_quantity_from_notional(notional, price)
                # Check validity AFTER calculation
                if pd.isna(price) or price <= 1e-12 or pd.isna(qty) or qty < 0: valid_entry = False; break
                bid_prices.append(price); bid_qtys.append(qty)
            if not valid_entry: continue

            # Asks: price = midpoint + distance
            for i in range(LOB_LEVELS_NEEDED):
                distance = row[ASK_DIST_COLS_1SEC[i]]
                notional = row[ASK_NOTIONAL_COLS_1SEC[i]]
                price = midpoint + distance
                qty = calculate_quantity_from_notional(notional, price)
                if pd.isna(price) or price <= 1e-12 or pd.isna(qty) or qty < 0: valid_entry = False; break
                ask_prices.append(price); ask_qtys.append(qty)
            if not valid_entry: continue

            # Final checks
            if ask_prices[0] <= bid_prices[0]: continue

            features = np.concatenate([bid_prices, bid_qtys, ask_prices, ask_qtys]).astype(np.float32)
            if features.shape[0] != INPUT_DIM_EXPECTED: continue

            lob_entry['features'] = features
            lob_entry['midpoint'] = midpoint
            processed_data.append(lob_entry)
        except Exception as inner_e:
             logging.debug(f"Skipping row in 1sec chunk due to error: {inner_e}")
             continue # Skip row on any calculation error

    if not processed_data: return None
    return pd.DataFrame(processed_data)


def process_csv_chunk_250ms(chunk: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Processes a chunk of data from a 250ms CSV file (no header) efficiently."""
    start_rows = len(chunk)
    logging.debug(f"Processing 250ms chunk with {start_rows} rows.")

    # 1. Kiểm tra số lượng cột tối thiểu
    if chunk.shape[1] <= max(ALL_REQUIRED_INDICES):
        logging.warning(f"Skipping 250ms chunk: Not enough columns ({chunk.shape[1]})")
        return None

    # 2. Chọn các cột cần thiết bằng iloc và đổi tên ngay lập tức
    col_mapping = {TIMESTAMP_COL_INDEX: 'timestamp_orig'}
    feature_names = []
    for i in range(LOB_LEVELS_NEEDED):
        p_bid_idx, v_bid_idx = BID_PRICE_INDICES[i], BID_VOLUME_INDICES[i]
        p_ask_idx, v_ask_idx = ASK_PRICE_INDICES[i], ASK_VOLUME_INDICES[i]
        col_mapping[p_bid_idx] = f'bid_p{i+1}'
        col_mapping[v_bid_idx] = f'bid_v{i+1}'
        col_mapping[p_ask_idx] = f'ask_p{i+1}'
        col_mapping[v_ask_idx] = f'ask_v{i+1}'
        feature_names.extend([f'bid_p{i+1}', f'bid_v{i+1}', f'ask_p{i+1}', f'ask_v{i+1}'])

    try:
        # Lấy các cột theo index, đổi tên, tạo bản copy
        df_processed = chunk.iloc[:, list(col_mapping.keys())].copy()
        df_processed.columns = list(col_mapping.values())
    except IndexError:
        logging.error("IndexError during column selection/renaming in 250ms chunk. Check column indices.")
        return None
    except Exception as e:
        logging.error(f"Error selecting/renaming columns in 250ms chunk: {e}")
        return None

    # 3. Parse Timestamp
    df_processed['timestamp_ms'] = parse_timestamps(df_processed['timestamp_orig'], source_hint="CSV 250ms")
    df_processed = df_processed.dropna(subset=['timestamp_ms'])
    if df_processed.empty: logging.debug("No valid timestamps in 250ms chunk."); return None
    df_processed['timestamp_ms'] = df_processed['timestamp_ms'].astype(np.int64)

    # 4. Convert LOB columns to float32 và xử lý lỗi/NaN
    lob_cols_for_check = [f'bid_p{i+1}' for i in range(LOB_LEVELS_NEEDED)] + \
                         [f'bid_v{i+1}' for i in range(LOB_LEVELS_NEEDED)] + \
                         [f'ask_p{i+1}' for i in range(LOB_LEVELS_NEEDED)] + \
                         [f'ask_v{i+1}' for i in range(LOB_LEVELS_NEEDED)]

    for col in lob_cols_for_check:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    initial_rows_before_dropna = len(df_processed)
    df_processed = df_processed.dropna(subset=lob_cols_for_check)
    rows_dropped_nan = initial_rows_before_dropna - len(df_processed)
    if rows_dropped_nan > 0:
        logging.debug(f"Dropped {rows_dropped_nan} rows due to NaNs in essential LOB columns (250ms chunk).")
    if df_processed.empty:
        logging.debug("No rows remaining after dropping LOB NaNs (250ms chunk).")
        return None

    # Chuyển đổi sang float32 sau khi đã dropna
    for col in lob_cols_for_check:
        df_processed[col] = df_processed[col].astype(np.float32)

    # 5. Áp dụng các kiểm tra logic (vectorized)
    valid_mask = (df_processed['ask_p1'] > df_processed['bid_p1']) & \
                 (df_processed['bid_p1'] > 1e-9) & \
                 (df_processed['ask_p1'] > 1e-9) & \
                 (df_processed[[f'bid_v{i+1}' for i in range(LOB_LEVELS_NEEDED)]] >= 0).all(axis=1) & \
                 (df_processed[[f'ask_v{i+1}' for i in range(LOB_LEVELS_NEEDED)]] >= 0).all(axis=1)

    initial_rows_before_filter = len(df_processed)
    df_processed = df_processed[valid_mask]
    rows_dropped_logic = initial_rows_before_filter - len(df_processed)
    if rows_dropped_logic > 0:
        logging.debug(f"Dropped {rows_dropped_logic} rows due to logic checks (price>0, vol>=0, ask>bid) (250ms chunk).")
    if df_processed.empty:
        logging.debug("No rows remaining after logic checks (250ms chunk).")
        return None

    # 6. Tạo cột 'features' và 'midpoint'
    try:
        # Tạo mảng features hiệu quả hơn
        feature_arrays = df_processed[feature_names].values
        # Kiểm tra shape lần cuối trước khi gán
        if feature_arrays.shape[1] != INPUT_DIM_EXPECTED:
             logging.error(f"Internal error: Feature array shape mismatch ({feature_arrays.shape[1]} vs {INPUT_DIM_EXPECTED}).")
             return None
        # Chuyển thành list các array để lưu vào cột DataFrame
        df_processed['features'] = list(feature_arrays)

        df_processed['midpoint'] = (df_processed['bid_p1'] + df_processed['ask_p1']) / 2.0

    except Exception as e:
        logging.error(f"Error creating features/midpoint columns in 250ms chunk: {e}", exc_info=True)
        return None

    # Chọn các cột cuối cùng
    result_df = df_processed[['timestamp_ms', 'features', 'midpoint']].copy()

    logging.debug(f"Successfully processed chunk, generated {len(result_df)} valid LOB entries.")
    return result_df


def read_csv_files(file_paths: List[Path], file_type: str) -> Optional[pd.DataFrame]:
    """Reads and processes CSV files chunk by chunk."""
    all_processed_dfs = []
    if not file_paths:
        logging.warning(f"No files provided for type '{file_type}'.")
        return None
    logging.info(f"Processing {len(file_paths)} {file_type} CSV files...")
    total_rows_processed = 0
    for i, filepath in enumerate(file_paths):
        logging.info(f"Reading CSV ({i+1}/{len(file_paths)}): {filepath.name}")
        processed_file_chunks = []
        try:
            header_option: Optional[int] = 0 if file_type == "1sec" else None
            process_func = process_csv_chunk_1sec if file_type == "1sec" else process_csv_chunk_250ms

            # Check if file exists before attempting to read
            if not filepath.exists():
                logging.warning(f"CSV file not found: {filepath}. Skipping.")
                continue

            for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=CSV_CHUNKSIZE, header=header_option, low_memory=False)):
                logging.debug(f"Processing chunk {chunk_num+1} from {filepath.name}...")
                processed_chunk_df = process_func(chunk)
                if processed_chunk_df is not None and not processed_chunk_df.empty:
                    processed_file_chunks.append(processed_chunk_df)
                    logging.debug(f"Finished processing chunk {chunk_num+1}, added {len(processed_chunk_df)} rows.")

            if processed_file_chunks:
                file_df = pd.concat(processed_file_chunks, ignore_index=True)
                all_processed_dfs.append(file_df)
                total_rows_processed += len(file_df)
                logging.info(f"Finished file {filepath.name}, total valid rows in file: {len(file_df)}")
            else:
                logging.warning(f"No valid data processed from file: {filepath.name}")

        except pd.errors.EmptyDataError:
            logging.warning(f"Skipping empty CSV file: {filepath.name}")
            continue
        except Exception as e:
            logging.error(f"Error reading/processing CSV file {filepath}: {e}", exc_info=True)

    if not all_processed_dfs:
        logging.warning(f"No data could be processed from {file_type} CSV files.")
        return None

    logging.info(f"Concatenating results from {len(all_processed_dfs)} processed files for {file_type}...")
    combined_df = pd.concat(all_processed_dfs, ignore_index=True)
    logging.info(f"Finished processing {file_type} CSVs. Total valid rows combined: {total_rows_processed}. Shape: {combined_df.shape}")
    return combined_df


def process_and_combine_data(symbols: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Reads, processes, combines data, and creates X, y."""
    all_data_dfs = []

    # 1. Read DB File (Assuming BTC Spot data for now based on schema)
    if DB_FILE_PATH.exists():
        logging.info(f"Processing DB file for BTC: {DB_FILE_PATH}")
        df_db = read_db_file(DB_FILE_PATH)
        if df_db is not None and not df_db.empty:
            all_data_dfs.append(df_db[['timestamp_ms', 'features', 'midpoint']])
            logging.info(f"Added {len(df_db)} rows from DB file.")
        else:
            logging.warning(f"No valid data returned from DB file: {DB_FILE_PATH}")
    else:
        logging.warning(f"DB file not found: {DB_FILE_PATH}")

    # 2. Read 250ms CSV (Assuming BTC Perpetual data)
    if CSV_250MS_PATH.exists():
        logging.info(f"Processing 250ms CSV file: {CSV_250MS_PATH}")
        df_250ms = read_csv_files([CSV_250MS_PATH], "250ms")
        if df_250ms is not None and not df_250ms.empty:
            all_data_dfs.append(df_250ms) # Already has needed columns
            logging.info(f"Added {len(df_250ms)} rows from 250ms CSV file.")
        else:
            logging.warning(f"No valid data returned from 250ms CSV file: {CSV_250MS_PATH}")

    else:
        logging.warning(f"250ms CSV not found: {CSV_250MS_PATH}")

    # 3. Read 1sec CSVs (Can be BTC or ETH)
    for symbol in symbols:
        prefix = symbol.split('/')[0] # BTC or ETH
        sec_files = sorted(list(CSV_1SEC_DIR.glob(f"{prefix}_1sec*.csv")))
        if sec_files:
             logging.info(f"Processing 1sec CSV files for {prefix}...")
             df_1sec = read_csv_files(sec_files, "1sec")
             if df_1sec is not None and not df_1sec.empty:
                 all_data_dfs.append(df_1sec)
                 logging.info(f"Added {len(df_1sec)} rows from {prefix} 1sec CSV files.")
        else:
             logging.warning(f"No 1sec CSV files found for {prefix} in {CSV_1SEC_DIR}")

    if not all_data_dfs:
        logging.error("No data loaded from any source.")
        return None

    # 4. Combine, Sort, Deduplicate
    logging.info(f"Combining data from {len(all_data_dfs)} sources...")
    final_df = pd.concat(all_data_dfs, ignore_index=True)
    logging.info(f"Initial combined shape: {final_df.shape}")
    if final_df.empty: logging.error("Combined dataframe is empty."); return None

    # Ensure timestamp is integer before sorting/deduplicating
    if not pd.api.types.is_integer_dtype(final_df['timestamp_ms']):
         logging.warning("Converting combined timestamp_ms to integer, dropping errors.")
         final_df['timestamp_ms'] = pd.to_numeric(final_df['timestamp_ms'], errors='coerce')
         final_df = final_df.dropna(subset=['timestamp_ms'])
         if final_df.empty: logging.error("No valid timestamps after final conversion."); return None
         final_df['timestamp_ms'] = final_df['timestamp_ms'].astype(np.int64)

    final_df = final_df.sort_values('timestamp_ms').reset_index(drop=True)
    initial_rows = len(final_df)
    final_df = final_df.drop_duplicates(subset=['timestamp_ms'], keep='last')
    logging.info(f"Shape after sorting and deduplicating (removed {initial_rows - len(final_df)} duplicates): {final_df.shape}")
    if final_df.empty: logging.error("No data remaining after deduplication."); return None

    # 5. Calculate Target
    logging.info("Calculating targets using vectorized approach...")
    start_target_time = time.time()

    # Đảm bảo index là duy nhất và liên tục sau khi sắp xếp và deduplicate
    final_df = final_df.reset_index(drop=True)

    timestamps_ms = final_df['timestamp_ms'].values
    mid_prices = final_df['midpoint'].values # Lấy dưới dạng numpy array để tăng tốc
    n_samples = len(final_df)
    targets = np.full(n_samples, np.nan) # Khởi tạo với NaN

    # Tìm index tương lai hiệu quả hơn trên mảng timestamp đã sắp xếp
    future_timestamps = timestamps_ms + PREDICTION_HORIZON_MS
    # searchsorted trả về vị trí nên chèn future_timestamps vào timestamps_ms để giữ thứ tự
    # Đây chính là index của snapshot tương lai (hoặc snapshot ngay sau thời điểm tương lai)
    future_indices = np.searchsorted(timestamps_ms, future_timestamps, side='left')

    # Lọc các index hợp lệ:
    # - future_index phải nhỏ hơn tổng số mẫu (không vượt quá cuối DataFrame)
    # - future_index phải lớn hơn index hiện tại (đảm bảo nhìn về tương lai)
    valid_future_mask = (future_indices < n_samples) & (future_indices > np.arange(n_samples))

    # Lấy các index hiện tại và tương lai hợp lệ
    current_indices = np.arange(n_samples)[valid_future_mask]
    valid_future_indices = future_indices[valid_future_mask]

    # Lấy giá trị mid-price tại các index hiện tại và tương lai trực tiếp từ mảng numpy
    current_mids = mid_prices[current_indices]
    future_mids = mid_prices[valid_future_indices] # Truy cập trực tiếp bằng index số nguyên

    # Tính toán target chỉ cho các cặp hợp lệ (loại bỏ NaN nếu có trong mid_prices)
    # và đảm bảo current_mid > 0 để tránh chia cho 0
    valid_calc_mask = ~np.isnan(current_mids) & ~np.isnan(future_mids) & (current_mids > 1e-9)

    # Lấy các index và giá trị thực sự sẽ dùng để tính toán
    final_valid_indices = current_indices[valid_calc_mask] # Index trong final_df để gán target
    valid_current_mids = current_mids[valid_calc_mask]
    valid_future_mids = future_mids[valid_calc_mask]

    # Tính phần trăm thay đổi giá
    price_change_pct = np.full_like(targets, np.nan, dtype=np.float64) # Init với NaN
    # Tính toán chỉ trên các giá trị hợp lệ
    price_change_pct[final_valid_indices] = (valid_future_mids - valid_current_mids) / valid_current_mids

    # Gán target dựa trên ngưỡng (chỉ gán tại các index hợp lệ)
    targets[final_valid_indices[price_change_pct[final_valid_indices] > STATIONARY_THRESHOLD_PCT]] = 1  # Up
    targets[final_valid_indices[price_change_pct[final_valid_indices] < -STATIONARY_THRESHOLD_PCT]] = 0 # Down
    targets[final_valid_indices[np.abs(price_change_pct[final_valid_indices]) <= STATIONARY_THRESHOLD_PCT]] = 2 # Stationary

    # Gán lại vào DataFrame và lọc NaN
    final_df['target'] = targets
    final_df = final_df.dropna(subset=['target']) # Loại bỏ các hàng không tính được target
    if final_df.empty: logging.error("No data remaining after target calculation."); return None
    final_df['target'] = final_df['target'].astype(np.int64)
    logging.info(f"Target calculation finished in {time.time() - start_target_time:.2f} seconds.")
    logging.info(f"Shape after calculating and filtering targets: {final_df.shape}")

    # 6. Extract Final X, y
    try:
        # Ensure 'features' column exists and contains numpy arrays
        if 'features' not in final_df.columns:
             logging.error("Column 'features' not found in final DataFrame.")
             return None
        if not final_df['features'].apply(lambda x: isinstance(x, np.ndarray)).all():
             logging.error("Column 'features' contains non-array elements.")
             # Attempt to filter out non-array rows
             final_df = final_df[final_df['features'].apply(lambda x: isinstance(x, np.ndarray))]
             if final_df.empty:
                 logging.error("No rows left after filtering non-array features.")
                 return None
        X = np.stack(final_df['features'].values)
        y = final_df['target'].values
    except Exception as e: logging.error(f"Error stacking features: {e}."); return None

    logging.info(f"Final Data Shapes - X: {X.shape}, y: {y.shape}")
    if X.shape[0] != y.shape[0] or (X.ndim == 2 and X.shape[1] != INPUT_DIM_EXPECTED):
        logging.error(f"FATAL: Final X/y shape mismatch or incorrect feature dimension."); return None
    if np.isnan(X).any() or np.isinf(X).any(): logging.warning("NaNs or Infs found in final X array!")
    if np.isnan(y).any(): logging.warning("NaNs found in final y array!")

    return X, y

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    logging.info("=== Starting LOB Data Preparation ===")
    overall_start_time = time.time()

    # Determine available symbols based on file existence
    symbols_available = []
    if DB_FILE_PATH.exists():
        # Assuming DB file is BTC based on name, add BTC if needed
        if "BTC/USDT" in SYMBOLS_TO_PROCESS and "BTC/USDT" not in symbols_available:
            symbols_available.append("BTC/USDT")
    if CSV_250MS_PATH.exists(): # Assuming 250ms is BTC
        if "BTC/USDT" in SYMBOLS_TO_PROCESS and "BTC/USDT" not in symbols_available:
            symbols_available.append("BTC/USDT")
    # Check for 1sec files for each symbol in config
    for sym in SYMBOLS_TO_PROCESS:
        prefix = sym.split('/')[0]
        if any(CSV_1SEC_DIR.glob(f"{prefix}_1sec*.csv")):
            if sym not in symbols_available:
                symbols_available.append(sym)

    if not symbols_available:
        logging.error(f"No input data files found for symbols {SYMBOLS_TO_PROCESS} in D:/. Exiting.")
        exit()

    logging.info(f"Found data for symbols: {symbols_available}")
    logging.info(f"Processing data located in: {BASE_DATA_DIR}")

    # Process data for the available symbols
    result = process_and_combine_data(symbols_available) # Pass the list of symbols found

    # Save results
    if result:
        X_final, y_final = result
        logging.info(f"Successfully processed data. Final X shape: {X_final.shape}, y shape: {y_final.shape}")
        output_path_x = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_X.npy"
        output_path_y = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_y.npy"
        try:
            np.save(output_path_x, X_final)
            np.save(output_path_y, y_final)
            logging.info(f"Saved processed X to {output_path_x}")
            logging.info(f"Saved processed y to {output_path_y}")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}", exc_info=True)
    else:
        logging.error("Data processing failed. No output files generated.")

    overall_end_time = time.time()
    logging.info(f"=== LOB Data Preparation Finished ===")
    logging.info(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")

    #thêm chỉ số