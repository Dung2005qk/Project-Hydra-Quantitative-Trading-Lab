import pandas as pd
import joblib
import pickle
import logging
import os
import sys
from typing import Dict, Optional

# --- Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler("extract_data.log", encoding="utf-8"), logging.StreamHandler()]
)

# --- Thông tin File và Symbol ---
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
# Giả định file PKL nằm cùng thư mục với script hoặc cung cấp đường dẫn đầy đủ
PKL_FILENAME = "BTC_USDT_data.pkl"
CSV_OUTPUT_FILENAME = "BTC_USDT_15m_last_1000.csv"
NUM_ROWS_TO_EXTRACT = 1000

def load_data_from_pkl(pkl_filepath: str) -> Optional[Dict]:
    """Tải dữ liệu từ file pkl, thử joblib rồi đến pickle."""
    if not os.path.exists(pkl_filepath):
        logging.error(f"PKL file not found: {pkl_filepath}")
        return None

    loaded_data = None
    try:
        loaded_data = joblib.load(pkl_filepath)
        logging.debug(f"Loaded data using joblib from {pkl_filepath}")
        if not isinstance(loaded_data, dict):
            logging.warning(f"Data loaded with joblib from {pkl_filepath} is not a dict. Trying pickle.")
            loaded_data = None # Reset để thử pickle
    except Exception as e_joblib:
        logging.warning(f"Failed to load {pkl_filepath} with joblib: {e_joblib}. Trying pickle...")
        loaded_data = None

    if loaded_data is None: # Thử pickle nếu joblib lỗi hoặc data không phải dict
        try:
            with open(pkl_filepath, 'rb') as f:
                loaded_data = pickle.load(f)
            logging.debug(f"Loaded data using pickle from {pkl_filepath}")
        except Exception as e_pickle:
            logging.error(f"Failed to load {pkl_filepath} with pickle: {e_pickle}.")
            return None

    if not isinstance(loaded_data, dict):
        logging.error(f"Data in {pkl_filepath} is not a dictionary after trying both methods.")
        return None

    return loaded_data

def extract_and_save_recent_data():
    """Hàm chính để trích xuất và lưu dữ liệu."""
    logging.info(f"Starting data extraction for {SYMBOL} {TIMEFRAME} from {PKL_FILENAME}")

    # 1. Tải dữ liệu từ file PKL
    all_data = load_data_from_pkl(PKL_FILENAME)
    if all_data is None:
        logging.critical("Failed to load data. Exiting.")
        return

    # 2. Lấy DataFrame của timeframe cần thiết
    if TIMEFRAME not in all_data:
        logging.error(f"Timeframe '{TIMEFRAME}' not found in the loaded data.")
        return

    df = all_data[TIMEFRAME]

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Data for timeframe '{TIMEFRAME}' is not a Pandas DataFrame.")
        return

    if df.empty:
        logging.warning(f"DataFrame for {SYMBOL} {TIMEFRAME} is empty. Nothing to extract.")
        return

    logging.info(f"Loaded DataFrame for {SYMBOL} {TIMEFRAME}. Shape: {df.shape}")

    # 3. Trích xuất N dòng cuối cùng
    if len(df) < NUM_ROWS_TO_EXTRACT:
        logging.warning(f"Available data ({len(df)} rows) is less than requested ({NUM_ROWS_TO_EXTRACT}). Extracting all available rows.")
        recent_df = df
    else:
        recent_df = df.tail(NUM_ROWS_TO_EXTRACT)
        logging.info(f"Extracted the last {len(recent_df)} rows.")

    # 4. Lưu vào file CSV
    try:
        # Đảm bảo index (timestamp) cũng được lưu vào CSV
        recent_df.to_csv(CSV_OUTPUT_FILENAME, index=True)
        logging.info(f"Successfully saved the last {len(recent_df)} rows to {CSV_OUTPUT_FILENAME}")
    except Exception as e:
        logging.error(f"Failed to save data to CSV file {CSV_OUTPUT_FILENAME}: {e}", exc_info=True)

if __name__ == "__main__":
    extract_and_save_recent_data()
    logging.info("Extraction script finished.")