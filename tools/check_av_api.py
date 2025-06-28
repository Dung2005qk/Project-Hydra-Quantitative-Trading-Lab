import asyncio
import aiohttp
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import platform
if platform.system() == "Windows":
    try:
         # Thử dùng policy selector nếu proactor gây lỗi
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
         logging.info("Set asyncio event loop policy to WindowsSelectorEventLoopPolicy.")
    except Exception as policy_e:
         logging.warning(f"Could not set WindowsSelectorEventLoopPolicy: {policy_e}. Using default.")

# --- Cấu hình ---
AV_API_KEY = os.getenv("AV_API_KEY", "6WRTJT84D47C72CY") # Lấy key từ env hoặc dùng key test
AV_BASE_URL = "https://www.alphavantage.co/query"
API_TIMEOUT = 20 # AV có thể chậm hơn

# *** DANH SÁCH FUNCTION AV CẦN KIỂM TRA (Lấy từ av_function_map_historical) ***
AV_FUNCTIONS_TO_TEST = [
    "CPI",
    "NONFARM_PAYROLL",
    "UNEMPLOYMENT_RATE",
    "REAL_GDP",
    "RETAIL_SALES",
    "MANUFACTURING_PMI", # Tên này có thể không đúng, AV có thể dùng tên khác
    "PRODUCER_PRICE_INDEX", # Thử cho PPI
    "FEDERAL_FUNDS_RATE",
    # Thêm các function khác bạn muốn kiểm tra từ tài liệu AV
    "TREASURY_YIELD", # Ví dụ
    "INVALID_FUNCTION_NAME_TEST" # Thêm một tên sai để kiểm tra lỗi
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def check_av_function(session: aiohttp.ClientSession, function_name: str):
    """Thử gọi API AV với một function và kiểm tra lỗi."""
    params = {
        "function": function_name,
        "apikey": AV_API_KEY
    }
    # Thêm tham số bắt buộc tối thiểu cho từng function nếu biết
    if function_name == "CPI": params["interval"] = "monthly"
    if function_name == "REAL_GDP": params["interval"] = "quarterly"
    if function_name == "TREASURY_YIELD": params["interval"] = "monthly"; params["maturity"] = "10year" # Ví dụ

    logging.info(f"Kiểm tra AV function: '{function_name}'...")
    try:
        async with session.get(AV_BASE_URL, params=params, timeout=API_TIMEOUT) as response:
            data = await response.json() # AV thường trả về JSON ngay cả khi lỗi
            if response.status == 200:
                # Kiểm tra nội dung lỗi phổ biến của AV
                if isinstance(data, dict) and ("Error Message" in data or "Information" in data and "Invalid API call" in data["Information"]):
                    error_msg = data.get("Error Message", data.get("Information", "Unknown Error Format"))
                    # Lỗi có thể do tên function sai hoặc thiếu tham số
                    if "Invalid API call" in error_msg or "invalid function" in error_msg.lower():
                         logging.warning(f"  >> LỖI GỌI API: Function '{function_name}' không hợp lệ hoặc thiếu tham số. Lỗi: {error_msg}")
                         return function_name, False, "Invalid Function or Parameters"
                    else: # Lỗi khác nhưng status 200?
                         logging.warning(f"  >> CÓ THỂ LỖI: API trả về lỗi dù status 200 cho '{function_name}'. Lỗi: {error_msg}")
                         return function_name, False, f"Error in Response: {error_msg}"
                elif isinstance(data, dict) and ("Note" in data or "Information" in data and "API key" in data["Information"]):
                     # Có thể là thông báo giới hạn API của gói miễn phí
                     info_msg = data.get("Note", data.get("Information"))
                     logging.warning(f"  >> THÔNG BÁO API: Nhận thông báo từ AV cho '{function_name}'. Có thể là giới hạn gói miễn phí/premium? Msg: {info_msg}")
                     # Vẫn coi là 'thành công' vì function có tồn tại, chỉ là bị giới hạn
                     return function_name, True, f"API Info/Limit: {info_msg}"
                elif isinstance(data, dict) and ("data" in data or len(data) > 1): # Có vẻ thành công nếu có key 'data' hoặc nhiều key khác
                     logging.info(f"  >> THÀNH CÔNG (Có vẻ): API trả về dữ liệu hợp lệ cho '{function_name}'.")
                     return function_name, True, None
                else: # Trường hợp không rõ
                    logging.warning(f"  >> KHÔNG RÕ: Phản hồi không xác định cho '{function_name}'. Response: {str(data)[:100]}...")
                    return function_name, False, "Unknown Response Format"

            elif response.status == 401 or response.status == 403:
                 logging.error(f"  >> LỖI XÁC THỰC ({response.status}): API key AV không hợp lệ.")
                 return function_name, False, f"Authentication Error ({response.status})"
            elif response.status == 429:
                 logging.error(f"  >> RATE LIMIT ({response.status}): Đã chạm giới hạn API call cho AV.")
                 return function_name, False, "Rate Limit (429)"
            else:
                error_text = await response.text()
                logging.error(f"  >> LỖI KHÁC ({response.status}): Lỗi khi gọi API cho '{function_name}'. Response: {error_text[:100]}...")
                return function_name, False, f"HTTP Error {response.status}"

    except asyncio.TimeoutError:
        logging.error(f"  >> TIMEOUT: Hết thời gian chờ khi gọi API cho '{function_name}'.")
        return function_name, False, "Timeout"
    except aiohttp.ClientConnectionError as e:
         logging.error(f"  >> LỖI KẾT NỐI: Lỗi kết nối khi gọi API cho '{function_name}': {e}")
         return function_name, False, "Connection Error"
    except Exception as e:
        logging.error(f"  >> LỖI KHÔNG XÁC ĐỊNH: Lỗi khi kiểm tra '{function_name}': {e}", exc_info=True)
        return function_name, False, f"Unexpected Error: {e}"

async def main_av_check():
    """Chạy kiểm tra đồng thời cho các hàm chức năng AV."""
    valid_functions = {}
    invalid_functions = {}
    async with aiohttp.ClientSession() as session:
        tasks = [check_av_function(session, func) for func in AV_FUNCTIONS_TO_TEST]
        results = await asyncio.gather(*tasks)

        for func_name, success, details in results:
            if success:
                valid_functions[func_name] = "OK" if details is None else details
            else:
                invalid_functions[func_name] = details

    print("\n--- Kết quả kiểm tra Alpha Vantage Functions ---")
    print("\nFunction Hợp lệ (API không báo lỗi function/params hoặc có dữ liệu):")
    if valid_functions:
        for name, status in valid_functions.items():
            print(f" - '{name}' : {status}")
    else:
        print(" (Không tìm thấy function hợp lệ nào)")

    print("\nFunction KHÔNG Hợp lệ hoặc Có Lỗi:")
    if invalid_functions:
        for name, reason in invalid_functions.items():
            print(f" - '{name}' : {reason}")
    else:
        print(" (Tất cả function đã kiểm tra đều có vẻ hợp lệ hoặc bị giới hạn API)")

    print("\n>> Gợi ý: Cập nhật 'av_function_map_historical' trong config, chỉ giữ lại các function hợp lệ.")

# Chạy kiểm tra AV
asyncio.run(main_av_check())

# --- Helper _safe_float và _deep_merge_config (nếu cần cho script độc lập) ---
def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None: return default
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        v = value.strip().replace('%','').replace('K','e3').replace('M','e6').replace('B','e9').replace(',','')
        if not v or v == '-' or v == '.': return default
        try: return float(v)
        except (ValueError, TypeError): return default
    return default

def _deep_merge_config(default: Dict, custom: Dict):
    for key, value in custom.items():
        if isinstance(value, dict) and isinstance(default.get(key), dict): _deep_merge_config(default[key], value)
        else: default[key] = value