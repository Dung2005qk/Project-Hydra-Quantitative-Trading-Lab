import asyncio
import aiohttp
import os
import logging
from datetime import datetime, timedelta
import platform

if platform.system() == "Windows":
    try:
         # Thử dùng policy selector nếu proactor gây lỗi
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
         logging.info("Set asyncio event loop policy to WindowsSelectorEventLoopPolicy.")
    except Exception as policy_e:
         logging.warning(f"Could not set WindowsSelectorEventLoopPolicy: {policy_e}. Using default.")

# --- Cấu hình ---
FMP_API_KEY = os.getenv("FMP_API_KEY", "LSZAqqdkAYe1vOUJ6NPzOEUu9o7PN42V") # Lấy key từ env hoặc dùng key test
FMP_STABLE_INDICATORS_URL = "https://financialmodelingprep.com/stable/economic-indicators"
API_TIMEOUT = 15

# *** DANH SÁCH TÊN FMP CẦN KIỂM TRA (Lấy từ fmp_indicator_name_map của bạn) ***
FMP_NAMES_TO_TEST = [
    "CPI",
    "Total Nonfarm Payroll",
    "Unemployment Rate",
    "GDP",
    "Retail Sales",
    "ISM Manufacturing PMI", # Hoặc "ISM Manufacturing Index"?
    "Federal Funds Rate",
    "Core PPI", # Tên thử nghiệm cho Core PPI
    "PPI",      # Tên thử nghiệm cho PPI
    "PCE Price Index", # Tên thử nghiệm cho PCE
    "Core PCE Price Index", # Tên thử nghiệm cho Core PCE
    # Thêm các tên khác bạn muốn kiểm tra từ tài liệu FMP hoặc suy đoán
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def check_fmp_indicator(session: aiohttp.ClientSession, indicator_name: str):
    """Thử gọi API FMP cho một tên chỉ số và kiểm tra kết quả."""
    params = {
        "name": indicator_name,
        "from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), # Lấy dữ liệu 1 tháng gần đây
        "to": datetime.now().strftime('%Y-%m-%d'),
        "apikey": FMP_API_KEY
    }
    logging.info(f"Kiểm tra FMP indicator: '{indicator_name}'...")
    try:
        async with session.get(FMP_STABLE_INDICATORS_URL, params=params, timeout=API_TIMEOUT) as response:
            if response.status == 200:
                data = await response.json()
                if isinstance(data, list) and data:
                    logging.info(f"  >> THÀNH CÔNG: Tìm thấy dữ liệu cho '{indicator_name}'. Ví dụ: {data[0]}")
                    return indicator_name, True, None # Tên, Thành công, Lỗi
                elif isinstance(data, list):
                     logging.warning(f"  >> KHÔNG CÓ DỮ LIỆU: API trả về list rỗng cho '{indicator_name}'. Tên có thể đúng nhưng không có data gần đây?")
                     return indicator_name, True, "Empty List" # Tên có thể đúng
                else:
                    logging.warning(f"  >> LỖI PHÂN TÍCH: API không trả về list cho '{indicator_name}'. Response: {str(data)[:100]}...")
                    return indicator_name, False, f"Unexpected format (status {response.status})"
            elif response.status == 401 or response.status == 403:
                 logging.error(f"  >> LỖI XÁC THỰC ({response.status}): API key không hợp lệ hoặc không có quyền truy cập '{indicator_name}'.")
                 return indicator_name, False, f"Authentication Error ({response.status})"
            elif response.status == 404:
                 logging.warning(f"  >> KHÔNG TÌM THẤY (404): Tên chỉ số '{indicator_name}' không tồn tại hoặc sai.")
                 return indicator_name, False, "Not Found (404)"
            else:
                error_text = await response.text()
                logging.error(f"  >> LỖI KHÁC ({response.status}): Lỗi khi gọi API cho '{indicator_name}'. Response: {error_text[:100]}...")
                return indicator_name, False, f"HTTP Error {response.status}"
    except asyncio.TimeoutError:
        logging.error(f"  >> TIMEOUT: Hết thời gian chờ khi gọi API cho '{indicator_name}'.")
        return indicator_name, False, "Timeout"
    except aiohttp.ClientConnectionError as e:
         logging.error(f"  >> LỖI KẾT NỐI: Lỗi kết nối khi gọi API cho '{indicator_name}': {e}")
         return indicator_name, False, "Connection Error"
    except Exception as e:
        logging.error(f"  >> LỖI KHÔNG XÁC ĐỊNH: Lỗi khi kiểm tra '{indicator_name}': {e}", exc_info=True)
        return indicator_name, False, f"Unexpected Error: {e}"

async def main_fmp_check():
    """Chạy kiểm tra đồng thời cho các tên chỉ số FMP."""
    valid_mappings = {}
    invalid_mappings = {}
    async with aiohttp.ClientSession() as session:
        tasks = [check_fmp_indicator(session, name) for name in FMP_NAMES_TO_TEST]
        results = await asyncio.gather(*tasks)

        for name, success, error_details in results:
            if success:
                valid_mappings[name] = "OK" if error_details is None else error_details # Lưu lại tên hợp lệ
            else:
                invalid_mappings[name] = error_details # Lưu lại tên không hợp lệ và lý do

    print("\n--- Kết quả kiểm tra FMP Indicator Names ---")
    print("\nMapping Hợp lệ (API trả về dữ liệu hoặc list rỗng):")
    if valid_mappings:
        for name, status in valid_mappings.items():
            print(f" - '{name}' : {status}")
    else:
        print(" (Không tìm thấy mapping hợp lệ nào)")

    print("\nMapping KHÔNG Hợp lệ hoặc Có Vấn đề:")
    if invalid_mappings:
        for name, reason in invalid_mappings.items():
            print(f" - '{name}' : {reason}")
    else:
        print(" (Tất cả tên đã kiểm tra đều có vẻ hợp lệ hoặc trả về list rỗng)")

    print("\n>> Gợi ý: Cập nhật 'fmp_indicator_name_map' trong config của bạn dựa trên kết quả 'Mapping Hợp lệ'.")

# Chạy kiểm tra FMP
asyncio.run(main_fmp_check())