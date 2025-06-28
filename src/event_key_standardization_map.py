import os
from investiny import economic_calendar
from datetime import datetime, timedelta
import re
from typing import Dict
import pandas as pd

EXISTING_EVENT_KEYS = {
    "US_CPI_MOM": "Consumer Price Index MoM",
    "US_NFP": "Nonfarm Payrolls",
    "US_UNEMP_RATE": "Unemployment Rate",
    "FOMC_RATE": "Federal Funds Rate",
    # Thêm các event_key khác từ DEFAULT_SENTIMENT_CONFIG nếu cần
}

# Mapping ban đầu (có thể mở rộng thủ công hoặc tự động)
event_key_standardization_map: Dict[str, str] = {
    "Consumer Price Index MoM": "US_CPI_MOM",
    "CPI m/m": "US_CPI_MOM",
    "Nonfarm Payrolls": "US_NFP",
    "Change in Nonfarm Payrolls": "US_NFP",
    "Unemployment Rate": "US_UNEMP_RATE",
    "Federal Funds Rate": "FOMC_RATE",
    "FOMC Interest Rate Decision": "FOMC_RATE",
}

def fetch_investiny_events(start_date: str, end_date: str) -> list:
    """
    Lấy dữ liệu sự kiện kinh tế từ investiny trong khoảng thời gian cho trước.
    Args:
        start_date: Ngày bắt đầu (YYYY-MM-DD).
        end_date: Ngày kết thúc (YYYY-MM-DD).
    Returns:
        Danh sách các sự kiện (dạng từ điển).
    """
    try:
        # Gọi hàm đúng tên
        # events = economic_data(from_date=start_date, to_date=end_date) # Dòng cũ
        # Cần kiểm tra tham số của economic_calendar
        events = economic_calendar( # Gọi hàm mới
            countries=['united states'], # Ví dụ
            importances=['high', 'medium'],
            # time_zone='UTC', # Kiểm tra tham số timezone
            from_date=start_date, # Kiểm tra định dạng ngày tháng YYYY-MM-DD hay d/m/Y
            to_date=end_date
        )
        # economic_calendar có thể trả về list hoặc DataFrame, cần xử lý phù hợp
        if isinstance(events, pd.DataFrame):
             # Chuyển DataFrame thành list of dicts nếu cần
             return events.to_dict('records')
        elif isinstance(events, list):
             return events # Trả về trực tiếp nếu là list
        else:
             print(f"investiny.economic_calendar returned unexpected type: {type(events)}")
             return []
    except Exception as e:
        print(f"Error fetching investiny calendar data: {e}")
        return []

def extract_event_names(events: list) -> set:
    """
    Trích xuất tất cả tên sự kiện duy nhất từ dữ liệu investiny.
    Args:
        events: Danh sách các sự kiện từ investiny.
    Returns:
        Tập hợp các tên sự kiện duy nhất.
    """
    event_names = set()
    for event in events:
        if "event" in event:
            event_names.add(event["event"])
    return event_names

def suggest_mapping(event_name: str, existing_keys: dict) -> str:
    """
    Đề xuất event_key chuẩn hóa dựa trên tên sự kiện thô.
    Args:
        event_name: Tên sự kiện từ investiny.
        existing_keys: Từ điển các event_key hiện có.
    Returns:
        event_key được đề xuất hoặc "UNMAPPED_<tên viết tắt>".
    """
    # Chuẩn hóa tên sự kiện: loại bỏ ký tự đặc biệt, chuyển thành chữ thường
    normalized_name = re.sub(r"[^\w\s]", "", event_name).lower().strip()

    # Kiểm tra xem tên đã có trong mapping chưa
    for raw_name, key in event_key_standardization_map.items():
        if raw_name.lower() == normalized_name:
            return key

    # Tìm kiếm tương đồng đơn giản (dựa trên từ khóa)
    if "cpi" in normalized_name and "mom" in normalized_name:
        return "US_CPI_MOM"
    elif "nonfarm" in normalized_name or "employment change" in normalized_name:
        return "US_NFP"
    elif "unemployment" in normalized_name:
        return "US_UNEMP_RATE"
    elif "federal" in normalized_name or "fomc" in normalized_name:
        return "FOMC_RATE"

    # Nếu không khớp, tạo key tạm thời để đánh dấu cần xem xét thủ công
    abbr = "".join(word[0].upper() for word in normalized_name.split() if word)
    return f"UNMAPPED_{abbr}"

def update_event_mapping(event_names: set) -> Dict[str, str]:
    """
    Cập nhật event_key_standardization_map với các sự kiện mới từ investiny.
    Args:
        event_names: Tập hợp các tên sự kiện từ investiny.
    Returns:
        Mapping đã cập nhật.
    """
    updated_map = event_key_standardization_map.copy()

    for event_name in event_names:
        if event_name not in updated_map:
            suggested_key = suggest_mapping(event_name, EXISTING_EVENT_KEYS)
            updated_map[event_name] = suggested_key
            print(f"Proposed mapping: '{event_name}' -> '{suggested_key}'")

    return updated_map

def main():
    # Thiết lập khoảng thời gian (ví dụ: 1 tháng gần đây)
    today = datetime.now()
    start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    print(f"Fetching events from {start_date} to {end_date}...")
    events = fetch_investiny_events(start_date, end_date)
    
    if not events:
        print("No events fetched. Please check investiny setup or API access.")
        return

    # Trích xuất tên sự kiện
    event_names = extract_event_names(events)
    print(f"Found {len(event_names)} unique event names:")
    for name in sorted(event_names):
        print(f" - {name}")

    # Cập nhật mapping
    updated_map = update_event_mapping(event_names)
    
    # In mapping hoàn thiện
    print("\nUpdated event_key_standardization_map:")
    print("event_key_standardization_map = {")
    for raw_name, key in sorted(updated_map.items()):
        print(f"    '{raw_name}': '{key}',")
    print("}")

if __name__ == "__main__":
    main()