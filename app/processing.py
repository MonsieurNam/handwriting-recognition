# app/processing.py

from PIL import Image
import re
import cv2
from .utils import is_checkbox_ticked # Import lại hàm kiểm tra checkbox

def _post_process_text(field_name, text):
    """Hậu xử lý văn bản sau khi trích xuất."""
    if not isinstance(text, str):
        text = str(text) # Đảm bảo đầu vào là chuỗi
        
    raw_text = text
    processed_text = text.strip()
    if field_name == 'ngay_sinh':
        processed_text = re.sub(r'\D', '', processed_text)
    elif field_name == 'lop':
        processed_text = re.sub(r'[^A-Z0-9\s]', '', processed_text.upper())
    elif field_name == 'ho_ten':
        processed_text = ' '.join([word.capitalize() for word in processed_text.split()])
    # Áp dụng cho tất cả các trường số
    elif 'thi_luc' in field_name or field_name in ['ngay', 'thang', 'nam']:
         processed_text = re.sub(r'[^0-9./]', '', processed_text)

    print(f"    - Hậu xử lý cho '{field_name}': '{raw_text}' -> '{processed_text}'")
    return processed_text

def run_combined_pipeline(aligned_image, roi_config, vintern_engine):
    """
    Chạy pipeline kết hợp:
    1. Dùng Vintern cho toàn bộ ảnh để lấy các trường văn bản.
    2. Dùng OpenCV trên từng ROI để xác định trạng thái checkbox.
    3. Kết hợp kết quả.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất Kết hợp ---")
    final_results = {}

    # --- BƯỚC 1: TRÍCH XUẤT VĂN BẢN VỚI VINTERN ---
    print("\n>>> Đang dùng Vintern để trích xuất các trường văn bản từ toàn bộ ảnh...")
    # Chuyển đổi ảnh từ OpenCV (BGR) sang PIL (RGB)
    aligned_pil = Image.fromarray(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
    
    # Gọi Vintern để lấy dictionary chứa các trường văn bản
    text_data_from_vintern = vintern_engine.extract_text_fields_from_image(aligned_pil)
    
    # Hậu xử lý và thêm kết quả văn bản vào final_results
    if text_data_from_vintern:
        print("\n>>> Đang hậu xử lý dữ liệu văn bản từ Vintern...")
        for field_name, value in text_data_from_vintern.items():
            # Chỉ xử lý nếu trường này có trong cấu hình ROI (để tránh các key lạ)
            if field_name in roi_config:
                final_results[field_name] = _post_process_text(field_name, value)
    else:
        print("  - Cảnh báo: Vintern không trả về dữ liệu văn bản nào.")

    # --- BƯỚC 2: XỬ LÝ CHECKBOX VỚI OPENCV ---
    print("\n>>> Đang xử lý các trường checkbox bằng phương pháp phân tích pixel...")
    for field_name, data in roi_config.items():
        if data.get('type') == 'checkbox':
            try:
                x, y, w, h = data['x'], data['y'], data['w'], data['h']
                roi_cv2 = aligned_image[y:y+h, x:x+w]

                if roi_cv2.size == 0:
                    print(f"  - Cảnh báo: Vùng ROI cho checkbox '{field_name}' bị rỗng.")
                    is_checked = False
                else:
                    # Sử dụng hàm is_checkbox_ticked từ utils.py
                    is_checked = is_checkbox_ticked(roi_cv2)
                
                final_results[field_name] = is_checked
                print(f"  - [Checkbox] '{field_name}': {is_checked}")
                
            except Exception as e:
                print(f"  - LỖI khi xử lý checkbox '{field_name}': {e}")
                final_results[field_name] = False # Mặc định là False nếu có lỗi

    print("\n--- Pipeline Trích xuất Hoàn tất ---")
    return final_results