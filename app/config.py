# app/config.py

import os
import json

# --- ĐƯỜNG DẪN CHÍNH ---
# Giả định script được chạy từ thư mục gốc của dự án
PROJECT_PATH = './'
INPUT_PATH = os.path.join(PROJECT_PATH, 'Data_Input')
TEMPLATE_PATH = os.path.join(PROJECT_PATH, 'Data_Templates')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Data_Output')

# --- ĐƯỜNG DẪN FILE CỤ THỂ ---
TEMPLATE_IMAGE_PATH = os.path.join(TEMPLATE_PATH, 'template_form.jpg')
ROI_CONFIG_PATH = os.path.join(TEMPLATE_PATH, 'roi_template.json')

def load_roi_config():
    """Tải và trả về cấu hình ROI từ file JSON."""
    try:
        with open(ROI_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file cấu hình ROI tại: {ROI_CONFIG_PATH}")
        return None
    except json.JSONDecodeError:
        print(f"LỖI: File cấu hình ROI không phải là file JSON hợp lệ: {ROI_CONFIG_PATH}")
        return None

# Tạo các thư mục cần thiết nếu chúng chưa tồn tại
def create_directories():
    """Tạo tất cả các thư mục cần thiết cho dự án."""
    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(TEMPLATE_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print("Các thư mục dự án đã được kiểm tra/tạo.")