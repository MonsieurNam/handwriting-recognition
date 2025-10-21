# app/processing.py

import cv2
import numpy as np
from PIL import Image
from app.config import PREPROCESSING_CONFIG
from pyvi import ViTokenizer
from .utils import is_checkbox_ticked
from .preprocessing import advanced_preprocessor

def _preprocess_roi_for_ocr(roi_image, field_name):
    """UPDATED: Use advanced pipeline"""
    if PREPROCESSING_CONFIG['barcode_detection']:
        return advanced_preprocessor.process_roi(roi_image, field_name)
    else:
        # Fallback to old method
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(dilated, -1, sharpen_kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
def _preprocess_roi_for_ocr(roi_image):
    """Hàm nội bộ: Áp dụng các bộ lọc để làm rõ chữ viết tay."""
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(dilated, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def _correct_vietnamese_name(text):
    """Hàm nội bộ: Chuẩn hóa cách viết hoa cho tên Tiếng Việt."""
    if not text:
        return ""
    return ' '.join([word.capitalize() for word in text.split()])

def _post_process_text(field_name, text):
    """Áp dụng các quy tắc hậu xử lý dựa trên tên trường."""
    if field_name == 'ho_ten':
        return _correct_vietnamese_name(text)
    # Thêm các quy tắc khác cho ngày tháng, số... nếu cần
    return text.strip()

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines):
    """
    Thực thi pipeline trích xuất thông tin hoàn chỉnh từ ảnh đã được căn chỉnh.

    Args:
        aligned_image (numpy.ndarray): Ảnh đã được căn chỉnh.
        roi_config (dict): Dữ liệu cấu hình các vùng quan tâm (ROI).
        ocr_engines (OCREngines): Đối tượng chứa các model OCR đã khởi tạo.

    Returns:
        dict: Kết quả trích xuất dưới dạng dictionary.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
    if not roi_config:
        print("LỖI: Cấu hình ROI rỗng, không thể xử lý.")
        return final_results

    for field_name, data in roi_config.items():
        try:
            field_type = data.get('type', 'text')
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            
            roi_cv2 = aligned_image[y:y+h, x:x+w]
            if roi_cv2.size == 0:
                print(f"CẢNH BÁO: ROI cho trường '{field_name}' bị rỗng.")
                continue
            
            if field_type == 'checkbox':
                result = is_checkbox_ticked(roi_cv2)
                final_results[field_name] = result
                print(f"  - [Checkbox] '{field_name}': {result}")
            else:
                # 1. Tiền xử lý ROI
                preprocessed_roi = _preprocess_roi_for_ocr(roi_cv2)
                
                # 2. Chạy OCR với VietOCR (hoặc engine chính của bạn)
                recognized_text = ""
                if ocr_engines.vietocr_engine:
                    try:
                        roi_pil = Image.fromarray(cv2.cvtColor(preprocessed_roi, cv2.COLOR_BGR2RGB))
                        recognized_text = ocr_engines.vietocr_engine.predict(roi_pil)
                    except Exception as e:
                        print(f"LỖI OCR VietOCR tại trường '{field_name}': {e}")
                
                # (Tùy chọn) Có thể thêm logic để chạy EasyOCR nếu VietOCR thất bại
                
                # 3. Hậu xử lý kết quả
                processed_text = _post_process_text(field_name, recognized_text)
                final_results[field_name] = processed_text
                print(f"  - [Text] '{field_name}': '{processed_text}' (Raw: '{recognized_text}')")

        except KeyError as e:
            print(f"LỖI: Trường '{field_name}' trong cấu hình ROI thiếu thông tin: {e}")
        except Exception as e:
            print(f"LỖI không xác định khi xử lý trường '{field_name}': {e}")
            
    print("--- Pipeline Trích xuất Hoàn tất ---")
    return final_results