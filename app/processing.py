# app/processing.py

import cv2
import numpy as np
from PIL import Image
import re
from thefuzz import process as fuzzy_process

from .utils import is_checkbox_ticked

def _preprocess_roi_for_ocr(roi_image):
    """Hàm nội bộ: Áp dụng các bộ lọc để làm rõ chữ viết tay."""
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(dilated, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# ==============================================================================
# === BẮT ĐẦU VÙNG MÃ NGUỒN HẬU XỬ LÝ NÂNG CAO (GIỮ NGUYÊN) ===
# ==============================================================================

def _post_process_date_regex(text):
    """
    Sử dụng Regex để làm sạch và chuẩn hóa chuỗi ngày tháng.
    Cố gắng đưa về định dạng DD/MM/YYYY.
    """
    if not text:
        return ""
    
    # 1. Loại bỏ tất cả các ký tự không phải là số
    digits = re.sub(r'\D', '', text)
    
    # 2. Nếu OCR đọc được chính xác 8 số (ví dụ: "06122014"), định dạng lại ngay
    if len(digits) == 8:
        return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
        
    # 3. Nếu OCR đọc ra các ký tự nhiễu (ví dụ: "06.12. 2014" hoặc "06 12 2014")
    #    Regex này sẽ tìm các nhóm số trông giống ngày/tháng/năm
    match = re.search(r'(\d{1,2})[./\s]*(\d{1,2})[./\s]*(\d{4})', text)
    if match:
        day, month, year = match.groups()
        # zfill(2) đảm bảo ngày và tháng luôn có 2 chữ số (vd: '6' -> '06')
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        
    # 4. Nếu tất cả các cách trên đều thất bại, trả về chuỗi số đã được làm sạch
    return text.strip()

def _post_process_class_regex(text):
    """
    Sử dụng Regex để trích xuất tên lớp học (vd: "5A", "9B1").
    Mẫu này tìm kiếm một số theo sau bởi một hoặc hai chữ cái.
    """
    if not text:
        return ""
    
    # Tìm mẫu như "5A", "12C"
    match = re.search(r'(\d{1,2}\s*[A-Z]\d?)', text.upper())
    if match:
        # Trả về kết quả đã loại bỏ khoảng trắng thừa
        return re.sub(r'\s', '', match.group(1))
        
    return text.strip()

def _post_process_text(field_name, text):
    """
    Hàm điều phối chính:
    Gọi hàm hậu xử lý phù hợp dựa trên tên của trường dữ liệu.
    """
    raw_text = text # Giữ lại văn bản gốc để so sánh
    processed_text = text.strip() # Bắt đầu bằng việc xóa khoảng trắng thừa

    # Áp dụng các quy tắc dựa trên tên trường
    if field_name == 'ngay_sinh':
        processed_text = _post_process_date_regex(processed_text)
    elif field_name == 'lop':
        processed_text = _post_process_class_regex(processed_text)
    elif field_name == 'ho_ten':
        # Đối với tên, chỉ cần chuẩn hóa viết hoa là một khởi đầu tốt
        processed_text = ' '.join([word.capitalize() for word in processed_text.split()])
    
    # In ra để so sánh trước và sau khi xử lý
    print(f"    - Hậu xử lý (cho TrOCR): '{raw_text}' -> '{processed_text}'")
    return processed_text

# ==============================================================================
# === KẾT THÚC VÙNG MÃ NGUỒN HẬU XỬ LÝ NÂNG CAO ===
# ==============================================================================


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
            
            # # Logic cũ chỉ chạy 1 model đã bị thay thế hoàn toàn
            else:
                preprocessed_roi = _preprocess_roi_for_ocr(roi_cv2)
                
                text_vietocr = ""
                text_easyocr = ""
                text_trocr = ""

                # 1. Chạy VietOCR
                if ocr_engines.vietocr_engine:
                    try:
                        # VietOCR cần ảnh PIL (RGB)
                        roi_pil_vietocr = Image.fromarray(cv2.cvtColor(preprocessed_roi, cv2.COLOR_BGR2RGB))
                        text_vietocr = ocr_engines.vietocr_engine.predict(roi_pil_vietocr)
                    except Exception as e:
                        print(f"    - LỖI VietOCR: {e}")

                # 2. Chạy EasyOCR
                if ocr_engines.easyocr_engine:
                    try:
                        # EasyOCR làm việc tốt với mảng numpy (BGR)
                        results = ocr_engines.easyocr_engine.readtext(preprocessed_roi, detail=0, paragraph=True)
                        text_easyocr = " ".join(results)
                    except Exception as e:
                        print(f"    - LỖI EasyOCR: {e}")
                
                # 3. Chạy TrOCR (Transformer OCR)
                if ocr_engines.vi_trocr_engine and ocr_engines.vi_trocr_processor:
                    try:
                        # TrOCR cần ảnh PIL (RGB) và processor
                        roi_pil_trocr = Image.fromarray(cv2.cvtColor(preprocessed_roi, cv2.COLOR_BGR2RGB))
                        pixel_values = ocr_engines.vi_trocr_processor(images=roi_pil_trocr, return_tensors="pt").pixel_values.to(ocr_engines.device)
                        # Tăng max_new_tokens để tránh text bị cắt ngắn
                        generated_ids = ocr_engines.vi_trocr_engine.generate(pixel_values, max_new_tokens=128)
                        text_trocr = ocr_engines.vi_trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    except Exception as e:
                        print(f"    - LỖI TrOCR: {e}")
                
                # Hậu xử lý cho một kết quả (ví dụ: TrOCR) để làm kết quả "cuối cùng"
                # Bạn có thể thay đổi logic này (ví dụ: chọn kết quả tốt nhất/voting)
                processed_text = _post_process_text(field_name, text_trocr)
                
                # Lưu tất cả kết quả vào dictionary
                field_result_data = {
                    "vietocr_raw": text_vietocr,
                    "easyocr_raw": text_easyocr,
                    "trocr_raw": text_trocr,
                    "final_processed": processed_text
                }
                
                final_results[field_name] = field_result_data
                
                # In kết quả ra console
                print(f"  - [Text] '{field_name}':")
                print(f"    - VietOCR: '{text_vietocr}'")
                print(f"    - EasyOCR: '{text_easyocr}'")
                print(f"    - TrOCR  : '{text_trocr}'")
            # except KeyError as e:
            print(f"LỖI: Trường '{field_name}' trong cấu hình ROI thiếu thông tin: {e}")
        except Exception as e:
            print(f"LỖI không xác định khi xử lý trường '{field_name}': {e}")
            
    print("--- Pipeline Trích xuất Hoàn tất ---")
    return final_results
