# app/processing.py

import cv2
import numpy as np
from PIL import Image
import re
from thefuzz import process as fuzzy_process

from .utils import is_checkbox_ticked

# Giữ nguyên các hàm hậu xử lý đã tạo ở bước trước
def _preprocess_roi_for_ocr(roi_image):
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def _post_process_date_regex(text):
    if not text: return ""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 8: return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
    match = re.search(r'(\d{1,2})[./\s]*(\d{1,2})[./\s]*(\d{4})', text)
    if match:
        day, month, year = match.groups()
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    return text.strip()

def _post_process_class_regex(text):
    if not text: return ""
    match = re.search(r'(\d{1,2}\s*[A-Z]\d?)', text.upper())
    if match: return re.sub(r'\s', '', match.group(1))
    return text.strip()

def _post_process_text(field_name, text):
    processed_text = text.strip()
    if field_name == 'ngay_sinh': processed_text = _post_process_date_regex(processed_text)
    elif field_name == 'lop': processed_text = _post_process_class_regex(processed_text)
    elif field_name == 'ho_ten': processed_text = ' '.join([word.capitalize() for word in processed_text.split()])
    print(f"    - Hậu xử lý cho '{field_name}': '{text}' -> '{processed_text}'")
    return processed_text

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines):
    """
    Thực thi pipeline trích xuất thông tin với logic OCR fallback nâng cao.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
    # Định nghĩa các trường chữ viết tay cần xử lý đặc biệt
    handwritten_fields = ['ho_ten', 'ngay_sinh', 'lop']
    
    if not roi_config: return {}

    for field_name, data in roi_config.items():
        try:
            field_type = data.get('type', 'text')
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            roi_cv2 = aligned_image[y:y+h, x:x+w]
            if roi_cv2.size == 0: continue
            
            if field_type == 'checkbox':
                final_results[field_name] = is_checkbox_ticked(roi_cv2)
                print(f"  - [Checkbox] '{field_name}': {final_results[field_name]}")
            else:
                preprocessed_roi = _preprocess_roi_for_ocr(roi_cv2)
                roi_pil = Image.fromarray(cv2.cvtColor(preprocessed_roi, cv2.COLOR_BGR2RGB))
                
                recognized_text = ""
                
                # <<< BẮT ĐẦU LOGIC OCR FALLBACK THÔNG MINH >>>
                
                # Ưu tiên TrOCR cho các trường chữ viết tay
                if field_name in handwritten_fields:
                    # 1. Thử mô hình TrOCR chuyên biệt cho chữ viết tay
                    if ocr_engines.trocr_handwritten_model:
                        try:
                            print(f"  - [OCR] Thử TrOCR-Handwritten cho '{field_name}'...")
                            processor = ocr_engines.trocr_handwritten_processor
                            model = ocr_engines.trocr_handwritten_model
                            pixel_values = processor(images=roi_pil, return_tensors="pt").pixel_values.to(ocr_engines.device)
                            generated_ids = model.generate(pixel_values, max_length=64)
                            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        except Exception as e:
                            print(f"    -> Lỗi TrOCR-Handwritten: {e}")

                    # 2. Nếu mô hình 1 thất bại, thử mô hình TrOCR tổng quát
                    if not recognized_text and ocr_engines.trocr_general_model:
                        try:
                            print(f"  - [OCR] Thử TrOCR-General (dự phòng) cho '{field_name}'...")
                            processor = ocr_engines.trocr_general_processor
                            model = ocr_engines.trocr_general_model
                            pixel_values = processor(images=roi_pil, return_tensors="pt").pixel_values.to(ocr_engines.device)
                            generated_ids = model.generate(pixel_values, max_length=64)
                            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        except Exception as e:
                            print(f"    -> Lỗi TrOCR-General: {e}")

                # 3. Nếu không phải trường viết tay, hoặc tất cả TrOCR đều thất bại, dùng VietOCR
                if not recognized_text and ocr_engines.vietocr_engine:
                    try:
                        print(f"  - [OCR] Thử VietOCR (cuối cùng) cho '{field_name}'...")
                        recognized_text = ocr_engines.vietocr_engine.predict(roi_pil)
                    except Exception as e:
                        print(f"    -> Lỗi VietOCR: {e}")
                
                # <<< KẾT THÚC LOGIC OCR FALLBACK >>>
                
                processed_text = _post_process_text(field_name, recognized_text)
                final_results[field_name] = processed_text
                print(f"  - [Text] '{field_name}': '{processed_text}' (Raw OCR: '{recognized_text}')")

        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi xử lý trường '{field_name}': {e}")
            
    print("--- Pipeline Trích xuất Hoàn tất ---")
    return final_results