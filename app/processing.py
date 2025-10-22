# app/processing.py

import cv2
import numpy as np
from PIL import Image
import re
import os

from .utils import is_checkbox_ticked
from .config import OUTPUT_PATH

# ==============================================================================
# === BƯỚC 1: HÀM TIỀN XỬ LÝ NÂNG CAO (THAY THẾ TOÀN BỘ CODE CŨ) ===
# ==============================================================================

def _advanced_preprocess_for_ocr(roi_image):
    """
    Pipeline tiền xử lý nâng cao để xử lý nhiễu hạt và đường kẻ.
    """
    # Bước 1: Chuyển sang ảnh xám
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Bước 2: Làm mịn ảnh để loại bỏ nhiễu hạt (texture của giấy)
    # MedianBlur rất hiệu quả với nhiễu "salt-and-pepper".
    # Kích thước kernel là 3 (phải là số lẻ).
    blurred = cv2.medianBlur(gray, 3)

    # Bước 3: Nhị phân hóa bằng phương pháp Otsu để tách biệt chữ và nền
    # Sau khi làm mịn, Otsu sẽ hoạt động rất hiệu quả.
    # THRESH_BINARY_INV: Chữ sẽ thành màu trắng (255), nền thành màu đen (0).
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Bước 4: Loại bỏ các đường chấm trên ảnh đen trắng đã sạch
    h, w = roi_image.shape[:2]
    # Tạo kernel ngang để phát hiện đường kẻ
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    # Phát hiện và xóa chúng
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # Tô màu đen lên các đường kẻ đã phát hiện
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), 2)

    # Bước 5: Chuyển ảnh đen trắng cuối cùng về định dạng 3 kênh (BGR)
    # mà các engine OCR mong đợi.
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ==============================================================================
# === BƯỚC 2: CÁC HÀM HẬU XỬ LÝ (GIỮ NGUYÊN) ===
# ==============================================================================

def _post_process_date_regex(text):
    if not text: return ""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 8: return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
    match = re.search(r'(\d{1,2})[./\s-]*(\d{1,2})[./\s-]*(\d{4})', text)
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
    raw_text = text
    processed_text = text.strip()
    if field_name == 'ngay_sinh': processed_text = _post_process_date_regex(processed_text)
    elif field_name == 'lop': processed_text = _post_process_class_regex(processed_text)
    elif field_name == 'ho_ten': processed_text = ' '.join([word.capitalize() for word in processed_text.split()])
    print(f"    - Hậu xử lý cho '{field_name}': '{raw_text}' -> '{processed_text}'")
    return processed_text

# ==============================================================================
# === BƯỚC 3: HÀM ĐIỀU PHỐI PIPELINE CHÍNH (CẬP NHẬT ĐỂ GỌI HÀM MỚI) ===
# ==============================================================================

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines):
    """
    Thực thi pipeline trích xuất thông tin với tiền xử lý nâng cao.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
    handwritten_fields = ['ho_ten', 'ngay_sinh', 'lop']
    
    if not roi_config: return {}

    for field_name, data in roi_config.items():
        try:
            field_type = data.get('type', 'text')
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            
            roi_cv2 = aligned_image[y:y+h, x:x+w]
            if roi_cv2.size == 0: continue
            
            if field_type == 'checkbox':
                result = is_checkbox_ticked(roi_cv2)
                final_results[field_name] = result
                print(f"  - [Checkbox] '{field_name}': {result}")
            else:
                # --- THAY ĐỔI: GỌI DUY NHẤT HÀM TIỀN XỬ LÝ NÂNG CAO ---
                print(f"  - [Preprocess] Áp dụng pipeline nâng cao cho '{field_name}'...")
                preprocessed_roi = _advanced_preprocess_for_ocr(roi_cv2)

                # --- LƯU ẢNH GỠ LỖI ---
                # Lưu ảnh gốc và ảnh sau khi xử lý để so sánh
                cv2.imwrite(os.path.join(OUTPUT_PATH, f"DEBUG_{field_name}_0_original.png"), roi_cv2)
                cv2.imwrite(os.path.join(OUTPUT_PATH, f"DEBUG_{field_name}_1_processed.png"), preprocessed_roi)
                
                roi_pil = Image.fromarray(preprocessed_roi)
                
                recognized_text = ""
                
                # --- LOGIC OCR FALLBACK (GIỮ NGUYÊN) ---
                if field_name in handwritten_fields:
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

                    if not recognized_text and o-cr_engines.trocr_general_model:
                        try:
                            print(f"  - [OCR] Thử TrOCR-General (dự phòng) cho '{field_name}'...")
                            processor = ocr_engines.trocr_general_processor
                            model = ocr_engines.trocr_general_model
                            pixel_values = processor(images=roi_pil, return_tensors="pt").pixel_values.to(ocr_engines.device)
                            generated_ids = model.generate(pixel_values, max_length=64)
                            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        except Exception as e:
                            print(f"    -> Lỗi TrOCR-General: {e}")
                
                if not recognized_text and ocr_engines.vietocr_engine:
                    try:
                        print(f"  - [OCR] Thử VietOCR (cuối cùng) cho '{field_name}'...")
                        recognized_text = ocr_engines.vietocr_engine.predict(roi_pil)
                    except Exception as e:
                        print(f"    -> Lỗi VietOCR: {e}")
                
                # --- HẬU XỬ LÝ (GIỮ NGUYÊN) ---
                processed_text = _post_process_text(field_name, recognized_text)
                final_results[field_name] = processed_text
                print(f"  - [Text] '{field_name}': '{processed_text}' (Raw OCR: '{recognized_text}')")

        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi xử lý trường '{field_name}': {e}")
            
    print("--- Pipeline Trích xuất Hoàn tất ---")
    return final_results