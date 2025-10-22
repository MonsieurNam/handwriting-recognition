# app/processing.py

import cv2
import numpy as np
from PIL import Image
import re
import os # Thêm import 'os' để tạo đường dẫn file an toàn

# Import các hàm tiện ích từ file utils.py trong cùng thư mục app
from .utils import is_checkbox_ticked
from .config import OUTPUT_PATH # Import đường dẫn OUTPUT_PATH từ config

# ==============================================================================
# === BƯỚC 1: CÁC HÀM TIỀN XỬ LÝ (PRE-PROCESSING) ===
# ==============================================================================

def _remove_horizontal_lines(roi_image):
    """
    Hàm chuyên dụng để phát hiện và loại bỏ các đường kẻ/chấm nằm ngang.
    Sử dụng kỹ thuật biến đổi hình thái học và inpainting.
    """
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    h, w = roi_image.shape[:2]
    # Thử nghiệm với các giá trị này nếu chữ viết bị ảnh hưởng
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 12), 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    cnts, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), 3)

    inpainted_image = cv2.inpaint(roi_image, mask, 7, cv2.INPAINT_NS)
    
    return inpainted_image

def _preprocess_roi_for_ocr(roi_image):
    """
    Áp dụng các bộ lọc cơ bản để làm rõ chữ sau khi đã xóa nhiễu.
    """
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# ==============================================================================
# === BƯỚC 2: CÁC HÀM HẬU XỬ LÝ (POST-PROCESSING) ===
# ==============================================================================

def _post_process_date_regex(text):
    """Sử dụng Regex để chuẩn hóa chuỗi ngày tháng về định dạng DD/MM/YYYY."""
    if not text: return ""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 8: return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
    
    match = re.search(r'(\d{1,2})[./\s-]*(\d{1,2})[./\s-]*(\d{4})', text)
    if match:
        day, month, year = match.groups()
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    return text.strip()

def _post_process_class_regex(text):
    """Sử dụng Regex để trích xuất tên lớp học (vd: "5A", "9B1")."""
    if not text: return ""
    match = re.search(r'(\d{1,2}\s*[A-Z]\d?)', text.upper())
    if match: return re.sub(r'\s', '', match.group(1))
    return text.strip()

def _post_process_text(field_name, text):
    """Hàm điều phối chính, gọi hàm hậu xử lý phù hợp dựa trên tên trường."""
    raw_text = text
    processed_text = text.strip()

    if field_name == 'ngay_sinh':
        processed_text = _post_process_date_regex(processed_text)
    elif field_name == 'lop':
        processed_text = _post_process_class_regex(processed_text)
    elif field_name == 'ho_ten':
        processed_text = ' '.join([word.capitalize() for word in processed_text.split()])
    
    print(f"    - Hậu xử lý cho '{field_name}': '{raw_text}' -> '{processed_text}'")
    return processed_text

# ==============================================================================
# === BƯỚC 3: HÀM ĐIỀU PHỐI PIPELINE CHÍNH ===
# ==============================================================================

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines):
    """
    Thực thi pipeline trích xuất thông tin hoàn chỉnh với tính năng gỡ lỗi.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
    handwritten_fields = ['ho_ten', 'ngay_sinh', 'lop']
    
    if not roi_config:
        print("LỖI: Cấu hình ROI rỗng, không thể xử lý.")
        return {}

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
                # --- LƯU ẢNH GỐC ĐỂ GỠ LỖI ---
                debug_original_path = os.path.join(OUTPUT_PATH, f"DEBUG_{field_name}_0_original.png")
                cv2.imwrite(debug_original_path, roi_cv2)

                # 1. LÀM SẠCH NHIỄU
                cleaned_roi = _remove_horizontal_lines(roi_cv2)
                debug_cleaned_path = os.path.join(OUTPUT_PATH, f"DEBUG_{field_name}_1_cleaned.png")
                cv2.imwrite(debug_cleaned_path, cleaned_roi)
                
                # 2. TIỀN XỬ LÝ
                preprocessed_roi = _preprocess_roi_for_ocr(cleaned_roi)
                debug_preprocessed_path = os.path.join(OUTPUT_PATH, f"DEBUG_{field_name}_2_preprocessed.png")
                cv2.imwrite(debug_preprocessed_path, preprocessed_roi)
                
                roi_pil = Image.fromarray(cv2.cvtColor(preprocessed_roi, cv2.COLOR_BGR2RGB))
                
                recognized_text = ""
                
                # 3. NHẬN DẠNG OCR (với logic Fallback)
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
                
                if not recognized_text and ocr_engines.vietocr_engine:
                    try:
                        print(f"  - [OCR] Thử VietOCR (cuối cùng) cho '{field_name}'...")
                        recognized_text = ocr_engines.vietocr_engine.predict(roi_pil)
                    except Exception as e:
                        print(f"    -> Lỗi VietOCR: {e}")
                
                # 4. HẬU XỬ LÝ
                processed_text = _post_process_text(field_name, recognized_text)
                final_results[field_name] = processed_text
                print(f"  - [Text] '{field_name}': '{processed_text}' (Raw OCR: '{recognized_text}')")

        except KeyError as e:
            print(f"LỖI: Trường '{field_name}' trong cấu hình ROI thiếu thông tin: {e}")
        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi xử lý trường '{field_name}': {e}")
            
    print("--- Pipeline Trích xuất Hoàn tất ---")
    return final_results