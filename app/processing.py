# app/processing.py

import cv2
import numpy as np
from PIL import Image
import re
from thefuzz import process as fuzzy_process

# Import các hàm tiện ích từ file utils.py trong cùng thư mục app
from .utils import is_checkbox_ticked

# ==============================================================================
# === BƯỚC 1: CÁC HÀM TIỀN XỬ LÝ (PRE-PROCESSING) ===
# ==============================================================================

def _remove_horizontal_lines(roi_image):
    """
    Hàm chuyên dụng để phát hiện và loại bỏ các đường kẻ/chấm nằm ngang.
    Sử dụng kỹ thuật biến đổi hình thái học và inpainting.
    """
    # 1. Chuyển ảnh sang ảnh xám
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Nhị phân hóa ảnh để các nét chữ/đường kẻ nổi bật
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    # 3. Tạo kernel (bộ lọc) hình chữ nhật nằm ngang để phát hiện đường kẻ
    h, w = roi_image.shape[:2]
    # Chiều dài kernel linh hoạt theo chiều rộng ảnh ROI
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 15), 1))

    # 4. Áp dụng biến đổi hình thái học để chỉ giữ lại các cấu trúc ngang
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 5. Làm dày các đường kẻ đã phát hiện trên một "mặt nạ" (mask)
    cnts, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), 3) # Vẽ với độ dày 3

    # 6. Sử dụng kỹ thuật Inpainting để "chữa lành" ảnh tại các vị trí có đường kẻ
    inpainted_image = cv2.inpaint(roi_image, mask, 7, cv2.INPAINT_NS)
    
    return inpainted_image

def _preprocess_roi_for_ocr(roi_image):
    """
    Áp dụng các bộ lọc cơ bản để làm rõ chữ sau khi đã xóa nhiễu.
    """
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    # Áp dụng một chút làm nét để tăng độ tương phản của các nét chữ
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
    
    match = re.search(r'(\d{1,2})[./\s]*(\d{1,2})[./\s]*(\d{4})', text)
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
    Thực thi pipeline trích xuất thông tin hoàn chỉnh, bao gồm:
    1. Làm sạch nhiễu đường kẻ.
    2. Logic OCR fallback thông minh (TrOCR -> VietOCR).
    3. Hậu xử lý kết quả.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
    # Định nghĩa các trường chữ viết tay cần áp dụng quy trình xử lý đặc biệt
    handwritten_fields = ['ho_ten', 'ngay_sinh', 'lop']
    
    if not roi_config:
        print("LỖI: Cấu hình ROI rỗng, không thể xử lý.")
        return {}

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
                # --- QUY TRÌNH XỬ LÝ VĂN BẢN ---
                
                # 1. LÀM SẠCH NHIỄU: Loại bỏ các dấu chấm và đường kẻ
                print(f"  - [Clean] Áp dụng bộ lọc xóa đường kẻ cho '{field_name}'...")
                cleaned_roi = _remove_horizontal_lines(roi_cv2)
                
                # 2. TIỀN XỬ LÝ: Áp dụng các bộ lọc cơ bản trên ảnh đã sạch
                preprocessed_roi = _preprocess_roi_for_ocr(cleaned_roi)
                roi_pil = Image.fromarray(cv2.cvtColor(preprocessed_roi, cv2.COLOR_BGR2RGB))
                
                recognized_text = ""
                
                # 3. NHẬN DẠNG OCR: Áp dụng logic fallback thông minh
                if field_name in handwritten_fields:
                    # 3.1. Thử mô hình TrOCR chuyên biệt cho chữ viết tay
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

                    # 3.2. Nếu mô hình 1 thất bại, thử mô hình TrOCR tổng quát
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

                # 3.3. Nếu không phải trường viết tay, hoặc tất cả TrOCR đều thất bại, dùng VietOCR
                if not recognized_text and ocr_engines.vietocr_engine:
                    try:
                        print(f"  - [OCR] Thử VietOCR (cuối cùng) cho '{field_name}'...")
                        recognized_text = ocr_engines.vietocr_engine.predict(roi_pil)
                    except Exception as e:
                        print(f"    -> Lỗi VietOCR: {e}")
                
                # 4. HẬU XỬ LÝ: Làm sạch và chuẩn hóa kết quả từ OCR
                processed_text = _post_process_text(field_name, recognized_text)
                final_results[field_name] = processed_text
                print(f"  - [Text] '{field_name}': '{processed_text}' (Raw OCR: '{recognized_text}')")

        except KeyError as e:
            print(f"LỖI: Trường '{field_name}' trong cấu hình ROI thiếu thông tin: {e}")
        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi xử lý trường '{field_name}': {e}")
            
    print("--- Pipeline Trích xuất Hoàn tất ---")
    return final_results