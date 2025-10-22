# app/processing.py

import cv2
import numpy as np
from PIL import Image
# Bỏ import ViTokenizer vì hậu xử lý tên sẽ làm khác đi
# from pyvi import ViTokenizer 
from .utils import is_checkbox_ticked

def _preprocess_roi_for_ocr(roi_image):
    """Hàm nội bộ: Áp dụng các bộ lọc để làm rõ chữ."""
    # Với TrOCR, đôi khi chỉ cần ảnh xám là đủ, hoặc không cần xử lý gì
    # Bạn có thể thử nghiệm trả về ảnh gốc hoặc chỉ chuyển sang ảnh xám
    # return roi_image 
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    # Thử thêm một chút làm nét
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def _correct_vietnamese_name(text):
    """Hàm nội bộ: Chuẩn hóa cách viết hoa cho tên Tiếng Việt."""
    if not text:
        return ""
    # TrOCR thường trả về kết quả tốt, chỉ cần viết hoa là đủ
    return ' '.join([word.capitalize() for word in text.split()])

def _post_process_text(field_name, text):
    """Áp dụng các quy tắc hậu xử lý dựa trên tên trường."""
    if field_name == 'ho_ten':
        return _correct_vietnamese_name(text)
    # Có thể thêm regex để chuẩn hóa ngày sinh, lớp...
    return text.strip()

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines):
    """
    Thực thi pipeline trích xuất thông tin hoàn chỉnh từ ảnh đã được căn chỉnh.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
    # --- THÊM MỚI: Định nghĩa các trường nên dùng TrOCR ---
    handwritten_fields = ['ho_ten', 'ngay_sinh', 'lop']
    
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
                preprocessed_roi = _preprocess_roi_for_ocr(roi_cv2)
                roi_pil = Image.fromarray(cv2.cvtColor(preprocessed_roi, cv2.COLOR_BGR2RGB))
                
                recognized_text = ""
                
                # --- THAY ĐỔI: Logic lựa chọn engine OCR ---
                # Ưu tiên sử dụng TrOCR cho các trường chữ viết tay đã định nghĩa
                if field_name in handwritten_fields and ocr_engines.trocr_model and ocr_engines.trocr_processor:
                    try:
                        print(f"  - (Sử dụng TrOCR cho trường '{field_name}')")
                        pixel_values = ocr_engines.trocr_processor(images=roi_pil, return_tensors="pt").pixel_values.to(ocr_engines.device)
                        generated_ids = ocr_engines.trocr_model.generate(pixel_values, max_length=50) # Tăng max_length
                        recognized_text = ocr_engines.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    except Exception as e:
                        print(f"LỖI OCR TrOCR tại trường '{field_name}': {e}. Thử lại với VietOCR...")
                
                # Nếu không phải trường viết tay, hoặc TrOCR lỗi, dùng VietOCR
                if not recognized_text and ocr_engines.vietocr_engine:
                    try:
                        print(f"  - (Sử dụng VietOCR cho trường '{field_name}')")
                        recognized_text = ocr_engines.vietocr_engine.predict(roi_pil)
                    except Exception as e:
                        print(f"LỖI OCR VietOCR tại trường '{field_name}': {e}")
                
                processed_text = _post_process_text(field_name, recognized_text)
                final_results[field_name] = processed_text
                print(f"  - [Text] '{field_name}': '{processed_text}' (Raw: '{recognized_text}')")

        except KeyError as e:
            print(f"LỖI: Trường '{field_name}' trong cấu hình ROI thiếu thông tin: {e}")
        except Exception as e:
            print(f"LỖI không xác định khi xử lý trường '{field_name}': {e}")
            
    print("--- Pipeline Trích xuất Hoàn tất ---")
    return final_results