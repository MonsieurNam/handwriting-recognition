# app/processing.py

from PIL import Image
import re
import cv2

def _post_process_text(field_name, text):
    """Hậu xử lý văn bản sau khi trích xuất."""
    raw_text = text
    processed_text = text.strip()
    if field_name == 'ngay_sinh':
        processed_text = re.sub(r'\D', '', processed_text)
    elif field_name == 'lop':
        processed_text = re.sub(r'[^A-Z0-9]', '', processed_text.upper())
    elif field_name == 'ho_ten':
        processed_text = ' '.join([word.capitalize() for word in processed_text.split()])
    print(f"    - Hậu xử lý cho '{field_name}': '{raw_text}' -> '{processed_text}'")
    return processed_text

def run_vintern_pipeline(aligned_image, roi_config, vintern_engine):
    """
    Chạy pipeline trích xuất thông tin bằng Vintern.
    """
    print("\n--- Bắt đầu Pipeline Trích xuất với Vintern ---")
    final_results = {}

    for field_name, data in roi_config.items():
        try:
            field_type = data.get('type', 'text')
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            roi_cv2 = aligned_image[y:y+h, x:x+w]

            if roi_cv2.size == 0:
                print(f"  - Cảnh báo: Vùng ROI cho '{field_name}' bị rỗng, bỏ qua.")
                continue

            # Chuyển đổi ảnh ROI từ OpenCV (BGR) sang PIL (RGB)
            roi_pil = Image.fromarray(cv2.cvtColor(roi_cv2, cv2.COLOR_BGR2RGB))

            if field_type == 'checkbox':
                print(f"  - [Checkbox] Đang xử lý trường '{field_name}'...")
                is_checked = vintern_engine.is_checkbox_checked(roi_pil)
                final_results[field_name] = is_checked
                print(f"    - Kết quả từ Vintern: {is_checked}")
            else: # Mặc định là 'text'
                print(f"  - [Text] Đang xử lý trường '{field_name}'...")
                # Đặt câu hỏi cụ thể hơn cho từng trường nếu cần
                # question = data.get('question', 'Văn bản trong hình là gì?')
                extracted_text = vintern_engine.extract_text(roi_pil)
                print(f"    - Văn bản gốc từ Vintern: '{extracted_text}'")
                
                # Hậu xử lý
                processed_text = _post_process_text(field_name, extracted_text)
                final_results[field_name] = processed_text

        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi xử lý trường '{field_name}': {e}")

    print("\n--- Pipeline Trích xuất Hoàn tất ---")
    return final_results