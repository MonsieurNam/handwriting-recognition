# app/processing.py

import cv2
import numpy as np
from PIL import Image
import re
import os
from thefuzz import fuzz # Import thư viện để so sánh độ tương đồng chuỗi

from .utils import is_checkbox_ticked
from .config import OUTPUT_PATH

# ==============================================================================
# === BƯỚC 1: HÀM TIỀN XỬ LÝ (GIỮ NGUYÊN PHIÊN BẢN TỐT NHẤT) ===
# ==============================================================================

def _advanced_preprocess_for_ocr(roi_image):
    """Pipeline tiền xử lý "Connect & Remove" để xóa nhiễu hạt và đường chấm."""
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    _, thresh_orig = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    w = roi_image.shape[1]
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
    closed_img = cv2.morphologyEx(thresh_orig, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
    line_mask = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, open_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)
    final_thresh = cv2.subtract(thresh_orig, dilated_line_mask)
    return cv2.cvtColor(final_thresh, cv2.COLOR_GRAY2BGR)

# ==============================================================================
# === BƯỚC 2: HÀM ENSEMBLE "BỎ PHIẾU" ĐỂ CHỌN KẾT QUẢ TỐT NHẤT ===
# ==============================================================================

def _get_best_result_from_votes(predictions):
    """
    Chọn ra kết quả tốt nhất từ một danh sách các dự đoán của nhiều model.
    Logic: Kết quả nào có độ tương đồng trung bình cao nhất với tất cả các kết quả khác
    sẽ được coi là kết quả "đồng thuận" và được chọn.
    """
    # Lọc bỏ các kết quả rỗng hoặc chỉ có khoảng trắng
    predictions = [p.strip() for p in predictions if p and p.strip()]

    if not predictions:
        return ""
    if len(predictions) == 1:
        return predictions[0]

    scores = {}
    for p1 in predictions:
        # Tính tổng điểm tương đồng của p1 với tất cả các dự đoán khác
        total_score = sum(fuzz.ratio(p1, p2) for p2 in predictions)
        # Lưu trữ điểm, nếu đã có thì cộng thêm để xử lý trường hợp các model ra kết quả y hệt
        scores[p1] = scores.get(p1, 0) + total_score
    
    # Trả về dự đoán có tổng điểm cao nhất
    return max(scores, key=scores.get)

# ==============================================================================
# === BƯỚC 3: CÁC HÀM HẬU XỬ LÝ (GIỮ NGUYÊN) ===
# ==============================================================================

def _post_process_text(field_name, text):
    # ... (Giữ nguyên các hàm hậu xử lý regex của bạn)
    raw_text = text
    processed_text = text.strip()
    if field_name == 'ngay_sinh': processed_text = re.sub(r'\D', '', processed_text) # Đơn giản hóa
    elif field_name == 'lop': processed_text = re.sub(r'[^A-Z0-9]', '', processed_text.upper()) # Đơn giản hóa
    elif field_name == 'ho_ten': processed_text = ' '.join([word.capitalize() for word in processed_text.split()])
    print(f"    - Hậu xử lý cho '{field_name}': '{raw_text}' -> '{processed_text}'")
    return processed_text
    
# ==============================================================================
# === BƯỚC 4: PIPELINE CHÍNH VỚI LOGIC ENSEMBLE ===
# ==============================================================================

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines):
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
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
                preprocessed_roi = _advanced_preprocess_for_ocr(roi_cv2)
                cv2.imwrite(os.path.join(OUTPUT_PATH, f"DEBUG_{field_name}_processed.png"), preprocessed_roi)
                roi_pil = Image.fromarray(preprocessed_roi)
                
                all_predictions = []

                # --- CHẠY DỰ ĐOÁN TRÊN TẤT CẢ CÁC ENGINE HIỆN CÓ ---
                print(f"\n>>> Đang xử lý trường '{field_name}' với tất cả các model...")

                # 1. VietOCR
                if 'vietocr' in ocr_engines.engines:
                    try:
                        text = ocr_engines.engines['vietocr'].predict(roi_pil)
                        all_predictions.append(text)
                        print(f"  - VietOCR Result: '{text}'")
                    except Exception as e:
                        print(f"  - Lỗi VietOCR: {e}")

                # 2. TrOCR Handwritten
                if 'trocr_handwritten' in ocr_engines.engines:
                    try:
                        model, processor = ocr_engines.engines['trocr_handwritten']
                        pixel_values = processor(images=roi_pil, return_tensors="pt").pixel_values.to(ocr_engines.device)
                        generated_ids = model.generate(pixel_values, max_length=64)
                        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        all_predictions.append(text)
                        print(f"  - TrOCR-Handwritten Result: '{text}'")
                    except Exception as e:
                        print(f"  - Lỗi TrOCR-Handwritten: {e}")

                # 3. TrOCR General
                if 'trocr_general' in ocr_engines.engines:
                    try:
                        model, processor = ocr_engines.engines['trocr_general']
                        pixel_values = processor(images=roi_pil, return_tensors="pt").pixel_values.to(ocr_engines.device)
                        generated_ids = model.generate(pixel_values, max_length=64)
                        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        all_predictions.append(text)
                        print(f"  - TrOCR-General Result: '{text}'")
                    except Exception as e:
                        print(f"  - Lỗi TrOCR-General: {e}")
                
                # --- TỔNG HỢP KẾT QUẢ TỐT NHẤT ---
                best_text = _get_best_result_from_votes(all_predictions)
                print(f"  -> Kết quả đồng thuận (Best Vote): '{best_text}'")

                # Áp dụng hậu xử lý trên kết quả tốt nhất
                processed_text = _post_process_text(field_name, best_text)
                final_results[field_name] = processed_text
                print(f"  => KẾT QUẢ CUỐI CÙNG cho '{field_name}': '{processed_text}' (Votes: {all_predictions})")

        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi xử lý trường '{field_name}': {e}")
            
    print("\n--- Pipeline Trích xuất Hoàn tất ---")
    return final_results