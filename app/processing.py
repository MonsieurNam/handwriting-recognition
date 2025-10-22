# app/processing.py

import cv2
import numpy as np
from PIL import Image
import re
import os
# from thefuzz import fuzz # Không cần thiết nữa

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
    final_thresh = cv2.bitwise_not(final_thresh)
    return cv2.cvtColor(final_thresh, cv2.COLOR_GRAY2BGR)

# ==============================================================================
# === BƯỚC 2: HÀM ENSEMBLE "BỎ PHIẾU" (ĐÃ BỊ LOẠI BỎ) ===
# ==============================================================================

# def _get_best_result_from_votes(predictions):
#     """Không cần thiết nữa vì chỉ sử dụng 1 model."""
#     pass

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
# === BƯỚC 4: PIPELINE CHÍNH VỚI CHỈ VINTERN-1B ===
# ==============================================================================

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines):
    print("\n--- Bắt đầu Pipeline Trích xuất Thông tin ---")
    final_results = {}
    
    # Lấy engine Vintern-1B ra ngoài vòng lặp
    vintern_engine = ocr_engines.engines.get('vintern_1b')
    if not vintern_engine:
        print("LỖI: Không tìm thấy engine 'vintern_1b'. Dừng pipeline.")
        return {}
        
    model, processor = vintern_engine
    
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
                
                print(f"\n>>> Đang xử lý trường '{field_name}' với Vintern-1B...")

                # --- CHẠY DỰ ĐOÁN VỚI VINTERN-1B ---
                text = ""
                try:
                    # Tạo chat prompt cho tác vụ OCR
                    # Sử dụng tiếng Việt để prompt
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "Vui lòng đọc chính xác tất cả văn bản trong ảnh này."}
                            ]
                        }
                    ]
                    
                    # Chuẩn bị input
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=roi_pil, return_tensors="pt").to(ocr_engines.device)
                    
                    # Generate
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=256, # Tăng token cho các trường dài
                        do_sample=False,
                        num_beams=3
                    )
                    
                    # Decode
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    
                    # Hậu xử lý: Tách phần trả lời của assistant
                    # Output sẽ có dạng: "...<|im_start|>assistant\n[NỘI DUNG OCR]<|im_end|>"
                    parts = generated_text.split("<|im_start|>assistant\n")
                    if len(parts) > 1:
                        text = parts[-1].replace("<|im_end|>", "").strip()
                    else:
                        text = generated_text.strip() # Fallback

                    print(f"  - Vintern-1B Result: '{text}'")
                
                except Exception as e:
                    print(f"  - Lỗi Vintern-1B: {e}")

                # --- Áp dụng hậu xử lý ---
                processed_text = _post_process_text(field_name, text)
                final_results[field_name] = processed_text
                print(f"  => KẾT QUẢ CUỐI CÙNG cho '{field_name}': '{processed_text}'")

        except Exception as e:
            print(f"LỖI KHÔNG XÁC ĐỊNH khi xử lý trường '{field_name}': {e}")
            
    print("\n--- Pipeline Trích xuất Hoàn tất ---")
    return final_results