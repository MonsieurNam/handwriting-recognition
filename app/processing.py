# app/processing.py - UPDATED
from .ocr_ensemble import ocr_ensemble
from .preprocessing import advanced_preprocessor

def run_ocr_pipeline(aligned_image, roi_config, ocr_engines=None):
    """NEW PIPELINE với Ensemble"""
    print("\n🚀 === ADVANCED OCR PIPELINE ===")
    final_results = {}
    
    for field_name, data in roi_config.items():
        try:
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            roi_cv2 = aligned_image[y:y+h, x:x+w]
            
            if roi_cv2.size == 0:
                continue
            
            # 1. ADVANCED PREPROCESS (Giai đoạn 1)
            preprocessed_roi = advanced_preprocessor.process_roi(roi_cv2, field_name)
            
            # 2. ENSEMBLE OCR (Giai đoạn 2)
            if data.get('type') == 'checkbox':
                result = is_checkbox_ticked(preprocessed_roi)
                final_results[field_name] = result
                print(f"  ✅ [Checkbox] '{field_name}': {result}")
            else:
                text, confidence = ocr_ensemble.predict(preprocessed_roi, field_name)
                final_results[field_name] = text
                print(f"  🎯 [Text] '{field_name}': '{text}' (conf: {confidence:.2f})")
                
        except Exception as e:
            print(f"❌ Lỗi {field_name}: {e}")
            final_results[field_name] = ""
    
    return final_results