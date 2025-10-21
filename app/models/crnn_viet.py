# app/models/crnn_viet.py - ✅ ULTRA LIGHT - NO ERRORS
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2
import numpy as np
from typing import Tuple

class CRNNVietOCR:
    """3 VietOCR configs - Production ready"""
    
    def __init__(self, device: str = 'cpu', config: str = 'primary'):
        print(f"⚡ Loading VietOCR ({config})...")
        
        # BASE CONFIG
        base_config = Cfg.load_config_from_name('vgg_transformer')
        base_config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
        base_config['device'] = device
        base_config['predictor']['beamsearch'] = False  # FAST
        
        # CONFIG VARIANTS
        configs = {
            'primary': {
                'beamsearch': False,
                'width': 256,
                'height': 32
            },
            'handwriting': {
                'beamsearch': True,
                'beamwidth': 3,
                'width': 384,
                'height': 64
            },
            'fast': {
                'beamsearch': False,
                'width': 128,
                'height': 32
            }
        }
        
        variant = configs.get(config, configs['primary'])
        base_config.update(variant)
        
        self.predictor = Predictor(base_config)
        print(f"✅ VietOCR ({config}) READY!")
    
    def predict_with_conf(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict + confidence"""
        try:
            # Convert BGR -> GRAY -> PIL
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize optimal
            h, w = gray.shape
            target_h = 64 if 'handwriting' in str(self) else 32
            scale = target_h / h
            new_w = max(int(w * scale), 50)
            gray = cv2.resize(gray, (new_w, target_h))
            
            # Enhance contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            
            pil_image = Image.fromarray(gray)
            text = self.predictor.predict(pil_image)
            
            # Confidence score
            text_len = len(text.strip())
            conf = min(text_len / 15.0, 0.95) if text_len > 0 else 0.0
            
            return text.strip(), conf
            
        except Exception as e:
            print(f"❌ VietOCR Error: {e}")
            return "", 0.0