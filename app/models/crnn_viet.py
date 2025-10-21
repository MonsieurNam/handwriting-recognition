# app/models/crnn_viet.py
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2
import numpy as np

class CRNNVietOCR:
    """Fast CRNN baseline"""
    
    def __init__(self, device: str = 'cpu'):
        print("⚡ Đang tải CRNN VietOCR...")
        
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
        config['device'] = device
        config['predictor']['beamsearch'] = False
        
        self.predictor = Predictor(config)
        print("✅ CRNN VietOCR loaded!")
    
    def predict_with_conf(self, image: np.ndarray) -> Tuple[str, float]:
        """Fast prediction"""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text = self.predictor.predict(pil_image)
            
            # Confidence estimation
            confidence = min(len(text.strip()) / 15.0, 1.0)
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"❌ CRNN Error: {e}")
            return "", 0.0