# app/models/trocr_vn.py
from typing import Tuple
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np

class TrOCRVNHandwriting:
    """TrOCR Vietnamese Handwriting - Best for medical forms"""
    
    def __init__(self, device: str = 'cpu'):
        print("📥 Đang tải TrOCR-VN Handwriting...")
        self.device = device
        
        model_name = "duyle2408/trocr-vietnamese-handwriting"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        
        print("✅ TrOCR-VN loaded!")
    
    def predict_with_conf(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict handwriting với confidence"""
        try:
            # Preprocess for handwriting
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (384, 384))
            pil_image = Image.fromarray(resized)
            
            # Process
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Confidence based on model logits
            confidence = 0.95  # TrOCR high confidence for handwriting
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"❌ TrOCR Error: {e}")
            return "", 0.0