# app/ocr_ensemble.py
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from .models.vit5_ocr import ViT5OCR
from .models.trocr_vn import TrOCRVNHandwriting
from .models.crnn_viet import CRNNVietOCR
from .validators import VietnameseFormValidator

class AdvancedOCREnsemble:
    """ENSEMBLE OCR với 3 models + confidence voting"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"🚀 OCR Ensemble khởi tạo trên: {self.device}")
        
        # Load 3 models
        self.vit5 = ViT5OCR(device=self.device)
        self.trocr = TrOCRVNHandwriting(device=self.device)
        self.crnn = CRNNVietOCR(device=self.device)
        
        # Validator
        self.validator = VietnameseFormValidator()
        
        # Field-specific weights
        self.weights = {
            'ho_ten': {'vit5': 0.5, 'trocr': 0.4, 'crnn': 0.1},
            'ngay_sinh': {'vit5': 0.3, 'trocr': 0.2, 'crnn': 0.5},
            'lop': {'vit5': 0.2, 'trocr': 0.3, 'crnn': 0.5},
            'default': {'vit5': 0.4, 'trocr': 0.4, 'crnn': 0.2}
        }
    
    def predict(self, image: np.ndarray, field_name: str) -> Tuple[str, float]:
        """
        Ensemble prediction với confidence score
        
        Returns:
            (final_text, confidence_score)
        """
        if image is None or image.size == 0:
            return "", 0.0
        
        # Get predictions từ 3 models
        predictions = {
            'vit5': self.vit5.predict_with_conf(image),
            'trocr': self.trocr.predict_with_conf(image),
            'crnn': self.crnn.predict_with_conf(image)
        }
        
        # Weighted voting
        final_text, confidence = self._weighted_voting(predictions, field_name)
        
        # Validate & correct
        validated_text = self.validator.validate(field_name, final_text)
        
        print(f"  🔍 [{field_name}] Raw: {final_text} → Validated: {validated_text} (conf: {confidence:.2f})")
        
        return validated_text, confidence
    
    def _weighted_voting(self, predictions: Dict, field_name: str) -> Tuple[str, float]:
        """Weighted voting dựa trên field type"""
        weights = self.weights.get(field_name, self.weights['default'])
        
        # Calculate weighted scores
        scores = {}
        for model, (text, conf) in predictions.items():
            if text.strip():
                weighted_conf = conf * weights[model]
                scores[text] = scores.get(text, 0) + weighted_conf
        
        if not scores:
            return "", 0.0
        
        # Best candidate
        best_text = max(scores, key=scores.get)
        final_conf = scores[best_text]
        
        return best_text, final_conf

# GLOBAL INSTANCE
ocr_ensemble = AdvancedOCREnsemble(use_gpu=True)