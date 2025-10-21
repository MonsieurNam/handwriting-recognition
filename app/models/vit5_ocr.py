# app/models/vit5_ocr.py
import torch
from transformers import ViTImageProcessor, ViTModel, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import numpy as np

class ViT5OCR:
    """ViT5 Vietnamese OCR - Best for structured forms"""
    
    def __init__(self, device: str = 'cpu'):
        print("📥 Đang tải ViT5-OCR...")
        self.device = device
        
        # Load models
        self.processor = ViTImageProcessor.from_pretrained("VietAI/vit5-base-vietnamese")
        self.vision_model = ViTModel.from_pretrained("VietAI/vit5-base-vietnamese").to(device)
        self.text_model = GPT2LMHeadModel.from_pretrained("VietAI/vit5-base-vietnamese").to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("VietAI/vit5-base-vietnamese")
        
        print("✅ ViT5-OCR loaded!")
    
    def predict_with_conf(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict với confidence score"""
        try:
            # PIL conversion
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Vision encoding
                vision_outputs = self.vision_model(**inputs)
                image_features = vision_outputs.last_hidden_state
                
                # Text generation
                outputs = self.text_model.generate(
                    image_features, 
                    max_length=50,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple confidence (length-based)
            confidence = min(len(text.strip()) / 20.0, 1.0)  # Normalize
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"❌ ViT5 Error: {e}")
            return "", 0.0
        