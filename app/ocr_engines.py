# app/ocr_engines.py

import torch
import easyocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoProcessor

class OCREngines:
    def __init__(self, use_gpu=True):
        self.device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")
        
        self.vietocr_engine = self._initialize_vietocr()
        self.easyocr_engine = self._initialize_easyocr()
        self.vi_trocr_engine, self.vi_trocr_processor = self._initialize_trocr("duyle2408/trocr-vietnamese-handwriting")

    def _initialize_vietocr(self):
        print("\n--- Đang khởi tạo VietOCR Engine ---")
        try:
            config = Cfg.load_config_from_name('vgg_transformer')
            config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
            config['device'] = self.device
            config['predictor']['beamsearch'] = False # Tắt để nhanh hơn
            engine = Predictor(config)
            print("VietOCR Engine đã khởi tạo thành công.")
            return engine
        except Exception as e:
            print(f"LỖI khi khởi tạo VietOCR: {e}")
            return None

    def _initialize_easyocr(self):
        print("\n--- Đang khởi tạo EasyOCR Engine ---")
        try:
            engine = easyocr.Reader(['vi', 'en'], gpu=(self.device != 'cpu'))
            print("EasyOCR Engine đã khởi tạo thành công.")
            return engine
        except Exception as e:
            print(f"LỖI khi khởi tạo EasyOCR: {e}")
            return None
            
    def _initialize_trocr(self, model_name):
        print(f"\n--- Đang khởi tạo TrOCR Engine: {model_name} ---")
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            print(f"TrOCR '{model_name}' đã khởi tạo thành công.")
            return model, processor
        except Exception as e:
            print(f"LỖI khi khởi tạo TrOCR '{model_name}': {e}")
            return None, None

# Tạo một instance duy nhất để sử dụng trong toàn bộ ứng dụng (Singleton pattern)
# ocr_models = OCREngines(use_gpu=True)