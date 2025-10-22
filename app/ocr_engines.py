# app/ocr_engines.py

import torch
import easyocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from transformers import VisionEncoderDecoderModel, AutoProcessor

class OCREngines:
    def __init__(self, use_gpu=True):
        self.device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")
        
        # Giữ lại VietOCR và EasyOCR làm phương án dự phòng
        self.vietocr_engine = self._initialize_vietocr()
        self.easyocr_engine = self._initialize_easyocr()
        
        # <<< THAY ĐỔI: Khởi tạo và lưu trữ 2 mô hình TrOCR riêng biệt >>>
        print("\n=== BẮT ĐẦU TẢI CÁC MÔ HÌNH TRANSFORMER OCR ===")
        print("Lưu ý: Lần chạy đầu tiên sẽ mất thời gian để tải model từ Hugging Face.")
        
        # Mô hình 1: Chuyên cho chữ viết tay
        self.trocr_handwritten_model, self.trocr_handwritten_processor = self._initialize_trocr(
            "nguyenvulebinh/trocr-base-vietnamese-handwritten"
        )
        
        # Mô hình 2: Một lựa chọn mạnh mẽ khác để dự phòng
        self.trocr_general_model, self.trocr_general_processor = self._initialize_trocr(
            "duyle2408/trocr-vietnamese-handwriting"
        )
        print("=== TẢI XONG CÁC MÔ HÌNH TRANSFORMER OCR ===")

    def _initialize_vietocr(self):
        print("\n--- Đang khởi tạo VietOCR Engine ---")
        try:
            config = Cfg.load_config_from_name('vgg_transformer')
            config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
            config['device'] = self.device
            config['predictor']['beamsearch'] = False
            engine = Predictor(config)
            print("VietOCR Engine đã khởi tạo thành công.")
            return engine
        except Exception as e:
            print(f"CẢNH BÁO: Không thể khởi tạo VietOCR: {e}")
            return None

    def _initialize_easyocr(self):
        # Giữ nguyên hàm này
        print("\n--- Đang khởi tạo EasyOCR Engine ---")
        try:
            engine = easyocr.Reader(['vi', 'en'], gpu=(self.device != 'cpu'))
            print("EasyOCR Engine đã khởi tạo thành công.")
            return engine
        except Exception as e:
            print(f"CẢNH BÁO: Không thể khởi tạo EasyOCR: {e}")
            return None
            
    def _initialize_trocr(self, model_name):
        # Hàm này đủ linh hoạt để tải bất kỳ mô hình nào từ Hugging Face
        print(f"\n--- Đang khởi tạo TrOCR Engine: {model_name} ---")
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            print(f"TrOCR '{model_name}' đã khởi tạo thành công.")
            return model, processor
        except Exception as e:
            print(f"LỖI khi khởi tạo TrOCR '{model_name}': {e}")
            return None, None