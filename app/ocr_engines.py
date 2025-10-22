# app/ocr_engines.py

import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from transformers import VisionEncoderDecoderModel, AutoProcessor
import easyocr

class OCREngines:
    """
    Quản lý tập hợp các engine OCR, khởi tạo một lần và tái sử dụng.
    """
    def __init__(self, use_gpu=True):
        self.device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")

        # Sử dụng một dictionary để quản lý tất cả các engine một cách linh hoạt
        self.engines = {}

        print("\n=== BẮT ĐẦU TẢI CÁC ENGINE OCR ===")
        print("Lưu ý: Lần chạy đầu tiên sẽ mất thời gian để tải các model.")

        self._initialize_vietocr()
        self._initialize_easyocr()
        
        # Sửa lại để tải 2 model TrOCR khác nhau và mạnh nhất
        self._initialize_trocr(
            key='trocr_handwritten', # Key để truy cập
            model_name="nguyenvulebinh/trocr-base-vietnamese-handwritten" # Model chuyên cho chữ viết tay
        )
        self._initialize_trocr(
            key='trocr_general', # Key để truy cập
            model_name="duyle2408/trocr-vietnamese-handwriting" # Model tổng quát hơn
        )
        
        print("\n=== TẢI XONG CÁC ENGINE OCR ===")
        print(f"Các engine đã được tải thành công: {list(self.engines.keys())}")

    def _initialize_vietocr(self):
        try:
            print("\n--- Đang khởi tạo VietOCR...")
            config = Cfg.load_config_from_name('vgg_transformer')
            config['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
            config['device'] = self.device
            config['predictor']['beamsearch'] = False
            engine = Predictor(config)
            self.engines['vietocr'] = engine
            print("VietOCR đã khởi tạo thành công.")
        except Exception as e:
            print(f"CẢNH BÁO: Không thể khởi tạo VietOCR: {e}")

    def _initialize_easyocr(self):
        try:
            print("\n--- Đang khởi tạo EasyOCR...")
            engine = easyocr.Reader(['vi', 'en'], gpu=(self.device != 'cpu'))
            self.engines['easyocr'] = engine
            print("EasyOCR đã khởi tạo thành công.")
        except Exception as e:
            print(f"CẢNH BÁO: Không thể khởi tạo EasyOCR: {e}")
            
    def _initialize_trocr(self, key, model_name):
        try:
            print(f"\n--- Đang khởi tạo TrOCR: {model_name}...")
            processor = AutoProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            # Lưu cả model và processor vào dictionary dưới dạng một tuple
            self.engines[key] = (model, processor)
            print(f"TrOCR '{key}' đã khởi tạo thành công.")
        except Exception as e:
            print(f"LỖI khi khởi tạo TrOCR '{key}': {e}")