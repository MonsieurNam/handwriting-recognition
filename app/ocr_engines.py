# app/ocr_engines.py

import torch
# Sửa đổi import - chỉ cần AutoModelForCausalLM và AutoProcessor
from transformers import AutoModelForCausalLM, AutoProcessor

class OCREngines:
    """
    Quản lý tập hợp các engine OCR, khởi tạo một lần và tái sử dụng.
    Hiện tại tập trung vào Vintern-1B.
    """
    def __init__(self, use_gpu=True):
        self.device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
        # Hỗ trợ thêm cho Apple Silicon
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            self.device = 'mps'
            
        print(f"Sử dụng thiết bị: {self.device}")

        self.engines = {}

        print("\n=== BẮT ĐẦU TẢI CÁC ENGINE OCR ===")
        print("Lưu ý: Lần chạy đầu tiên sẽ mất thời gian để tải các model.")

        # Chỉ tải Vintern-1B
        self._initialize_vintern(
            key='vintern_1b', # Key để truy cập
            model_name="5CD-AI/Vintern-1B-v2" # Model VLM 1B chuyên cho Tiếng Việt
        )
        
        print("\n=== TẢI XONG CÁC ENGINE OCR ===")
        print(f"Các engine đã được tải thành công: {list(self.engines.keys())}")

    def _initialize_vintern(self, key, model_name):
        """Khởi tạo VLM Vintern-1B."""
        try:
            print(f"\n--- Đang khởi tạo VLM Vintern-1B: {model_name}...")
            
            # Vintern-1B yêu cầu trust_remote_code=True
            model_kwargs = {"trust_remote_code": True}
            
            # Tối ưu hóa: Sử dụng float16 cho GPU (cuda hoặc mps)
            if self.device != 'cpu':
                model_kwargs["torch_dtype"] = torch.float16
            
            # AutoModelForCausalLM là class phù hợp
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(self.device)
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            self.engines[key] = (model, processor)
            print(f"VLM '{key}' đã khởi tạo thành công.")
        except Exception as e:
            print(f"LỖI khi khởi tạo VLM '{key}': {e}")