# app/vintern_engine.py

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class VinternEngine:
    """
    Quản lý việc tải và sử dụng mô hình Vintern.
    """
    def __init__(self, model_name="5CD-AI/Vintern-1B-v3_5", use_gpu=True):
        self.device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialize_vintern()
        self.transform = self._build_transform()

    def _initialize_vintern(self):
        try:
            print(f"\n--- Đang khởi tạo Vintern: {self.model_name}...")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            print("Vintern đã khởi tạo thành công.")
        except Exception as e:
            print(f"LỖI khi khởi tạo Vintern: {e}")
            raise

    def _build_transform(self, input_size=448):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _prepare_image(self, roi_image_pil):
        pixel_values = self.transform(roi_image_pil).unsqueeze(0)
        return pixel_values.to(torch.bfloat16).to(self.device)

    def extract_text(self, roi_image_pil, question="Văn bản trong hình là gì?"):
        pixel_values = self._prepare_image(roi_image_pil)
        generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3)
        
        full_question = f"<image>\n{question}"
        response = self.model.chat(self.tokenizer, pixel_values, full_question, generation_config)
        return response.strip()

    def is_checkbox_checked(self, roi_image_pil):
        question = "Trong ảnh có ô checkbox được tích không? Trả lời 'CÓ' hoặc 'KHÔNG'."
        response = self.extract_text(roi_image_pil, question)
        # Phân tích câu trả lời của mô hình
        # Đây là một cách tiếp cận đơn giản, có thể cần cải tiến để xử lý các câu trả lời đa dạng hơn
        return "có" in response.lower()