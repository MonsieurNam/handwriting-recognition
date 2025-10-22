# app/vintern_engine.py

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import json
import re

class VinternEngine:
    """
    Quản lý việc tải và sử dụng mô hình Vintern để trích xuất dữ liệu có cấu trúc từ toàn bộ ảnh.
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
            # Cấu hình để tối ưu hóa việc sử dụng GPU
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

    def _prepare_image_tensor(self, image_pil):
        pixel_values = self.transform(image_pil).unsqueeze(0)
        return pixel_values.to(torch.bfloat16).to(self.device)

    def extract_text_fields_from_image(self, image_pil):
        """
        Trích xuất tất cả các trường văn bản từ toàn bộ ảnh và trả về dưới dạng dictionary.
        """
        # Câu lệnh prompt yêu cầu Vintern trả về kết quả dưới dạng JSON.
        # Điều này giúp việc phân tích cú pháp trở nên dễ dàng và đáng tin cậy hơn.
        prompt = """<image>\nMô tả hình ảnh một cách chi tiết trả về dưới dạng JSON"""
        pixel_values = self._prepare_image_tensor(image_pil)
        generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=3)

        response_text = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        print(f"  - Phản hồi thô từ Vintern:\n{response_text}")

        # Cố gắng trích xuất và phân tích cú pháp JSON từ phản hồi của mô hình
        try:
            # Tìm chuỗi JSON trong phản hồi (bao gồm cả trường hợp có markdown code block)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                print("  - Cảnh báo: Không tìm thấy chuỗi JSON hợp lệ trong phản hồi của Vintern.")
                return {}
        except json.JSONDecodeError:
            print(f"  - Lỗi: Không thể phân tích cú pháp JSON từ phản hồi của Vintern.")
            return {}