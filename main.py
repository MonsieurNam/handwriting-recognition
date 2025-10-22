# main.py

import cv2
import os
import json
from app.config import (
    INPUT_PATH, 
    TEMPLATE_IMAGE_PATH,
    OUTPUT_PATH,
    create_directories, 
    load_roi_config
)
from app.alignment import align_image
from app.vintern_engine import VinternEngine
# Thay đổi import để trỏ đến hàm pipeline mới
from app.processing import run_combined_pipeline

def main():
    """
    Hàm chính điều phối toàn bộ quy trình:
    1. Thiết lập môi trường.
    2. Tải cấu hình và model Vintern.
    3. Xử lý từng ảnh trong thư mục đầu vào.
    4. Lưu kết quả.
    """
    print("=============================================")
    print("=== BẮT ĐẦU CHƯƠNG TRÌNH TRÍCH XUẤT KẾT HỢP ===")
    print("=============================================")

    # Bước 1: Thiết lập môi trường
    create_directories()
    
    # Bước 2: Tải cấu hình và khởi tạo Vintern Engine
    roi_config = load_roi_config()
    if not roi_config:
        print("Dừng chương trình do không tải được cấu hình ROI.")
        return

    vintern_engine = VinternEngine(use_gpu=True)
    
    template_image = cv2.imread(TEMPLATE_IMAGE_PATH)
    if template_image is None:
        print(f"LỖI: Không thể đọc ảnh mẫu tại: {TEMPLATE_IMAGE_PATH}. Dừng chương trình.")
        return

    # Bước 3: Xử lý từng ảnh trong thư mục đầu vào
    for image_name in os.listdir(INPUT_PATH):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(INPUT_PATH, image_name)
            print(f"\n>>> Đang xử lý ảnh: {image_name} <<<")
            
            input_image = cv2.imread(image_path)
            if input_image is None:
                print(f"  - Cảnh báo: Bỏ qua file không thể đọc: {image_name}")
                continue

            aligned_image = align_image(input_image, template_image)
            
            # Chạy pipeline trích xuất kết hợp
            extracted_data = run_combined_pipeline(aligned_image, roi_config, vintern_engine)
            
            # Bước 4: Lưu kết quả
            output_filename = f"{os.path.splitext(image_name)[0]}_result.json"
            output_path = os.path.join(OUTPUT_PATH, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=4)
                
            print(f"  - Kết quả đã được lưu tại: {output_path}")

    print("\n=============================================")
    print("=== HOÀN TẤT XỬ LÝ TẤT CẢ CÁC ẢNH ===")
    print("=============================================")

if __name__ == "__main__":
    main()