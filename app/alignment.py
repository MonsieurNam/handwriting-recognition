# app/alignment.py

import cv2
import numpy as np
from .utils import order_points # Import từ module utils trong cùng package

def _find_page_contour(image):
    """
    Hàm nội bộ: Tìm đường viền (contour) của toàn bộ trang giấy trong ảnh.
    Phương pháp này tập trung vào việc tìm đa giác 4 cạnh lớn nhất chiếm phần lớn diện tích ảnh.
    """
    # 1. Tiền xử lý: Chuyển sang ảnh xám, làm mờ và phát hiện cạnh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Tìm các đường viền
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None

    # 3. Sắp xếp các đường viền theo diện tích từ lớn đến nhỏ
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 4. Lặp qua các đường viền để tìm trang giấy
    for c in contours:
        # Bỏ qua các contour quá nhỏ để tránh nhiễu
        # Giả định trang giấy phải chiếm ít nhất 20% diện tích ảnh
        if cv2.contourArea(c) < (image.shape[0] * image.shape[1] * 0.2):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx  # Trả về contour 4 đỉnh lớn đầu tiên tìm được

    return None

def align_image(image_to_align, template_image):
    """
    Căn chỉnh ảnh đầu vào dựa trên ảnh mẫu bằng cách khớp các góc của trang giấy.
    Phiên bản này sửa lỗi cắt hình bằng cách sử dụng các góc cố định của ảnh mẫu làm đích.
    """
    print("Bắt đầu quá trình căn chỉnh ảnh (phiên bản sửa lỗi cắt hình)...")
    
    # --- THAY ĐỔI QUAN TRỌNG ---

    # 1. Chỉ tìm contour trên ảnh ĐẦU VÀO (ảnh bị méo)
    input_page_contour = _find_page_contour(image_to_align)

    if input_page_contour is None:
        print("CẢNH BÁO: Không tìm thấy viền trang giấy trong ảnh đầu vào. Bỏ qua căn chỉnh.")
        return image_to_align

    # Sắp xếp các điểm của contour đầu vào
    ordered_input_points = order_points(input_page_contour.reshape(4, 2))
    
    # 2. Xác định các điểm ĐÍCH một cách tĩnh từ kích thước của ảnh MẪU
    # Đây là "khuôn" hoàn hảo mà chúng ta muốn nắn ảnh đầu vào theo.
    h, w = template_image.shape[:2]
    # Các điểm đích là 4 góc của ảnh mẫu, đã được sắp xếp theo đúng thứ tự.
    ordered_template_points = np.array([
        [0, 0],         # Trên-trái
        [w - 1, 0],     # Trên-phải
        [w - 1, h - 1], # Dưới-phải
        [0, h - 1]      # Dưới-trái
    ], dtype="float32")
    
    # ----------------------------

    # Tính toán ma trận biến đổi phối cảnh từ các điểm đầu vào đến các điểm đích
    M = cv2.getPerspectiveTransform(ordered_input_points, ordered_template_points)
    
    # Áp dụng phép biến đổi. Kích thước đầu ra (w, h) giờ đây khớp chính xác với ảnh mẫu.
    aligned_image = cv2.warpPerspective(image_to_align, M, (w, h))

    print("Căn chỉnh ảnh hoàn tất.")
    return aligned_image