# app/alignment.py

import cv2
import numpy as np
from .utils import order_points # Import từ module utils trong cùng package

def _find_main_content_frame(image):
    """
    Hàm nội bộ: Tìm khung viền hình chữ nhật lớn nhất trong ảnh, được giả định là khung nội dung chính.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_rect_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_rect_contour = approx
    
    return best_rect_contour

def align_image(image_to_align, template_image):
    """
    Căn chỉnh ảnh đầu vào dựa trên ảnh mẫu bằng cách khớp các khung nội dung chính của chúng.

    Args:
        image_to_align (numpy.ndarray): Ảnh cần được căn chỉnh.
        template_image (numpy.ndarray): Ảnh mẫu chuẩn.

    Returns:
        numpy.ndarray: Ảnh đã được căn chỉnh, hoặc ảnh gốc nếu căn chỉnh thất bại.
    """
    print("Bắt đầu quá trình căn chỉnh ảnh...")
    
    input_frame = _find_main_content_frame(image_to_align)
    template_frame = _find_main_content_frame(template_image)

    if input_frame is None:
        print("CẢNH BÁO: Không tìm thấy khung nội dung trong ảnh đầu vào. Bỏ qua căn chỉnh.")
        return image_to_align
    if template_frame is None:
        print("CẢNH BÁO: Không tìm thấy khung nội dung trong ảnh mẫu. Bỏ qua căn chỉnh.")
        return image_to_align

    ordered_input_points = order_points(input_frame.reshape(4, 2))
    ordered_template_points = order_points(template_frame.reshape(4, 2))
    
    # Tính toán ma trận biến đổi phối cảnh
    M = cv2.getPerspectiveTransform(ordered_input_points, ordered_template_points)
    
    # Áp dụng phép biến đổi
    h, w = template_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, M, (w, h))

    print("Căn chỉnh ảnh hoàn tất.")
    return aligned_image