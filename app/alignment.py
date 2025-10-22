# app/alignment.py

import cv2
import numpy as np
from .utils import order_points # Import từ module utils trong cùng package

def _find_page_contour(image):
    """
    Hàm nội bộ: Tìm đường viền (contour) của toàn bộ trang giấy trong ảnh.
    Phương pháp này tập trung vào việc tìm đa giác 4 cạnh lớn nhất trong ảnh.
    """
    # Tạo một bản sao để vẽ debug mà không ảnh hưởng ảnh gốc
    debug_image = image.copy()

    # 1. Tiền xử lý: Chuyển sang ảnh xám, làm mờ và phát hiện cạnh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Phát hiện cạnh bằng thuật toán Canny
    edged = cv2.Canny(blurred, 75, 200)
    
    # Lưu ảnh đã phát hiện cạnh để debug (rất hữu ích)
    # cv2.imwrite("Data_Output/debug_edged.png", edged)

    # 2. Tìm các đường viền trong ảnh đã phát hiện cạnh
    # - cv2.RETR_EXTERNAL: Chỉ lấy các đường viền ngoài cùng.
    # - cv2.CHAIN_APPROX_SIMPLE: Nén các đoạn thẳng ngang/dọc/chéo để tiết kiệm điểm.
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu không tìm thấy contour nào, trả về None
    if len(contours) == 0:
        return None

    # 3. Sắp xếp các đường viền theo diện tích từ lớn đến nhỏ
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 4. Lặp qua các đường viền đã sắp xếp để tìm trang giấy
    for c in contours:
        # Xấp xỉ đường viền thành một đa giác
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Nếu đa giác xấp xỉ có 4 đỉnh, chúng ta giả định đó là trang giấy
        if len(approx) == 4:
            # Vẽ đường viền tìm được lên ảnh debug
            cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 3)
            # cv2.imwrite("Data_Output/debug_page_contour.png", debug_image)
            return approx  # Trả về contour 4 đỉnh tìm được

    # Nếu không có contour 4 đỉnh nào được tìm thấy, trả về None
    return None

def align_image(image_to_align, template_image):
    """
    Căn chỉnh ảnh đầu vào dựa trên ảnh mẫu bằng cách khớp các góc của trang giấy.

    Args:
        image_to_align (numpy.ndarray): Ảnh cần được căn chỉnh.
        template_image (numpy.ndarray): Ảnh mẫu chuẩn.

    Returns:
        numpy.ndarray: Ảnh đã được căn chỉnh, hoặc ảnh gốc nếu căn chỉnh thất bại.
    """
    print("Bắt đầu quá trình căn chỉnh ảnh bằng góc trang giấy...")
    
    # Tìm contour của trang giấy trên cả ảnh đầu vào và ảnh mẫu
    input_page_contour = _find_page_contour(image_to_align)
    template_page_contour = _find_page_contour(template_image)

    # Kiểm tra nếu không tìm thấy contour
    if input_page_contour is None:
        print("CẢNH BÁO: Không tìm thấy viền trang giấy trong ảnh đầu vào. Bỏ qua căn chỉnh.")
        return image_to_align
    if template_page_contour is None:
        print("CẢNH BÁO: Không tìm thấy viền trang giấy trong ảnh mẫu. Bỏ qua căn chỉnh.")
        return image_to_align

    # Sắp xếp các điểm của contour theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái
    ordered_input_points = order_points(input_page_contour.reshape(4, 2))
    ordered_template_points = order_points(template_page_contour.reshape(4, 2))
    
    # Tính toán ma trận biến đổi phối cảnh
    M = cv2.getPerspectiveTransform(ordered_input_points, ordered_template_points)
    
    # Áp dụng phép biến đổi để "duỗi" ảnh đầu vào cho khớp với kích thước ảnh mẫu
    h, w = template_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, M, (w, h))

    print("Căn chỉnh ảnh hoàn tất.")
    return aligned_image