# app/alignment.py

import cv2
import numpy as np
from .utils import order_points

def _find_page_contour(image):
    """
    Hàm nội bộ: Tìm đường viền của trang giấy bằng phương pháp ngưỡng hóa tự động.
    Đây là phương pháp mạnh mẽ hơn, hoạt động tốt với nhiều điều kiện ánh sáng và nền khác nhau.
    """
    # 1. Tiền xử lý
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Tăng kích thước kernel để làm mờ tốt hơn

    # 2. Ngưỡng hóa tự động (Otsu's Method)
    # THRESH_BINARY_INV: Đảo ngược màu, biến trang giấy (sáng màu) thành màu trắng và nền (tối màu) thành màu đen.
    # THRESH_OTSU: Tự động tính toán giá trị ngưỡng tối ưu thay vì dùng một số cố định.
    # Đây là trái tim của phương pháp mới.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # (Tùy chọn) Lưu ảnh ngưỡng để debug
    # cv2.imwrite("Data_Output/debug_threshold.png", thresh)

    # 3. Tìm các đường viền trên ảnh đã được ngưỡng hóa
    # Chỉ tìm các đường viền ngoài cùng để giảm nhiễu
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  - Lỗi phụ: Không tìm thấy bất kỳ contour nào sau khi ngưỡng hóa.")
        return None

    # 4. Tìm đường viền lớn nhất
    # Giả định rằng đường viền có diện tích lớn nhất chính là trang giấy
    page_contour = max(contours, key=cv2.contourArea)

    # 5. Xấp xỉ đường viền lớn nhất thành một đa giác 4 đỉnh
    peri = cv2.arcLength(page_contour, True)
    approx = cv2.approxPolyDP(page_contour, 0.02 * peri, True)

    # 6. Kiểm tra xem có đúng 4 đỉnh không
    if len(approx) == 4:
        # (Tùy chọn) Vẽ contour tìm được để debug
        # debug_image = image.copy()
        # cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 5)
        # cv2.imwrite("Data_Output/debug_page_contour_found.png", debug_image)
        return approx
    else:
        print(f"  - Lỗi phụ: Contour lớn nhất tìm được không có 4 đỉnh (tìm thấy {len(approx)} đỉnh).")
        return None


def align_image(image_to_align, template_image):
    """
    Căn chỉnh ảnh đầu vào dựa trên ảnh mẫu bằng cách khớp các góc của trang giấy.
    Sử dụng thuật toán phát hiện trang giấy mạnh mẽ và giữ nguyên logic sửa lỗi cắt hình.
    """
    print("Bắt đầu quá trình căn chỉnh ảnh (phiên bản nâng cấp)...")

    # 1. Tìm contour 4 đỉnh của trang giấy trên ảnh ĐẦU VÀO
    input_page_contour = _find_page_contour(image_to_align)

    if input_page_contour is None:
        print("CẢNH BÁO: Không tìm thấy viền trang giấy trong ảnh đầu vào. Bỏ qua căn chỉnh.")
        return image_to_align

    # 2. Sắp xếp các điểm của contour đầu vào theo thứ tự chuẩn
    ordered_input_points = order_points(input_page_contour.reshape(4, 2))

    # 3. Xác định các điểm ĐÍCH một cách tĩnh từ kích thước của ảnh MẪU (để không bị cắt hình)
    h, w = template_image.shape[:2]
    ordered_template_points = np.array([
        [0, 0],         # Trên-trái
        [w - 1, 0],     # Trên-phải
        [w - 1, h - 1], # Dưới-phải
        [0, h - 1]      # Dưới-trái
    ], dtype="float32")

    # 4. Tính toán ma trận biến đổi và áp dụng
    M = cv2.getPerspectiveTransform(ordered_input_points, ordered_template_points)
    aligned_image = cv2.warpPerspective(image_to_align, M, (w, h))

    print("Căn chỉnh ảnh hoàn tất.")
    return aligned_image