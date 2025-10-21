# app/utils.py

import cv2
import numpy as np

def order_points(pts):
    """Sắp xếp 4 điểm của một hình chữ nhật theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def is_checkbox_ticked(roi_img, threshold_ratio=0.03):
    """
    Kiểm tra xem một checkbox có được đánh dấu hay không bằng cách phân tích contour.
    
    Args:
        roi_img (numpy.ndarray): Ảnh của vùng checkbox.
        threshold_ratio (float): Tỷ lệ diện tích tối thiểu của contour so với ROI
                                 để được coi là một dấu tick.

    Returns:
        bool: True nếu được đánh dấu, False nếu không.
    """
    if roi_img is None or roi_img.size == 0:
        return False

    # Tiền xử lý ảnh ROI
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    # Ngưỡng nhị phân hóa ngược để các dấu tick (thường tối màu) trở thành màu trắng
    thresh = cv2.threshold(gray_roi, 170, 255, cv2.THRESH_BINARY_INV)[1]

    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False

    # Tính diện tích tối thiểu cần thiết
    min_contour_area = roi_img.shape[0] * roi_img.shape[1] * threshold_ratio

    # Kiểm tra xem có contour nào đủ lớn không
    for c in contours:
        if cv2.contourArea(c) > min_contour_area:
            return True
            
    return False