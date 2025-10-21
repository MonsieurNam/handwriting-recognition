# app/utils.py - UPDATED
import cv2
import numpy as np
import math
from typing import Tuple

def detect_barcode_regions(image: np.ndarray, min_confidence: float = 0.7) -> np.ndarray:
    """
    PHÁT HIỆN & MASK BARCODE/QR CODE
    Returns: Binary mask (1 = barcode region)
    """
    if image is None or len(image.shape) != 2:
        return np.zeros_like(image)
    
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. HORIZONTAL LINES (Barcode)
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=50,
        minLineLength=w//3, maxLineGap=5
    )
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 5:  # Horizontal line
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    
    # 2. DENSE VERTICAL LINES (QR Code)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
    
    # Count vertical lines
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:  # QR size
            cv2.fillPoly(mask, [cnt], 255)
    
    # Dilate để cover toàn bộ barcode
    kernel = np.ones((3, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Confidence filter
    barcode_area = np.sum(mask > 0)
    total_area = h * w
    confidence = barcode_area / total_area
    
    return mask if confidence > min_confidence else np.zeros_like(mask)

def compute_skew_angle(image: np.ndarray) -> float:
    """Tính góc nghiêng của text"""
    gray = image.copy()
    
    # Threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Get largest contour (text block)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    return angle * -1  # Negative for rotation

def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Xoay thẳng ảnh"""
    if abs(angle) < 0.5:
        return image
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate
    rotated = cv2.warpAffine(image, M, (w, h), 
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# Existing functions (order_points, is_checkbox_ticked) remain unchanged
def order_points(pts):
    """Sắp xếp 4 điểm của một hình chữ nhật"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def is_checkbox_ticked(roi_img, threshold_ratio=0.03):
    """Kiểm tra checkbox - UNCHANGED"""
    if roi_img is None or roi_img.size == 0:
        return False

    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_roi, 170, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False

    min_contour_area = roi_img.shape[0] * roi_img.shape[1] * threshold_ratio
    for c in contours:
        if cv2.contourArea(c) > min_contour_area:
            return True
    return False