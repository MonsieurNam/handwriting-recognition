# app/preprocessing.py
import cv2
import numpy as np
import math
from typing import Tuple, Optional
from .utils import detect_barcode_regions, compute_skew_angle, deskew_image

class AdvancedPreprocessing:
    """Pipeline tiền xử lý chuyên sâu cho form y khoa Việt Nam"""
    
    def __init__(self):
        # Config tối ưu cho form y khoa
        self.BARCODE_MIN_CONFIDENCE = 0.7
        self.DESKEW_THRESHOLD = 0.5  # degrees
        self.ADAPTIVE_BLOCK_SIZE = 11
        self.ADAPTIVE_C = 2
        
    def process_roi(self, roi_image: np.ndarray, field_name: str = "") -> np.ndarray:
        """
        Pipeline hoàn chỉnh cho 1 ROI
        
        Args:
            roi_image: Ảnh ROI BGR
            field_name: Tên trường để apply rule cụ thể
            
        Returns:
            Ảnh đã tiền xử lý sẵn sàng cho OCR
        """
        if roi_image is None or roi_image.size == 0:
            return roi_image
            
        steps = [
            self._remove_noise_and_barcode,
            self._adaptive_thresholding,
            self._morphological_cleanup,
            self._deskew_text,
            self._sharpen_for_handwriting,
            self._contrast_enhancement
        ]
        
        processed = roi_image.copy()
        for step in steps:
            processed = step(processed, field_name)
            
        return processed
    
    def _remove_noise_and_barcode(self, image: np.ndarray, field_name: str) -> np.ndarray:
        """BƯỚC 1: Loại barcode + noise"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect barcode regions
        barcode_mask = detect_barcode_regions(gray)
        
        # Mask barcode với white (255)
        gray[barcode_mask] = 255
        
        # Remove salt & pepper noise
        denoised = cv2.medianBlur(gray, 3)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _adaptive_thresholding(self, image: np.ndarray, field_name: str) -> np.ndarray:
        """BƯỚC 2: Ngưỡng thích ứng cho chữ viết tay"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold tối ưu cho form y khoa
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.ADAPTIVE_BLOCK_SIZE, self.ADAPTIVE_C
        )
        
        # Invert nếu background tối
        if np.mean(gray) < 128:
            thresh = cv2.bitwise_not(thresh)
            
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _morphological_cleanup(self, image: np.ndarray, field_name: str) -> np.ndarray:
        """BƯỚC 3: Morphological operations"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Kernel khác nhau cho text vs date
        if 'ngay' in field_name or 'thang' in field_name:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))  # Horizontal
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))  # Vertical
            
        # Close gaps
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small noise
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    def _deskew_text(self, image: np.ndarray, field_name: str) -> np.ndarray:
        """BƯỚC 4: Xoay thẳng chữ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = compute_skew_angle(gray)
        
        if abs(angle) > self.DESKEW_THRESHOLD:
            return deskew_image(gray, angle)
        return image
    
    def _sharpen_for_handwriting(self, image: np.ndarray, field_name: str) -> np.ndarray:
        """BƯỚC 5: Làm sắc nét chữ viết tay"""
        if 'ho_ten' in field_name:
            # Unsharp mask cho handwriting
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        else:
            # Simple sharpen kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
        return sharpened
    
    def _contrast_enhancement(self, image: np.ndarray, field_name: str) -> np.ndarray:
        """BƯỚC 6: Tăng contrast"""
        # CLAHE cho medical forms
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# GLOBAL INSTANCE
advanced_preprocessor = AdvancedPreprocessing()