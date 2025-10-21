# app/validators.py
import re
from datetime import datetime
from typing import Dict, Optional

class VietnameseFormValidator:
    """Domain-specific validation cho form y khoa"""
    
    def __init__(self):
        self.vietnamese_names = [
            "Thị Hồng Nhung", "Lê Thị Hồng", "Nguyễn Văn", "Trần Thị"
        ]  # Load from file in production
        
    def validate(self, field_name: str, text: str) -> str:
        """Validate theo field type"""
        validators = {
            'ho_ten': self._validate_name,
            'ngay_sinh': self._validate_date,
            'lop': self._validate_class,
            'ngay': self._validate_day,
            'thang': self._validate_month,
            'nam': self._validate_year
        }
        
        return validators.get(field_name, lambda x: x)(text)
    
    def _validate_name(self, text: str) -> str:
        """Remove duplicates + capitalize"""
        if not text:
            return ""
        
        # Remove repeated words
        words = re.split(r'\s+', text)
        unique_words = []
        [unique_words.append(word) for word in words if word not in unique_words]
        
        # Capitalize Vietnamese style
        cleaned = ' '.join(word.capitalize() for word in unique_words)
        
        # Match common patterns
        for name in self.vietnamese_names:
            if name.lower() in cleaned.lower():
                return name
        
        return cleaned[:50]  # Truncate
    
    def _validate_date(self, text: str) -> str:
        """Extract valid DD/MM/YYYY"""
        patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
            r'(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    day, month, year = map(int, match)
                    if year < 100: year += 2000
                    date = datetime(year, month, day)
                    return date.strftime('%d/%m/%Y')
                except:
                    continue
        return ""
    
    def _validate_class(self, text: str) -> str:
        """Validate class: 6A1, 10A2..."""
        match = re.search(r'(\d+)[A-K]', text)
        return match.group(0) if match else text[:4]
    
    def _validate_day(self, text: str) -> str:
        num = re.search(r'\d+', text)
        return num.group(0) if num else ""
    
    def _validate_month(self, text: str) -> str:
        num = re.search(r'\d+', text)
        month = int(num.group(0)) if num else 0
        return str(min(month, 12))
    
    def _validate_year(self, text: str) -> str:
        num = re.search(r'\d{4}|\d{2}', text)
        year = int(num.group(0)) if num else 0
        if year < 100: year += 2000
        return str(year)[-4:]