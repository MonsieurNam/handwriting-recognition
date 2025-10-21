```
|-- Data_Input/
|   |-- scan_01.jpg
|   `-- (các ảnh quét khác...)
|
|-- Data_Templates/
|   |-- template_form.jpg
|   `-- roi_template.json
|
|-- Data_Output/
|   `-- (kết quả sẽ được lưu ở đây)
|
|-- app/
|   |-- __init__.py
|   |-- alignment.py         # Chứa các hàm căn chỉnh ảnh
|   |-- ocr_engines.py       # Khởi tạo và quản lý các engine OCR
|   |-- processing.py        # Các hàm tiền xử lý, hậu xử lý và pipeline chính
|   |-- utils.py             # Các hàm phụ trợ (ví dụ: xử lý checkbox)
|   `-- config.py            # Tải và quản lý cấu hình
|
|-- main.py                  # Tệp chính để chạy toàn bộ pipeline
|-- requirements.txt         # Danh sách các thư viện cần thiết
`-- setup.sh                 # (Tùy chọn) Script cài đặt môi trường
```