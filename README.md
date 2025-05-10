# Driver Behavior Detection System

Hệ thống phát hiện hành vi của tài xế sử dụng Computer Vision, có khả năng phát hiện:
- Tình trạng buồn ngủ thông qua theo dõi mắt (Eye Aspect Ratio)
- Phát hiện sử dụng điện thoại
- Phát hiện hướng nhìn của tài xế (trái/phải)

## Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Webcam
- GPU (không bắt buộc nhưng khuyến khích để tăng hiệu suất)

## Cài Đặt

1. Clone repository này:
```bash
git clone https://github.com/YOUR_USERNAME/Driver_Detect.git
cd Driver_Detect
```

2. Tạo và kích hoạt môi trường ảo (Virtual Environment):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu Trúc Thư Mục

```
Driver_Detect/
│
├── test.py                 # File chính chạy chương trình
├── requirements.txt        # File chứa danh sách thư viện cần thiết
│
├── data audio/
│   └── alarm.wav          # File âm thanh cảnh báo
│
└── models/
    ├── shape_predictor_68_face_landmarks.dat    # Model Dlib landmark detection
    └── yolov8l.pt                              # Model YOLOv8 large
```

## Cách Sử Dụng

1. Đảm bảo webcam đã được kết nối
2. Kích hoạt môi trường ảo (nếu chưa):
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Chạy chương trình:
```bash
python test.py
```

4. Sử dụng:
   - Nhấn 'q' để thoát chương trình
   - Hệ thống sẽ tự động phát cảnh báo khi:
     * Phát hiện tài xế buồn ngủ
     * Phát hiện sử dụng điện thoại
     * Phát hiện tài xế nhìn trái/phải quá lâu

## Ghi Chú

- File `shape_predictor_68_face_landmarks.dat` và `yolov8l.pt` là các model đã được train sẵn
- Có thể điều chỉnh các ngưỡng cảnh báo trong file `test.py`:
  * `ALARM_THRESHOLD`: Ngưỡng EAR để phát hiện buồn ngủ
  * `HEAD_TURN_THRESHOLD`: Ngưỡng góc quay đầu
  * `PHONE_CONFIDENCE`: Ngưỡng tin cậy cho phát hiện điện thoại

## License

[MIT License](LICENSE)

## Liên Hệ

Nếu bạn có bất kỳ câu hỏi hoặc góp ý nào, vui lòng tạo issue trong repository này.
