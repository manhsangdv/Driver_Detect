import cv2
import dlib
import mediapipe as mp
from scipy.spatial import distance
import pygame
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime
from gtts import gTTS

# Khởi tạo pygame mixer để phát âm thanh cảnh báo
pygame.mixer.init()

# Tạo thư mục cho file âm thanh nếu chưa tồn tại
os.makedirs('data_audio', exist_ok=True)

# Tạo các file âm thanh cảnh báo bằng gTTS
def create_audio_files():
    if not os.path.exists('data_audio/buonngu.mp3'):
        tts = gTTS(text="Cảnh báo! Tài xế đang buồn ngủ", lang='vi')
        tts.save('data_audio/buonngu.mp3')
    
    if not os.path.exists('data_audio/dienthoai.mp3'):
        tts = gTTS(text="Cảnh báo! Tài xế đang sử dụng điện thoại", lang='vi')
        tts.save('data_audio/dienthoai.mp3')
    
    if not os.path.exists('data_audio/quaydau.mp3'):
        tts = gTTS(text="Cảnh báo! Tài xế đang quay đầu không tập trung", lang='vi')
        tts.save('data_audio/quaydau.mp3')

create_audio_files()

# Tải các file âm thanh cảnh báo
sound_buonngu = pygame.mixer.Sound('data_audio/buonngu.mp3')
sound_dienthoai = pygame.mixer.Sound('data_audio/dienthoai.mp3')
sound_quaydau = pygame.mixer.Sound('data_audio/quaydau.mp3')

# Thiết lập trạng thái ban đầu cho hệ thống cảnh báo
alert_playing = False            # Có cảnh báo nào đang phát hay không
current_alert_type = None        # Loại cảnh báo đang phát hiện tại
alert_queue = []                 # Hàng đợi các cảnh báo chờ phát

# Biến đếm số lần vi phạm cho từng loại
count_buonngu = 0
count_dienthoai = 0
count_quaydau = 0

# Biến lưu trạng thái phát hiện trước đó
prev_buonngu = False
prev_dienthoai = False
prev_quaydau = False

# Mở file log để ghi lại các vi phạm
log_file = open('violation_log.txt', 'a', encoding='utf-8')

# Định nghĩa các tham số
ALARM_THRESHOLD = 0.2            # Ngưỡng EAR để phát hiện buồn ngủ
HEAD_TURN_THRESHOLD_LEFT = 40    # Ngưỡng quay đầu trái (độ)
HEAD_TURN_THRESHOLD_RIGHT = -40  # Ngưỡng quay đầu phải (độ)
PHONE_CONFIDENCE = 0.5           # Ngưỡng tin cậy cho nhận diện điện thoại
HEAD_TURN_DELAY = 2              # Độ trễ cảnh báo cho hướng nhìn (giây)

# Biến thời gian theo dõi hướng nhìn
left_looking_start_time = 0
right_looking_start_time = 0
current_looking_direction = None  # 'left', 'right', hoặc None

# Hàm tính Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Khởi tạo các model MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo các model Dlib
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join('models', 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path) if os.path.exists(predictor_path) else None

# Khởi tạo model YOLO
yolo_model = YOLO(os.path.join('models', 'yolov8l.pt'), verbose=False) if os.path.exists(os.path.join('models', 'yolov8l.pt')) else None

# Hàm tính góc xoay đầu từ vector pháp tuyến khuôn mặt
def get_head_pose(face_landmarks):
    if not face_landmarks:
        return None

    # Lấy các điểm mốc quan trọng để tính toán
    left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
    right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
    nose = np.array([face_landmarks.landmark[4].x, face_landmarks.landmark[4].y, face_landmarks.landmark[4].z])
    eye_center = (left_eye + right_eye) / 2

    # Tính góc yaw (quay đầu trái/phải)
    eye_direction = right_eye - left_eye
    face_normal_approx = np.cross(eye_direction, nose - eye_center)
    face_normal_approx = face_normal_approx / np.linalg.norm(face_normal_approx)
    yaw_angle_rad = np.arctan2(face_normal_approx[0], face_normal_approx[2])
    return np.degrees(yaw_angle_rad)

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh sang RGB cho MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face_mesh = face_mesh.process(rgb_frame)

    # Chuyển ảnh sang thang độ xám cho dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Biến để kiểm tra trạng thái
    buon_ngu_detected = False
    sd_dienthoai_detected = False
    quay_dau_detected = False

    # Phát hiện điện thoại bằng YOLOv8
    if yolo_model:
        results_yolo = yolo_model.predict(frame, verbose=False)[0]
        for result in results_yolo.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if results_yolo.names[int(class_id)] == 'cell phone' and score > PHONE_CONFIDENCE:
                # Vẽ khung xung quanh điện thoại
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(frame, "Phone Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                sd_dienthoai_detected = True

    # Xử lý phát hiện khuôn mặt và hướng nhìn bằng MediaPipe
    if results_face_mesh.multi_face_landmarks:
        for face_landmarks in results_face_mesh.multi_face_landmarks:            # Vẽ các điểm mốc khuôn mặt
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
            
            yaw_angle = get_head_pose(face_landmarks)

            if yaw_angle is not None:
                current_time = time.time()
                # Xử lý nhìn trái
                if yaw_angle > HEAD_TURN_THRESHOLD_LEFT:
                    cv2.putText(frame, "Looking Left", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if current_looking_direction != 'left':
                        left_looking_start_time = current_time
                        current_looking_direction = 'left'
                    elif current_time - left_looking_start_time >= HEAD_TURN_DELAY:
                        quay_dau_detected = True
                
                # Xử lý nhìn phải
                elif yaw_angle < HEAD_TURN_THRESHOLD_RIGHT:
                    cv2.putText(frame, "Looking Right", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if current_looking_direction != 'right':
                        right_looking_start_time = current_time
                        current_looking_direction = 'right'
                    elif current_time - right_looking_start_time >= HEAD_TURN_DELAY:
                        quay_dau_detected = True
                
                # Reset khi nhìn thẳng
                else:
                    current_looking_direction = None

    # Tính EAR từ Dlib landmarks (nếu phát hiện khuôn mặt và có predictor)
    if predictor:
        faces_dlib = detector(gray)
        for face in faces_dlib:
            landmarks_dlib = predictor(gray, face)

            # Chọn các điểm mắt và tính EAR
            left_eye_points = [(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in range(42, 48)]

            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)

            ear = (left_ear + right_ear) / 2.0

            # Phát hiện trạng thái buồn ngủ nếu EAR nhỏ hơn ngưỡng
            if ear < ALARM_THRESHOLD:
                cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                buon_ngu_detected = True

            # Vẽ các điểm mốc màu xanh lá cây trên mắt từ Dlib
            for (x, y) in left_eye_points:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye_points:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # --- Kiểm tra và xử lý cảnh báo cho mỗi hành vi vi phạm ---
    # Xử lý cảnh báo Buồn ngủ
    if buon_ngu_detected and not prev_buonngu:
        if not alert_playing:
            sound_buonngu.play()
            alert_playing = True
            current_alert_type = 'buonngu'
        else:
            if current_alert_type != 'buonngu' and 'buonngu' not in alert_queue:
                alert_queue.append('buonngu')
        
        count_buonngu += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Buồn ngủ\n")
        log_file.flush()
        prev_buonngu = True

    if not buon_ngu_detected:
        prev_buonngu = False

    # Xử lý cảnh báo Sử dụng điện thoại
    if sd_dienthoai_detected and not prev_dienthoai:
        if not alert_playing:
            sound_dienthoai.play()
            alert_playing = True
            current_alert_type = 'dienthoai'
        else:
            if current_alert_type != 'dienthoai' and 'dienthoai' not in alert_queue:
                alert_queue.append('dienthoai')
        
        count_dienthoai += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Sử dụng điện thoại\n")
        log_file.flush()
        prev_dienthoai = True

    if not sd_dienthoai_detected:
        prev_dienthoai = False

    # Xử lý cảnh báo Quay đầu mất tập trung
    if quay_dau_detected and not prev_quaydau:
        if not alert_playing:
            sound_quaydau.play()
            alert_playing = True
            current_alert_type = 'quaydau'
        else:
            if current_alert_type != 'quaydau' and 'quaydau' not in alert_queue:
                alert_queue.append('quaydau')
        
        count_quaydau += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Quay đầu\n")
        log_file.flush()
        prev_quaydau = True

    if not quay_dau_detected:
        prev_quaydau = False

    # Kiểm tra nếu âm thanh cảnh báo hiện tại đã phát xong
    if alert_playing:
        if not pygame.mixer.get_busy():
            alert_playing = False
            current_alert_type = None
            
            if len(alert_queue) > 0:
                next_alert = alert_queue.pop(0)
                if next_alert == 'buonngu':
                    sound_buonngu.play()
                elif next_alert == 'dienthoai':
                    sound_dienthoai.play()
                elif next_alert == 'quaydau':
                    sound_quaydau.play()
                
                alert_playing = True
                current_alert_type = next_alert

    # Hiển thị thông tin vi phạm lên khung hình
    cv2.putText(frame, f"Buon ngu: {count_buonngu}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Dien thoai: {count_dienthoai}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Quay dau: {count_quaydau}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hiển thị trạng thái cảnh báo hiện tại
    if current_alert_type:
        cv2.putText(frame, f"ALERT: {current_alert_type.upper()}", (frame.shape[1] - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hiển thị kết quả
    cv2.imshow('Driver Behavior Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
log_file.close()
