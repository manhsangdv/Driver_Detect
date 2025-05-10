import cv2
import dlib
import mediapipe as mp
from scipy.spatial import distance
from pygame import mixer
import numpy as np
from ultralytics import YOLO
import os

# Khởi tạo âm thanh cảnh báo
mixer.init()
audio_path = os.path.join('data audio', 'alarm.wav')
mixer.music.load(audio_path)
is_alarm_playing = False  # Biến để kiểm soát trạng thái âm thanh

# Định nghĩa các tham số
ALARM_THRESHOLD = 0.2  # Ngưỡng EAR để phát hiện buồn ngủ
ALARM_DURATION = 5  # Thời gian cảnh báo (giây)
PHONE_CONFIDENCE = 0.5  # Ngưỡng tin cậy cho nhận diện điện thoại
HEAD_TURN_THRESHOLD = 30  # Ngưỡng góc quay đầu (độ)
# HEAD_PITCH_THRESHOLD = 10 # Ngưỡng góc cúi đầu (độ) # Bỏ ngưỡng cúi đầu

# Hàm tính Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Khởi tạo các model MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Khởi tạo các model Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join('models', 'shape_predictor_68_face_landmarks.dat'))
yolo_model = YOLO(os.path.join('models', 'yolov8l.pt'), verbose=False)  # Tải model YOLOv8

# Hàm tính góc xoay đầu từ vector pháp tuyến khuôn mặt (đã bỏ phần cúi đầu)
def get_head_pose(face_landmarks):
    if not face_landmarks:
        return None

    # Lấy các điểm mốc quan trọng
    left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
    right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
    nose = np.array([face_landmarks.landmark[4].x, face_landmarks.landmark[4].y, face_landmarks.landmark[4].z])
    eye_center = (left_eye + right_eye) / 2

    # Tính vector pháp tuyến gần đúng của khuôn mặt
    eye_direction = right_eye - left_eye
    face_normal_approx = np.cross(eye_direction, nose - eye_center)
    face_normal_approx = face_normal_approx / np.linalg.norm(face_normal_approx)

    # Tính góc xoay đầu (yaw)
    yaw_angle_rad = np.arctan2(face_normal_approx[0], face_normal_approx[2])
    yaw_angle_deg = np.degrees(yaw_angle_rad)

    return yaw_angle_deg

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

    # Phát hiện khuôn mặt bằng Dlib (để lấy landmarks cho EAR)
    faces_dlib = detector(gray)

    # Biến để kiểm tra trạng thái
    is_drowsy = False
    phone_detected = False
    looking_left = False
    looking_right = False
    # looking_down = False # Bỏ biến looking_down

    # Phát hiện điện thoại bằng YOLOv8
    results_yolo = yolo_model.predict(frame, verbose=False)[0]  # Thêm verbose=False để tắt thông tin debug
    for result in results_yolo.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if results_yolo.names[int(class_id)] == 'cell phone' and score > PHONE_CONFIDENCE:
            # Vẽ khung xung quanh điện thoại
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, "Phone Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            phone_detected = True

    # Xử lý phát hiện khuôn mặt và hướng nhìn bằng MediaPipe
    if results_face_mesh.multi_face_landmarks:
        for face_landmarks in results_face_mesh.multi_face_landmarks:
            # Vẽ các điểm mốc khuôn mặt (tùy chọn)
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec)

            yaw_angle = get_head_pose(face_landmarks)

            if yaw_angle is not None:
                if yaw_angle > HEAD_TURN_THRESHOLD:
                    cv2.putText(frame, "Looking Left", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    looking_left = True
                elif yaw_angle < -HEAD_TURN_THRESHOLD:
                    cv2.putText(frame, "Looking Right", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    looking_right = True

                # Bỏ phần phát hiện cúi đầu
                # if pitch_angle > HEAD_PITCH_THRESHOLD:
                #     cv2.putText(frame, "Looking Down", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                #     looking_down = True

    # Tính EAR từ Dlib landmarks (nếu phát hiện khuôn mặt)
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
            cv2.putText(frame, "DROWSY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            is_drowsy = True

        # Vẽ các điểm mốc màu xanh lá cây trên mắt từ Dlib
        for (x, y) in left_eye_points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye_points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Xử lý cảnh báo âm thanh (đã bỏ looking_down)
    if is_drowsy or phone_detected or looking_left or looking_right:
        if not is_alarm_playing:
            mixer.music.play(-1)
            is_alarm_playing = True
    else:
        if is_alarm_playing:
            mixer.music.stop()
            is_alarm_playing = False

    # Hiển thị kết quả
    cv2.imshow('Driver Behavior Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()