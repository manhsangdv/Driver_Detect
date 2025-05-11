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

# ==== CẤU HÌNH ====
AUDIO_DIR = 'data_audio'
LOG_FILE = 'violation_log.txt'
MODEL_DIR = 'models'
ALARM_THRESHOLD = 0.2
HEAD_TURN_THRESHOLD_LEFT = 40
HEAD_TURN_THRESHOLD_RIGHT = -40
PHONE_CONFIDENCE = 0.5
HEAD_TURN_DELAY = 2

ALERTS = {
    'buonngu': {
        'text': 'Cảnh báo! Tài xế đang buồn ngủ',
        'filename': 'buonngu.mp3',
        'log': 'Buồn ngủ'
    },
    'dienthoai': {
        'text': 'Cảnh báo! Tài xế đang sử dụng điện thoại',
        'filename': 'dienthoai.mp3',
        'log': 'Sử dụng điện thoại'
    },
    'quaydau': {
        'text': 'Cảnh báo! Tài xế đang quay đầu không tập trung',
        'filename': 'quaydau.mp3',
        'log': 'Quay đầu'
    }
}

# ==== HÀM TIỆN ÍCH ====
def create_audio_files():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for alert in ALERTS.values():
        path = os.path.join(AUDIO_DIR, alert['filename'])
        if not os.path.exists(path):
            tts = gTTS(text=alert['text'], lang='vi')
            tts.save(path)

def load_sounds():
    return {k: pygame.mixer.Sound(os.path.join(AUDIO_DIR, v['filename'])) for k, v in ALERTS.items()}

def log_violation(log_file, alert_type):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file.write(f"{timestamp} - {ALERTS[alert_type]['log']}\n")
    log_file.flush()

def play_alert(alert_type, sounds, state):
    sounds[alert_type].play()
    state['playing'] = True
    state['current'] = alert_type

def handle_alert(alert_type, detected, state, sounds, counts, prev, queue, log_file):
    if detected and not prev[alert_type]:
        if not state['playing']:
            play_alert(alert_type, sounds, state)
        elif state['current'] != alert_type and alert_type not in queue:
            queue.append(alert_type)
        counts[alert_type] += 1
        log_violation(log_file, alert_type)
        prev[alert_type] = True
    if not detected:
        prev[alert_type] = False

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pose(face_landmarks):
    if not face_landmarks:
        return None
    left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y, face_landmarks.landmark[33].z])
    right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y, face_landmarks.landmark[263].z])
    nose = np.array([face_landmarks.landmark[4].x, face_landmarks.landmark[4].y, face_landmarks.landmark[4].z])
    eye_center = (left_eye + right_eye) / 2
    eye_direction = right_eye - left_eye
    face_normal_approx = np.cross(eye_direction, nose - eye_center)
    face_normal_approx = face_normal_approx / np.linalg.norm(face_normal_approx)
    yaw_angle_rad = np.arctan2(face_normal_approx[0], face_normal_approx[2])
    return np.degrees(yaw_angle_rad)

# ==== KHỞI TẠO ====
pygame.mixer.init()
create_audio_files()
sounds = load_sounds()
counts = {k: 0 for k in ALERTS}
prev = {k: False for k in ALERTS}
queue = []
state = {'playing': False, 'current': None}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path) if os.path.exists(predictor_path) else None

yolo_path = os.path.join(MODEL_DIR, 'yolov8l.pt')
yolo_model = YOLO(yolo_path, verbose=False) if os.path.exists(yolo_path) else None

left_looking_start_time = 0
right_looking_start_time = 0
current_looking_direction = None

cap = cv2.VideoCapture(0)
with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face_mesh = face_mesh.process(rgb_frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = {k: False for k in ALERTS}

        # YOLO: Phát hiện điện thoại
        if yolo_model:
            results_yolo = yolo_model.predict(frame, verbose=False)[0]
            for result in results_yolo.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if results_yolo.names[int(class_id)] == 'cell phone' and score > PHONE_CONFIDENCE:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                    cv2.putText(frame, "Phone Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    detected['dienthoai'] = True

        # MediaPipe: Hướng nhìn
        if results_face_mesh.multi_face_landmarks:
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
                yaw_angle = get_head_pose(face_landmarks)
                if yaw_angle is not None:
                    current_time = time.time()
                    if yaw_angle > HEAD_TURN_THRESHOLD_LEFT:
                        cv2.putText(frame, "Looking Left", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        if current_looking_direction != 'left':
                            left_looking_start_time = current_time
                            current_looking_direction = 'left'
                        elif current_time - left_looking_start_time >= HEAD_TURN_DELAY:
                            detected['quaydau'] = True
                    elif yaw_angle < HEAD_TURN_THRESHOLD_RIGHT:
                        cv2.putText(frame, "Looking Right", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        if current_looking_direction != 'right':
                            right_looking_start_time = current_time
                            current_looking_direction = 'right'
                        elif current_time - right_looking_start_time >= HEAD_TURN_DELAY:
                            detected['quaydau'] = True
                    else:
                        current_looking_direction = None

        # Dlib: EAR buồn ngủ
        if predictor:
            faces_dlib = detector(gray)
            for face in faces_dlib:
                landmarks_dlib = predictor(gray, face)
                left_eye_points = [(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in range(36, 42)]
                right_eye_points = [(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in range(42, 48)]
                left_ear = eye_aspect_ratio(left_eye_points)
                right_ear = eye_aspect_ratio(right_eye_points)
                ear = (left_ear + right_ear) / 2.0
                if ear < ALARM_THRESHOLD:
                    cv2.putText(frame, "DROWSY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    detected['buonngu'] = True
                for (x, y) in left_eye_points + right_eye_points:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Xử lý cảnh báo
        for alert_type in ALERTS:
            handle_alert(alert_type, detected[alert_type], state, sounds, counts, prev, queue, log_file)

        # Kiểm tra nếu âm thanh cảnh báo hiện tại đã phát xong
        if state['playing'] and not pygame.mixer.get_busy():
            state['playing'] = False
            state['current'] = None
            if queue:
                next_alert = queue.pop(0)
                play_alert(next_alert, sounds, state)

        # Hiển thị thông tin vi phạm
        for idx, alert_type in enumerate(ALERTS):
            cv2.putText(frame, f"{ALERTS[alert_type]['log']}: {counts[alert_type]}", (10, 30 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if state['current']:
            cv2.putText(frame, f"ALERT: {state['current'].upper()}", (frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Driver Behavior Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()