import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 모델 로드
model_path = 'best_model_converted.keras'
model = load_model(model_path)
print("모델 로드드")

class_names = [
    "D0X - Non-gesture",
    "B0A - Pointing with one finger",
    "B0B - Pointing with two fingers",
    "G01 - Click with one finger",
    "G02 - Click with two fingers",
    "G03 - Throw up",
    "G04 - Throw down",
    "G05 - Throw left",
    "G06 - Throw right",
    "G07 - Open twice",
    "G08 - Double click with one finger",
    "G09 - Double click with two fingers",
    "G10 - Zoom in",
    "G11 - Zoom out"
]

# 볼륨 제어 설정
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# 전처리 준비비
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 시퀀스 형식식
SEQUENCE_LENGTH = 80
FEATURE_DIM = 64
FRAME_INTERVAL = 1

seq_buffer = []
frame_counter = 0
prev_index_pos = None
cooldown_time = 0

# 카메라 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라 실패")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    feature_vector = np.zeros((FEATURE_DIM,), dtype=np.float32)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = np.array([[pt.x, pt.y, pt.z] for pt in hand_landmarks.landmark])
        rel_lm = lm - lm[0]
        dist_index_wrist = np.linalg.norm(lm[8] - lm[0])
        fv = np.concatenate([rel_lm.flatten(), [dist_index_wrist]])
        if fv.shape[0] == 64:
            feature_vector = fv

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        seq_buffer.append(feature_vector)

        if len(seq_buffer) > SEQUENCE_LENGTH:
            seq_buffer.pop(0)

        if len(seq_buffer) == SEQUENCE_LENGTH and time.time() > cooldown_time:
            input_data = np.expand_dims(np.array(seq_buffer, dtype=np.float32), axis=0)
            pred_probs = model.predict(input_data, verbose=0)
            pred_class = np.argmax(pred_probs)-1
            gesture = class_names[pred_class]

            cv2.putText(frame, f'Gesture: {gesture} ({np.max(pred_probs):.2f})',
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 🎯 동작 추가
            if pred_class == 1:
                screen_w, screen_h = pyautogui.size()
                index_pos = lm[8][:2]
                pyautogui.moveTo(index_pos[0] * screen_w, index_pos[1] * screen_h)

            elif pred_class == 2:
                if prev_index_pos is not None:
                    dx = lm[8][0] - prev_index_pos[0]
                    dy = lm[8][1] - prev_index_pos[1]
                    if abs(dx) > abs(dy):
                        if dx > 0.02:
                            current = volume.GetMasterVolumeLevelScalar()
                            volume.SetMasterVolumeLevelScalar(min(current + 0.1, 1.0), None)
                        elif dx < -0.02:
                            current = volume.GetMasterVolumeLevelScalar()
                            volume.SetMasterVolumeLevelScalar(max(current - 0.1, 0.0), None)
                    else:
                        if dy > 0.02:
                            pyautogui.scroll(-200)
                        elif dy < -0.02:
                            pyautogui.scroll(200)
                prev_index_pos = lm[8][:2]

            elif pred_class == 5:
                pyautogui.hotkey('f11')
            elif pred_class == 6:
                pyautogui.hotkey('win', 'down')
            elif pred_class == 7:
                pyautogui.hotkey('win', 'left')
            elif pred_class == 8:
                pyautogui.hotkey('win', 'right')
            elif pred_class == 9:
                volume.SetMute(1, None)
            elif pred_class == 11:
                pyautogui.mouseDown()
                time.sleep(0.05)
                pyautogui.mouseUp()
            elif pred_class == 12:
                pyautogui.hotkey('ctrl', '+')
            elif pred_class == 13:
                pyautogui.hotkey('ctrl', '-')

            if np.max(pred_probs) >= 6:
                cooldown_time = time.time() + 2

    else:
        seq_buffer = []
        prev_index_pos = None
        cv2.putText(frame, 'No Hand Detected', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow('Real-time Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
