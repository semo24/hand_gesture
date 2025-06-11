import os
import cv2
import numpy as np
import mediapipe as mp
import time

# 경로 설정
annot_path = '/content/drive/MyDrive/IPN_Hand/annotations/Annot_TestList.txt'
frames_root = '/content/drive/MyDrive/IPN_Hand/frames_use/frames'
output_x = '/content/drive/MyDrive/IPN_Hand/landmark_data/wrist_x_npy.npy'
output_y = '/content/drive/MyDrive/IPN_Hand/landmark_data/wrist_y_npy.npy'
last_id_path = '/content/drive/MyDrive/IPN_Hand/landmark_data/last_id.txt'
SAVE_INTERVAL = 20
MAX_LEN = 80
FRAME_STEP = 5

os.makedirs(os.path.dirname(output_x), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

if os.path.exists(output_x) and os.path.exists(output_y):
    X_seq = list(np.load(output_x, allow_pickle=True))
    y_seq = list(np.load(output_y))
    print(f"📂 기존 데이터 불러옴: {len(X_seq)}개")
else:
    X_seq, y_seq = [], []
    print("🆕 새로 시작")

if os.path.exists(last_id_path):
    with open(last_id_path, 'r') as f:
        last_processed_id = f.read().strip()
    print(f"🔁 마지막 처리 ID: {last_processed_id}")
else:
    last_processed_id = None

with open(annot_path, 'r') as f:
    all_lines = sorted([line.strip() for line in f if line.strip()])

included, skipped = 0, 0
count_since_save = 0
start_processing = last_processed_id is None

print(f"총 어노테이션 수: {len(all_lines)}")

for idx, line in enumerate(all_lines):
    parts = line.split(',')
    if len(parts) != 6:
        continue

    video_id, _, gesture_index, start_f, end_f, _ = parts

    if not start_processing:
        if video_id == last_processed_id:
            start_processing = True
        continue

    gesture_index = int(gesture_index)
    start_f, end_f = int(start_f), int(end_f)
    frame_folder = os.path.join(frames_root, video_id)

    sequence = []

    for i in range(start_f, end_f + 1, FRAME_STEP):
        if len(sequence) >= MAX_LEN:
            break

        img_name = f'{video_id}_{i:06d}.jpg'
        img_path = os.path.join(frame_folder, img_name)

        if not os.path.exists(img_path):
            sequence.append(np.zeros((64,), dtype=np.float32))
            continue

        img = cv2.imread(img_path)
        if img is None:
            sequence.append(np.zeros((64,), dtype=np.float32))
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            landmarks = np.array([[pt.x, pt.y, pt.z] for pt in lm.landmark])
            rel_lm = landmarks - landmarks[0]
            dist_index_wrist = np.linalg.norm(landmarks[8] - landmarks[0])
            feature_vector = np.concatenate([rel_lm.flatten(), [dist_index_wrist]])
        else:
            feature_vector = np.zeros((64,), dtype=np.float32)

        sequence.append(feature_vector)

    if len(sequence) >= 5:
        if len(sequence) < MAX_LEN:
            pad = [np.zeros((64,), dtype=np.float32)] * (MAX_LEN - len(sequence))
            sequence += pad
        sequence = np.array(sequence)
        X_seq.append(sequence)
        y_seq.append(gesture_index)
        included += 1
        count_since_save += 1
        print(f"[{idx+1}/{len(all_lines)}] o 포함됨: {video_id} / 누적 포함 {included}")

        with open(last_id_path, 'w') as f:
            f.write(video_id)

        if count_since_save >= SAVE_INTERVAL:
            np.save(output_x, np.array(X_seq, dtype=object))
            np.save(output_y, np.array(y_seq))
            print(f"💾 {included}개 시퀀스 저장 완료")
            count_since_save = 0

    else:
        skipped += 1
        print(f"[{idx+1}/{len(all_lines)}] x 제외됨: {video_id} / {end_f-start_f+1}프레임 중 부족")

# 마지막 저장
np.save(output_x, np.array(X_seq, dtype=object))
np.save(output_y, np.array(y_seq))
print("\n********전처리 완료")
print(f"포함된 시퀀스: {included}개")
print(f"제외된 시퀀스: {skipped}개")
print(f"저장 위치: {output_x}, {output_y}")
