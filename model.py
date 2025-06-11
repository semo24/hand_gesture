import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class NoEarlyStop90(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc >= 0.90:
            print(f"\nval_accuracy {val_acc:.4f} 이상 도달했지만 조기 종료 없이 계속 학습합니다.")

# 데이터 로드
X = np.load('/content/drive/MyDrive/IPN_Hand/landmark_data/wrist_x_npy.npy', allow_pickle=True)
y = np.load('/content/drive/MyDrive/IPN_Hand/landmark_data/wrist_y_npy.npy', allow_pickle=True)

X = np.array([np.array(seq, dtype=np.float32) for seq in X], dtype=np.float32)
y = np.array(y)

num_classes = np.max(y) + 1
y_cat = to_categorical(y, num_classes=num_classes)

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential([
    Masking(mask_value=0., input_shape=(X.shape[1], X.shape[2])),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 콜백 설정
checkpoint_path = '/content/drive/MyDrive/models/best_gesture_model.h5'
saved_model_path = '/content/drive/MyDrive/models/best__gesture_model_tf'

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',
                             save_best_only=True,
                             verbose=1)

no_early_stop_90 = NoEarlyStop90()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[reduce_lr, checkpoint, no_early_stop_90]
)

best_model = load_model(checkpoint_path)
print("체크포인트로 저장된 .keras 모델 불러오기 완료")

best_model.export(saved_model_path)
print(f"avedModel 포맷으로 '{saved_model_path}' 에 저장 완료")
loss, accuracy = loaded_tf_model.evaluate(X_val, y_val)
print(f"\n검증 정확도: {accuracy*100:.2f}%")
