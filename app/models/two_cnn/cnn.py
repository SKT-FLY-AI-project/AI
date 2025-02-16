# python models/two_cnn/cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 경로 설정
train_dir = r"app\models\two_cnn\data\cnn_train_data"
validation_dir = r"app\models\two_cnn\data\cnn_validation_data"
# train_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_train_data"
# validation_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_validation_data"

# 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1.0/255, brightness_range=[0.8, 1.2])
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# CNN 모델 구축
# cnn_model_250216_3.h5 
# 사전 학습된 모델(Pre-trained Models) VGG16 사용.
from tensorflow.keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # 사전 학습된 가중치 고정

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])


# cnn_model_250216_2.h5
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),

#     layers.Flatten(),
#     layers.Dropout(0.5),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(len(train_generator.class_indices), activation='softmax')
# ])


# cnn_model_250216.h5
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),

#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(len(train_generator.class_indices), activation='softmax')
# ])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

# EarlyStopping 설정
early_stop = EarlyStopping(
    monitor='val_loss',      # val_loss가 개선되지 않을 때 학습을 중단
    patience=5,              # 5 epoch 동안 개선이 없으면 종료
    restore_best_weights=True # 가장 좋은 가중치를 복원
)

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30, # 20번 학습
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stop]   # EarlyStopping 추가
)


# 모델 저장
# model.save('app/moidels/two_cnn/cnn_model_250216_3.h5')
model.save('cnn_model_250216_3.h5')


import matplotlib.pyplot as plt

# 학습 결과 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()



# 클래스 이름 출력
print("Class indices:", train_generator.class_indices)
