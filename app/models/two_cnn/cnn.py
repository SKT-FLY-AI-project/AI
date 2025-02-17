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

# cnn_model_250217_1.h5
# CNN + VGG16 융합 모델 
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# CNN 모델 (Feature Extractor 역할)
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),  # (150, 150, 32) # Parameters 896
    layers.MaxPooling2D((2, 2)),  # (75, 75, 32)

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),  # (75, 75, 64) # Parameters 18496
    layers.MaxPooling2D((2, 2)),  # (37, 37, 64)

    # VGG16 입력 크기 맞추기 위해 UpSampling + Padding 추가
    layers.UpSampling2D(size=(2, 2)),  # (74, 74, 64)
    layers.ZeroPadding2D(((1, 0), (1, 0))),  # (75, 75, 64)

    layers.Conv2D(3, (1, 1), activation='relu', padding='same')  # (75, 75, 3) # Parameters 195
]) # 총 CNN 파라미터 수: 19,971

# VGG16 로드 (Feature Extractor / 복잡한 패턴(모양, 사물, 구조 등) 인식 역할) # 입력: (75, 75, 3) → 출력: (2, 2, 512)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(75, 75, 3))  # 3채널 입력
base_model.trainable = False  # VGG16의 가중치는 고정

# 모델 연결
model = models.Sequential([
    cnn_model,     # CNN 모델 적용
    base_model,    # VGG16 Feature Extractor 연결
    layers.Flatten(), # 출력: (2,2,512) → (2048,) # 2D Feature Map을 1D 벡터로 변환 (MLP Fully Connected Layer에 입력하기 위해)
    layers.Dense(512, activation='relu'), # 출력: (2048,) → (512,) # 파라미터 수: 1,049,088
    layers.Dropout(0.5), # 출력: (512,) → (512,) # 과적합 방지를 위해 50%의 뉴런을 랜덤하게 제거!
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # 최종 분류층 # 출력: (512,) → (317,) # 317개의 클래스를 예측하는 Softmax 활성화 함수 적용 (즉, 각 클래스의 확률값을 출력) 
    # 파라미터 수: 162,621
])

# 모델 구조 출력
model.summary()






# # cnn_model_250216_3.h5 
# # 사전 학습된 모델(Pre-trained Models) VGG16 사용.
# from tensorflow.keras.applications import VGG16
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# base_model.trainable = False  # 사전 학습된 가중치 고정

# model = models.Sequential([
#     base_model,
#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(len(train_generator.class_indices), activation='softmax')
# ])


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
model.save('app/models/two_cnn/cnn_model_250217_1.h5')
# model.save('cnn_model_250217_1.h5')


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
# print("Class indices:", train_generator.class_indices)
