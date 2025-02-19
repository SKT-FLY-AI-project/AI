import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# ✅ CNN 모델 경로 설정
MODEL_PATH = "app/ai_models/cnn_model_250217_1.h5"
model = None  # 모델을 처음에는 None으로 설정

# ✅ CNN 클래스 인덱스 로드
CLASS_INDICES = {
    0: "A Basket of Clams",
    1: "The Starry Night",
    2: "Mona Lisa",
    3: "The Persistence of Memory",
    4: "The Scream",
}  # 기존 cnn_test.py에서 class indices 가져옴

def load_cnn_model():
    """ CNN 모델을 로드 (필요 시 한 번만 실행) """
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(image_path, target_size=(150, 150)):
    """ 이미지를 CNN 입력 포맷으로 변환 """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # 0~1 범위로 정규화
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3) 형태로 변환
    return img_array

def predict_image(image_path):
    """ CNN을 이용해 이미지 예측 """
    model = load_cnn_model()
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)

    max_prob = np.max(predictions)
    predicted_class_idx = np.argmax(predictions)

    if max_prob < 0.7:  # 신뢰도가 낮으면 "Unknown"
        return {"title": "Unknown Title", "confidence": f"{max_prob*100:.2f}%"}

    class_name = CLASS_INDICES.get(predicted_class_idx, "Unknown Title")
    return {"title": class_name, "confidence": f"{max_prob*100:.2f}%"}