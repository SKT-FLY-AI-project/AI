import cv2
import numpy as np
from collections import Counter
from PIL import Image

def resize_image(image_path, target_size=(512, 512)):
    """
    이미지를 OpenCV를 사용하여 지정된 크기로 리사이징
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def convert_to_grayscale(image_path):
    """
    이미지를 그레이스케일로 변환
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    return image

def extract_dominant_colors(image_path, num_colors=5):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    image = np.float32(image)

    k = num_colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centers = cv2.kmeans(image, k, None, criteria, 10, flags)

    dominant_colors = centers.astype(int).tolist()

    # ✅ 여기서 리스트를 문자열로 변환
    return [f"RGB({r}, {g}, {b})" for r, g, b in dominant_colors]

def detect_edges(image_path, threshold1=100, threshold2=200):
    """
    Canny 엣지 감지 적용
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    edges = cv2.Canny(image, threshold1, threshold2)
    return np.sum(edges) > 10000  # 엣지 감지가 강하게 나타나는지 여부 반환

