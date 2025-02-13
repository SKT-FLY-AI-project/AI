########################### SETP 1 : openCV #####################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 및 전처리
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# # 그림 영역 탐지 (Contour Detection)
def detect_painting_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 사각형 또는 근사한 사각형 찾기
    detected_regions = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # 사각형만 고려
            detected_regions.append(cv2.boundingRect(approx))
    
    # 가장 큰 사각형을 선택 (여러 후보가 있을 경우)
    if detected_regions:
        x, y, w, h = max(detected_regions, key=lambda r: r[2] * r[3])  # 영역 크기 기준
        return image[y:y+h, x:x+w]
    else:
        return image  # 탐지 실패 시 원본 반환


# 주요 객체 검출 (Edge Detection 사용)
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 주요 색상 추출
def extract_dominant_colors(image, k=5):
    image = image.reshape((-1, 3))
    image = np.float32(image)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    dominant_colors = palette[np.argsort(-counts)]
    return dominant_colors.astype(int)

# 결과 시각화
def display_results(image_path):
    image = load_and_preprocess_image(image_path)
    painting_region = detect_painting_region(image)
    edges = detect_edges(image)
    dominant_colors = extract_dominant_colors(painting_region)
    
    
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    plt.subplot(1, 4, 2)
    plt.imshow(painting_region)
    plt.title("Detected Painting Region")

    plt.subplot(1, 4, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    print("edges 결과 값 : ",edges)
    """
    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]]
    """
    
    plt.subplot(1, 4, 4)
    plt.imshow([dominant_colors / 255])
    plt.title("Dominant Colors (Painting Only)")
    print("dominant_colors 결과 값 : ",dominant_colors)
    """
    [[ 47 103 161]
     [ 98 154 200]
     [ 21  42  70]
     [174 198 189]
     [159 132  83]]
    """

    
    plt.show()