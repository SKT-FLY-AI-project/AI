########################### SETP 1 : openCV #####################################

import os # colab용
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

# 이미지 로드 및 전처리
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상단
    rect[2] = pts[np.argmax(s)]  # 우하단
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상단
    rect[3] = pts[np.argmax(diff)]  # 좌하단
    return rect

# # 그림 영역 탐지 (Contour Detection)
def detect_painting_region(image, min_area_ratio=0.2, aspect_ratio_range=(0.75, 1.5)):      
    # aspect_ratio_range ✅ 가로/세로 비율 고려 → 너무 넓거나 좁은 탐지를 방지
    """
    그림이 너무 작은 영역으로 잘려 어두워지는 문제를 방지하기 위해,
    최소 면적 비율을 설정하여 너무 작은 영역은 무시하고 원본을 유지하도록 한다.
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(gray, 100, 200)
    edge = cv2.bitwise_or(thresh, edges)

    # Contour Detection 적용
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_regions = []
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # 사각형인지 확인 # +) 사각형뿐만 아니라 특정 크기 이상의 다각형도 포함하도록 하려면 # if 4 <= len(approx) <= 8:  # 4~8각형도 포함
            # x, y, w, h = cv2.boundingRect(approx)
            # aspect_ratio = w / float(h) # +) 사각형뿐만 아니라 특정 크기 이상의 다각형도 포함하도록 하려면 # x, y, w, h = cv2.boundingRect(approx) # 모든 다각형을 고려하지만, bounding box 기준으로 정사각형 형태로 변환
            
            # # 가로세로 비율이 일정 범위 내에 있는지 확인
            # if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            #     detected_regions.append((x, y, w, h))
            detected_regions.append(approx)
    
    if detected_regions:
        # # 가장 큰 사각형 영역 선택
        # x, y, w, h = max(detected_regions, key=lambda r: r[2] * r[3])
        
        # 가장 큰 영역 선택
        largest_region = max(detected_regions, key=cv2.contourArea)
        ordered_points = order_points(largest_region.reshape(4, 2))
                
        # 사각형 너비와 높이 계산
        (tl, tr, br, bl) = ordered_points
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        
        # 전체 이미지 대비 크롭된 영역이 너무 작으면 원본 반환
        img_area = image.shape[0] * image.shape[1]
        cropped_area = width * height
        if cropped_area < min_area_ratio * img_area:
            print("⚠️ 그림 영역이 너무 작아 원본 이미지를 반환합니다.")
            return image
        
        # 변환할 대상 좌표 설정
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        
        # 원근 변환 수행
        M = cv2.getPerspectiveTransform(ordered_points, dst)
        corrected = cv2.warpPerspective(image, M, (width, height))
        return corrected
    
    # 사각형이 없을 경우, 가장 큰 윤곽선이라도 반환
    # if not detected_regions:
    #     x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    #     return image[y:y+h, x:x+w]  # 가장 큰 윤곽선 영역만 반환
    print("⚠️ 사각형을 찾을 수 없습니다. 원본 이미지를 반환합니다.")
    return image


# 주요 객체 검출 (Edge Detection 사용)
def detect_edges(image):
    image = detect_painting_region(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Adaptive Thresholding 적용하여 대비 강화 
    # # Canny(100, 200) 만으로도 대부분의 경우 잘 작동하지만, 배경과 작품의 명암 차이가 적은 경우 문제가 발생할 수 있다.
    # 이런 경우 Adaptive Thresholding을 추가하면 작품의 영역을 더 명확하게 구분할 수 있다.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Canny Edge Detection 적용
    edges = cv2.Canny(gray, 100, 200)
    # Thresholding과 Edge Detection 결합
    combined_edges = cv2.bitwise_or(thresh, edges)
    return combined_edges

# 주요 색상 추출
def extract_dominant_colors(image, k=5):
    image = image.reshape((-1, 3))
    image = np.float32(image)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    dominant_colors = palette[np.argsort(-counts)]
    return dominant_colors.astype(int)


# ✅ HSV 기반 색상 보정
def adjust_hsv_lightness_and_saturation(rgb, lightness_factor=1.4, saturation_factor=1.3):
    """
    HSV 색 공간에서 명도(Value)와 채도(Saturation)을 조정하여  
    사람이 인식하는 색감과 비슷하게 변환하는 함수.

    - `lightness_factor`: 명도(Value) 조정 강도
    - `saturation_factor`: 채도(Saturation) 조정 강도
    """
    # RGB → HSV 변환
    rgb_array = np.array([[rgb]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[0, 0]  # 단일 픽셀 값 추출

    # ✅ 명도(Value) 조정
    if v < 150:  # 기존보다 어두운 색상은 더 밝게
        v = min(v * lightness_factor, 255)
    elif v > 220:  # 너무 밝은 색상은 과하지 않게 보정
        v = min(v * 1.1, 255)

    # ✅ 채도(Saturation) 증가하여 원색 계열을 더 살림
    s = min(s * saturation_factor, 255)

    # ✅ 특정 색 계열(파란색, 노란색, 초록색 등)에 대한 추가 보정
    if 180 <= h <= 260:  # 파란색 계열
        v = min(v * 1.4, 255)
        s = min(s * 1.3, 255)
    elif 40 <= h <= 80:  # 노란색 계열
        v = min(v * 1.5, 255)
        s = min(s * 1.4, 255)
    elif 80 <= h <= 160:  # 초록색 계열
        v = min(v * 1.4, 255)
        s = min(s * 1.3, 255)

    # HSV → RGB 변환
    new_hsv = np.array([[[h, int(s), int(v)]]], dtype=np.uint8)
    new_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)[0, 0]

    return tuple(new_rgb)

def get_color_name(rgb):
    """
    RGB 값을 HSV 기반으로 사람이 인식하기 쉬운 색상 계열로 변환하는 함수.
    """

    # ✅ RGB → HSV 변환
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv

    # ✅ 색상 카테고리 정의 (HSV Hue 기준, 갈색을 넓히고 중복 해결)
    color_categories = {
        "푸른색": (90, 150),  # 푸른색 범위 확장 및 수정 (기존: 170, 270)
        "하늘색": (150, 180),  # 하늘색 추가
        "민트색": (170, 190),
        "초록색": (45, 90),   # 초록색 범위 조정 (기존: 80, 150)
        "노란색": (20, 45),   # 노란색 범위 조정 (기존: 50, 60)
        "주황색": (10, 20),   # 주황색 범위 조정 (기존: 20, 50)
        "붉은색": [(0, 10), (330, 360)],  # 붉은색 범위 조정
        "보라색": (270, 330), # 보라색 범위 조정 (기존: 270, 290)
        "갈색": (0, 20),      # 갈색 범위 수정 (기존: 250, 360) - 갈색은 낮은 채도/명도로 판단
    }

    result_colors = set()  # 다중 색상 결과를 담을 리스트 (중복 제거)

    # ✅ 회색 및 밝기 계열 분류 (채도가 낮을 때)
    if s < 40:  
        if v < 80:
            result_colors.add("어두운 색")
        elif v > 200:
            result_colors.add("밝은 색")
        else:
            result_colors.add("회색")
    else:
        # HSV의 H값은 0-179 범위인 경우가 있으므로 정규화
        # OpenCV의 H는 0-179, S와 V는 0-255 범위
        h_normalized = h * 2 if h <= 90 else h  # H값 정규화 (OpenCV에서는 0-180)
        
        # 색상 범주 매칭 로직 개선
        for category, hue_range in color_categories.items():
            if isinstance(hue_range, tuple):
                if hue_range[0] <= h_normalized < hue_range[1]:
                    result_colors.add(category)
            elif isinstance(hue_range, list):
                for hr in hue_range:
                    if hr[0] <= h_normalized < hr[1]:
                        result_colors.add(category)

    # 채도와 명도에 따른 추가 분류
    if s < 100 and v < 100:  # 채도와 명도가 낮은 경우 갈색 추가
        result_colors.add("갈색")
        # 회색 계열인 경우 제거
        if "푸른색" in result_colors and s < 60:
            result_colors.remove("푸른색")
    
    # 하늘색 보정 - 명도가 높고 채도가 낮은 경우
    if "하늘색" in result_colors or "푸른색" in result_colors:
        if v > 200 and s < 100:
            if "하늘색" not in result_colors:
                result_colors.add("하늘색")
            if "푸른색" in result_colors:
                result_colors.remove("푸른색")

    # 최종 색상 리스트 정리 (중복 제거 + 정렬)
    result_colors = sorted(result_colors)  # 정렬하여 일관된 순서 유지

    # 최종 색상 리스트 반환
    return ", ".join(result_colors) if result_colors else "회색"



# 결과 시각화
def display_results(image_path):
    image = load_and_preprocess_image(image_path)
    painting_region = detect_painting_region(image)  # 밝기 조정 없이 원본 그대로 사용
    edges = detect_edges(image)
    dominant_colors = extract_dominant_colors(painting_region)
    adjusted_colors = [adjust_hsv_lightness_and_saturation(tuple(color)) for color in dominant_colors] 
    
    
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title("Original Image")
    
    plt.subplot(1, 5, 2)
    plt.imshow(painting_region)
    plt.title("Detected Painting Region")

    plt.subplot(1, 5, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    
    plt.subplot(1, 5, 4)
    plt.imshow([dominant_colors / 255])
    plt.title("Dominant Colors")
    
    # 주요 색상 (명도 조정 후)
    plt.subplot(1, 5, 5)
    plt.imshow([np.array(adjusted_colors) / 255])
    plt.title("Brightness Adjusted")
    plt.axis("off")

    # 기존 plt.show() 대신 저장 방식으로 변경
    plt.show()
    #plt.savefig("/content/drive/MyDrive/Project/output_image.png") # 이거는 절대경로가 필요한 듯.
    return edges, adjusted_colors

if __name__ == "__main__":
    image_path = r"app/models/one_imageDetection/GardenatSainte-Adresse_monet.png"
    # 🔹 OpenCV 분석 실행
    display_results(image_path)