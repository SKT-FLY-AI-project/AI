# pip install opencv-python tensorflow numpy matplotlib
# pip install openai
# pip install requests
# pip install python-dotenv
# pip install langchain langchain-openai openai
# pip install python-dotenv groq
# pip install gTTS

# (venv) PS C:\Users\007\Documents\TEAM3_GITHUB_FORK\Back-End> python app/models/visionArt_final.py

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

# 실행
# image_path = "app\models\Van Gogh's The Starry Night.png"  # 사용자가 제공한 이미지 경로
# image_path = "app\models\London_CourtauldGallery_Cezanne's.png"  # 사용자가 제공한 이미지 경로
image_path = "app\models\London_CourtauldGallery_Manet'sABar.jpg"
display_results(image_path)

# LLM으로 넘기기
image = load_and_preprocess_image(image_path)
edges = detect_edges(image)
dominant_colors = extract_dominant_colors(image)

########################### SETP 2 : LLM #####################################

import openai
import os
import requests
import json
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from langchain.prompts import PromptTemplate
import numpy as np

# DeepSeek API 키를 환경 변수로 설정
client = Groq(api_key="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")


# # OpenAI API 키 설정 (자신의 API 키 입력)
# openai.api_key = "YOUR_API_KEY"

def generate_description(dominant_colors, edges):
    """
    OpenCV 분석 결과를 기반으로 자연어 설명을 생성하는 함수.
    
    Parameters:
        dominant_colors (np.ndarray): 대표 색상 목록 (RGB 형식)
        edges (np.ndarray): Edge Detection 결과 이미지
    
    Returns:
        str: 생성된 설명 문장
    """
    
    # 대표 색상을 텍스트로 변환
    color_names = [f"RGB({int(c[0])}, {int(c[1])}, {int(c[2])})" for c in dominant_colors[:5]]
    dominant_colors_text = ", ".join(color_names)
    
    # Edge Detection 결과 분석
    edges_detected = "명확히 탐지됨" if np.sum(edges) > 10000 else "불명확하게 탐지됨"



    # 프롬프트 템플릿 생성
    prompt_template = PromptTemplate(
        input_variables=["dominant_colors", "edges_detected"],
        template="""
        다음은 그림에 대한 설명을 생성하는 작업입니다.
        - 대표 색상은 다음과 같습니다: {dominant_colors}.
        - 이 그림의 경계는 {edges_detected}.
        
        이 정보를 바탕으로 시각장애인에게 이 그림을 설명하는 문장을 만들어주세요.
        감성적이면서 직관적으로 설명해주세요.
        한국어로 답변해주세요.
        """
    )
    
    # 프롬프트 포맷팅
    formatted_prompt = prompt_template.format(
        dominant_colors=dominant_colors_text,
        edges_detected=edges_detected
    )

    # Groq DeepSeek API 호출
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",  # Groq의 Mixtral 모델 사용
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.6,
        max_tokens=512,
        top_p=0.95
    )

    response_text = completion.choices[0].message.content.strip()

    # # '<think>'가 있는 경우 해당 부분을 제거
    # if "<think>" in response_text:
    #     response_text = response_text.split("</think>")[-1].strip()

    return response_text
    


# LLM 결과
# 설명 생성
description = generate_description(dominant_colors, edges)
print("생성된 설명:", description)
print("설명 끝.")
"""
생성된 설명: <think>
Alright, I need to describe this image to a visually impaired person. Let me start by understanding the given information. The dominant colors are RGB(212,213,215), which is a light gray, RGB(35,48,67), a dark navy, RGB(132,106,76), a warm brown, RGB(178,163,129), a soft beige, and RGB(76,132,184), a clear blue. The image has clear boundaries.

I should create a sentence that's both emotional and intuitive. I'll translate the colors into relatable terms. Light gray might represent a calm base, dark navy could be mysterious, warm brown as a cozy element, soft beige as gentle, and clear blue as refreshing. The clear boundaries suggest a structured and organized scene.

Putting it all together, I want to evoke a sense of balance and harmony. Maybe something like a serene landscape where each color adds depth and emotion. I'll make sure it's simple and paints a vivid mental picture.
</think>

이 그림은 차가운 회색 RGB(212,213,215)과 深海藍 RGB(35,48,67)로 이루어진 조화로운 배경 위에, 따뜻한 브라운 RGB(132,106,76)과 부드러운 베 이지 RGB(178,163,129)가 균형을 이루며 달콤한 분위기를 자아내는 모습을 하고 있습니다. 또한, 맑은 스카이 블루 RGB(76,132,184)가 그림의 경계를 명확하게 정의하며, 각각의 색상이 조화롭게 어우러져 감정적인 거동을 전달하는 인상적인 작품입니다.
설명 끝.
"""


########################### SETP 3 : TTS #####################################
from gtts import gTTS
import os

def text_to_speech(text, output_file="output.mp3"):
    """
    생성된 텍스트를 음성으로 변환하고 파일로 저장하는 함수.

    Parameters:
        text (str): 음성으로 변환할 텍스트
        output_file (str): 저장할 음성 파일의 이름 (기본값: "output.mp3")
    """
    try:
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()

        # gTTS를 사용해 텍스트를 음성으로 변환
        tts = gTTS(text=text, lang='ko')  # 한국어 사용 시 lang='ko'로 변경
        tts.save(output_file)
        print(f"음성 파일이 '{output_file}'로 저장되었습니다.")
        
        # 저장된 음성 파일 실행
        os.system(f"start {output_file}")  # Windows 사용 (macOS는 open, Linux는 xdg-open)
    except Exception as e:
        print(f"음성 변환 중 오류 발생: {e}")


# 텍스트를 음성으로 변환 및 실행
text_to_speech(description, output_file="description_audio.mp3")