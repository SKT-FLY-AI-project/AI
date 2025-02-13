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