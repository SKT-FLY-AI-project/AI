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
import torch
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration
from one_imageDetection.opencv_utils import get_color_name

# DeepSeek API 키를 환경 변수로 설정
client = Groq(api_key="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")


# # OpenAI API 키 설정 (자신의 API 키 입력)
# openai.api_key = "YOUR_API_KEY"

def generate_blip_description(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

import re

# 정제 코드
def clean_and_restore_spacing(text):
    """
    Qwen2.5-VL의 출력에서 시스템 메시지를 제거하고 띄어쓰기를 복원하는 함수.
    """
    # ✅ 1. "이 그림은" 또는 "이 장면은"이 나오기 전까지 모든 텍스트 제거
    text = re.sub(r".*?(이 그림은|이 장면은)", r"\1", text, flags=re.IGNORECASE | re.DOTALL)

    # ✅ 2. "이 이미지를 보고 ~ 설명하세요" 같은 프롬프트 제거
    prompt_text = "이 이미지를 보고 장면, 색채, 구도, 분위기, 주요 특징을 설명하세요."
    text = text.replace(prompt_text, "").strip()

    # ✅ 3. 연속된 공백을 한 개의 공백으로 변경
    text = re.sub(r"\s+", " ", text).strip()

    # ✅ 4. 한글과 영어/숫자 사이에 공백 추가 (자연스러운 띄어쓰기 복원)
    text = re.sub(r"([가-힣])([a-zA-Z0-9])", r"\1 \2", text)  # 한글 + 영어/숫자
    text = re.sub(r"([a-zA-Z0-9])([가-힣])", r"\1 \2", text)  # 영어/숫자 + 한글

    return text

# 이미지 설명 VLM
def generate_vlm_description_qwen(image_path):
    # ✅ 이미지 로드 및 리사이징 (512x512)
    # image_path가 numpy 배열일 경우 변환
    if isinstance(image_path, np.ndarray):
        image = Image.fromarray(image_path).convert("RGB")
    elif isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        raise TypeError("image_path must be a file path (str) or a numpy.ndarray.")

    image = image.resize((512, 512)) # 일단은 크기 정규화 했는데 추후 수정 필요.
    
    with torch.no_grad():
        output_with_prompt = model.generate(**inputs, input_ids=prompt_inputs.input_ids, max_length=150)
        caption_with_prompt = processor.batch_decode(output_with_prompt, skip_special_tokens=True)[0]

    print(f"🔹 **BLIP Prompt-Based Caption:** {caption_with_prompt}")

    return caption_with_prompt

def generate_rich_description(title, blip_desc, dominant_colors, edges):
    color_names = [get_color_name(c) for c in dominant_colors[:5]]
    dominant_colors_text = ", ".join(color_names)
    edges_detected = "명확히 탐지됨" if np.sum(edges) > 10000 else "불명확하게 탐지됨"

    prompt_template = PromptTemplate(
        input_variables=["title", "blip_desc", "dominant_colors", "edges_detected"],
        template=f"""
        Please generate a detailed description of the painting titled **"{title}"**.  

        - **Painting description (BLIP):** {blip_desc}  
        - **Dominant Colors (Natural Language):** {dominant_colors_text}  
        - **Edge Detection Analysis:** {edges_detected}  

        The text **must consist ONLY of Korean Hangul syllables (가-힣), without any exceptions**.  
        """
    )

    formatted_prompt = prompt_template.format(
        title=title,
        blip_desc=blip_desc,
        dominant_colors=dominant_colors_text,
        edges_detected=edges_detected
    )
    
    completion = client.chat.completions.create(
        model="qwen-2.5-coder-32b",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.5,
        max_tokens=1024,
        top_p=0.95
    )
    
    return completion.choices[0].message.content.strip()


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