########################### SETP 2 : LLM #####################################

import openai
import os
import requests
import json
from dotenv import load_dotenv
from groq import Groq
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

from one_imageDetection.opencv_utils import get_color_name
from langchain.prompts import PromptTemplate


# Hugging Face 모델 캐시 경로 설정
os.environ['HF_HOME'] = "D:/huggingface_models"
client = Groq(api_key="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")

# 모델 정보 설정
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 모델 로드 (FP16으로 변경)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # ✅ FP16 사용 (BF16 문제 방지)
    device_map="auto",
    max_memory={0: "10GiB", "cpu": "30GiB"}
)

processor = AutoProcessor.from_pretrained(model_name)

import re

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

def generate_vlm_description_qwen(image_path):
    # ✅ 이미지 로드 및 리사이징 (512x512)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    
    prompt = "이 이미지를 보고 장면, 색채, 구도, 분위기, 주요 특징을 설명하세요."

    # ✅ 메시지 형식으로 변환 (apply_chat_template 사용)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # ✅ Chat Template 적용 (Qwen2.5-VL에서는 필수)
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # 이거 없으면 그냥 안돌아갑니다 진짜 중요함

    # ✅ 모델 입력 변환
    inputs = processor(
        text=[text_input],  # ✅ 변환된 텍스트 입력
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # ✅ 모델 실행 (토큰 수 최적화)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256) # 128로 하니까 좀 짤리는듯;

    # ✅ 결과 디코딩 및 세로 출력 문제 해결
    description = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    description = clean_and_restore_spacing(description)

    return description


########################### STEP 3 : 텍스트 생성 및 음성 변환 ###############################
from gtts import gTTS

def generate_rich_description(title, vlm_desc, dominant_colors, edges):
    """
    AI가 생성한 기본 설명을 기반으로 보다 풍부한 그림 설명을 생성하는 함수.
    """
    color_names = [get_color_name(c) for c in dominant_colors[:5]]
    dominant_colors_text = ", ".join(color_names)
    edges_detected = "명확히 탐지됨" if np.sum(edges) > 10000 else "불명확하게 탐지됨"

    prompt_template = PromptTemplate(
        input_variables=["title", "vlm_desc", "dominant_colors", "edges_detected"],
        template=f"""
        당신은 그림 설명 전문가입니다.  
        다음 그림에 대해 상세한 설명을 생성해주세요.
        시각장애인에게 설명할 수 있도록 자세하게 작성해 주세요.

        - **제목:** "{title}"  
        - **VLM 기반 기본 설명:** "{vlm_desc}"   

        위 정보를 바탕으로 그림에 대한 상세한 설명을 작성해 주세요.  
        작품의 분위기, 색채, 구도, 표현 기법 등을 분석하고,  
        가능하다면 역사적, 예술적 배경도 함께 제공해 주세요.  
        설명은 반드시 **한글(가-힣)만 사용하여 작성해야 합니다.**  
        영어, 숫자, 특수문자는 포함할 수 없습니다.  
        """
    )

    formatted_prompt = prompt_template.format(
        title=title,
        vlm_desc=vlm_desc,
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

def text_to_speech(text, output_file="output.mp3"):
    """
    생성된 텍스트를 음성 파일로 변환하여 저장하는 함수.
    """
    try:
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()
        tts = gTTS(text=text, lang='ko')
        tts.save(output_file)
        print(f"음성 파일이 '{output_file}'로 저장되었습니다.")
        os.system(f"start {output_file}")  # Windows (macOS: open, Linux: xdg-open)
    except Exception as e:
        print(f"음성 변환 중 오류 발생: {e}")