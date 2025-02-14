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

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    prompt = "Describe this painting's scene, colors, composition, and mood in detail."
    
    # 🔹 프롬프트를 input_ids로 변환
    prompt_inputs = processor.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
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