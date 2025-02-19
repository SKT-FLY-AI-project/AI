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
from PIL import Image

#from app.models.one_imageDetection.opencv_utils import get_color_name
from one_imageDetection.opencv_utils import get_color_name
from langchain.prompts import PromptTemplate

import random


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

########################### SETP 2 : VLM (Vision to LLM) #####################################

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
    # 만약 image_path가 numpy 배열이라면:
    if isinstance(image_path, np.ndarray):
        image = Image.fromarray(image_path).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512)) # 일단은 크기 정규화 했는데 추후 수정 필요.
    
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


########################### STEP 3 : (VLM결과를 바탕으로) LLM ###############################
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
        **단, 200자 ~ 500자 사이의 길이로만 생성해야 합니다!**

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


########################### STEP 4 : 질문 답변 모드를 진행하는 함수 ###############################        
def answer_user_question(image_title, vlm_description, dominant_colors, edges):
    """사용자의 질문을 받아 LLM을 통해 답변을 생성하는 함수"""
    while True:
        user_question = input("\n❓ 추가 질문 (종료하려면 'exit' 입력): ")
        if user_question.lower() == "exit":
            print("📢 질문 모드 종료.")
            break
        
        # LLM에 전달할 프롬프트 생성
        prompt = f"""
        사용자는 '{image_title}' 작품에 대해 질문하고 있습니다.
        작품 설명: {vlm_description}
        주요 색상: {dominant_colors}
        엣지 감지 결과: {edges}
        
        사용자의 질문: "{user_question}"
        
        위 정보를 기반으로 사용자의 질문에 대해 상세하고 유익한 답변을 제공하세요.
        """
        
        # LLM을 이용한 답변 생성
        answer = generate_rich_description(image_title, prompt, dominant_colors, edges)
        print("\n💬 AI의 답변:")
        print(answer)

        # 음성 변환
        text_to_speech(answer, output_file=f"answer_{image_title}.mp3")
        
########################### STEP 5 : 질문 답변 모드를 진행하는 함수 ###############################
# 1. RAG
# 1-1. VTS 질문지 RAG에서 질문을 가져오는 함수 (예시) : VTS_RAG_questions.json
def load_vts_questions():
    # 현재 파일(llm.py)이 있는 폴더 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))  # three_llm 폴더 경로

    # JSON 파일 경로 설정
    file_path = os.path.join(current_dir, "data", "VTS_RAG_questions.json")

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ RAG 질문 파일을 찾을 수 없습니다: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# 1-2. 미술 정보 RAG에서 질문을 가져오는 함수 (예시) : art_RAG_questions.json
def load_art_questrion(query):
    """
    사용자의 감상 및 질문에 대해 관련 미술 정보를 검색하는 함수.
    """
    # 예시임. # art_RAG_question.json 파일 가져오는 코드로 수정하자.
    # retriever = FAISS.load_local("art_database", embeddings).as_retriever()
    # results = retriever.get_relevant_documents(query)
    # return results[0].page_content if results else "관련된 미술 정보를 찾지 못했습니다."

# 2. 질문 유형 파악하여 -> 알맞은 질문 형성하기
def retrieve_question(user_responses,image_title, vlm_description, dominant_colors, edges):
    """
    사용자의 이전 답변을 기반으로 적절한 'VTS 질문'을 RAG에서 검색
    """
    """
    - 사용자의 입력을 분석하여 VTS 질문을 생성할지, 미술 정보를 제공할지 결정.
    - 감정적 공감 또는 정보 제공이 필요한 경우: 미술 정보 검색
    - 질문을 생성할 경우: VTS 질문 매뉴얼 검색
    """
    # 사용자의 마지막 질문 (AI가 질문 생성을 위해 넣어줘야할 값)
    previous_responses = user_responses[-1]
    
    rag_questions = load_vts_questions() # 일단 테스트용으로 여기다 두긴 하는데... 나중에 정리합시다.
    # RAG에서 검색 (현재는 임시로 JSON에서 질문을 선택하는 형태)
    

    # 🎨 2-1. 사용자가 작품 설명을 요구함.
    if "느낌" in previous_responses or "설명" in previous_responses or "배경" in previous_responses: # NLP로 개선하기
        relevant_questions = []
        # 👉 RAG : 작품 정보 설명하기.
        relevant_questions = load_art_questrion(previous_responses)
        # 👉 RAG : VTS의 질문을 전달.
        relevant_questions = [q for q in rag_questions if q["classification"] == "질문"]

        print(f"📚 ART AI LLM : {relevant_questions}")
        return relevant_questions

    # 🎨 2-2. 사용자가 자신의 생각을 말함. 
    else:
        relevant_questions = []
        # 👉 RAG : : VTS의 반응와 관련된 말을 전달.
        # 사용자의 이전 답변을 분석 (여기서는 단순히 랜덤으로 선택, 실제 구현 시 NLP 활용 가능)
        relevant_questions = [q for q in rag_questions if q["classification"] == "반응"]

        # 👉 LLM : AI가 작품에 대해 생각하는 말을 전다. + LLM기반 작품 정보
        prompt = f"""
        사용자와 '{image_title}' 작품에 대해 대화하고 있습니다.
        작품 설명: {vlm_description}
        주요 색상: {dominant_colors}
        엣지 감지 결과: {edges}
        
        사용자의 생각: "{previous_responses}"
        
        위 정보를 기반으로 사용자의 생각에 대해 유익한 답변을 제공하세요.
        사용자 생각에대해 동의 및 다른 의견을 제시해주세요.
        """
        
        # LLM을 이용한 답변 생성
        answer = generate_rich_description(image_title, prompt, dominant_colors, edges)
        print("\n💬 AI의 답변:")
        print(answer)

        # 👉 RAG : VTS의 질문을 전달.
        # 첫 질문은 "전체에 대한 적극적인 관계 만들기"에서 선택
        relevant_questions = [q for q in rag_questions if q["classification"] == "질문"]
        return relevant_questions

    # # 랜덤으로 질문 하나 선택
    # if relevant_questions:
    #     return random.choice(relevant_questions)["question"]
    # else:
    #     return "이 작품을 어떻게 바라보면 좋을까요?"
    # return relevant_questions


# 감상 대화 함수 (개선 버전)
def start_vts_conversation(image_title, vlm_description, dominant_colors, edges):
    """VTS 방식의 감상 대화를 진행하는 함수"""
    print("\n🖼️ VTS 감상 모드 시작!")

    user_responses = []  # 사용자 응답 저장 리스트 # 단순 리스트로 정의하는 대신, DB에 저장하자.

    # 첫번째 물음 던지기.
    user_response_first = input("🎨 가장 크게 와닿는 부분이 무엇인가요? 전체적인 느낌은 어떤가요? (종료하려면 'exit' 입력): ") # GPT : 제시하는 첫 질문은 VTS 방식에 잘 맞아
    if user_response_first.lower() == "exit":
        print("📢 VTS 감상 모드 종료.")
        return
    user_responses.append(user_response_first)

    while True:
        # 이전 응답을 반영하여 적절한 질문 선택
        question = retrieve_question(user_responses,image_title, vlm_description, dominant_colors, edges)

        # 사용자 입력 받기
        user_response = input(f"\n🎨 {question} (종료하려면 'exit' 입력): ")
        if user_response.lower() == "exit":
            print("📢 VTS 감상 모드 종료.")
            break

        # 사용자 응답 저장
        user_responses.append(user_response) # 단순 리스트로 정의하는 대신, DB에 저장하자.