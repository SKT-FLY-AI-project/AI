import json
import torch
import asyncio
from app.ai_models.qwen_model import get_coder_model

# ✅ Qwen2.5-Coder-32B 모델 가져오기
coder_model, coder_tokenizer = get_coder_model()

# ✅ VTS 질문 데이터 로드
VTS_QUESTIONS_PATH = "app/data/vts_questions.json"

def load_vts_questions():
    """ VTS 질문 파일을 불러오는 함수 """
    
    if not os.path.exists(VTS_QUESTIONS_PATH):
        raise FileNotFoundError(f"❌ VTS 질문 파일을 찾을 수 없습니다: {VTS_QUESTIONS_PATH}")
    
    with open(VTS_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def clean_vts_response(response: str):
    """ VTS 응답을 반응과 질문으로 분리하는 함수 """
    try:
        response_parts = response.split("\n")
        reaction = response_parts[0].strip() if response_parts else "흥미로운 생각이에요."
        question = response_parts[1].strip() if len(response_parts) > 1 else "이 작품을 보고 어떤 점이 가장 인상적이었나요?"
    except:
        reaction, question = response, "이 작품을 보고 어떤 점이 가장 인상적이었나요?"
    return reaction, question    

async def generate_vts_question(user_input, conversation_history):
    """
    사용자의 입력과 대화 히스토리를 기반으로 적절한 반응과 질문을 생성하는 비동기 함수.
    """
    coder_model, coder_tokenizer = get_coder_model()
    
    # 🔹 대화 맥락 정리 (최근 3개만 유지)
    context = "\n".join(conversation_history[-3:])

    prompt = f"""
    사용자가 미술 작품을 감상하고 있습니다.
    이전 대화:
    {context}

    사용자의 입력:
    "{user_input}"

    AI의 역할:
    1. 사용자의 감상에 대해 적절한 반응을 제공합니다.
    2. 새로운 질문을 생성하여 자연스럽게 대화를 이어갑니다.

    AI의 응답 형식:
    1. 반응: (사용자의 감상을 반영한 피드백)
    2. 질문: (VTS 기반의 적절한 추가 질문)
    """
    
    inputs = coder_tokenizer(prompt, return_tensors="pt").to(coder_model.device)
    with torch.no_grad():
        output = await asyncio.to_thread(coder_model.generate, **inputs, max_new_tokens=150)
    
    response = coder_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return clean_vts_response(response)


async def answer_user_question(history, user_input):
    """ Qwen2.5-Coder-32B를 사용해 사용자 질문에 답변 생성 """
    prompt = f"다음 미술 감상 대화 기록을 참고하여 사용자의 질문에 대답하세요.\n\n"
    prompt += "\n".join(history)
    prompt += f"\n\n사용자 질문: {user_input}\n\nAI의 답변:"

    inputs = coder_tokenizer(prompt, return_tensors="pt").to(coder_model.device)

    with torch.no_grad():
        output = await asyncio.to_thread(coder_model.generate, **inputs, max_new_tokens=128)

    return coder_tokenizer.decode(output[0], skip_special_tokens=True).strip()

async def start_vts_conversation(title, description, dominant_colors, edges):
    """ 
    VTS 방식 감상 대화 진행 
    - 작품 제목과 설명을 바탕으로 질문을 생성하고, 사용자의 응답을 받아 진행 
    """
    print("\n🖼️ VTS 감상 모드 시작!")

    conversation_history = [f"작품 제목: {title}", f"작품 설명: {description}", f"주요 색상: {', '.join(map(str, dominant_colors))}", f"엣지 감지 결과: {'명확함' if edges else '불명확함'}"]

    user_response = input("🎨 작품을 보고 떠오른 느낌이나 궁금한 점을 말해주세요 (종료: exit): ")

    while user_response.lower() != "exit":
        conversation_history.append(f"사용자: {user_response}")

        # 🔹 AI 반응 및 질문 생성
        reaction = await answer_user_question(conversation_history, user_response)
        next_question = await generate_vts_question(conversation_history)

        # 🔹 대화 히스토리에 추가
        conversation_history.append(f"AI: {reaction}")
        conversation_history.append(f"AI 질문: {next_question}")

        # 🔹 피드백 및 다음 질문 출력
        print(f"\n💬 {reaction}")
        user_response = input(f"🎨 {next_question} (종료: exit): ")

    print("📢 VTS 감상 모드 종료.")