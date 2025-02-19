import random
from app.ai_models.vts_model import start_vts_conversation

# 샘플 VTS 질문 목록
VTS_QUESTIONS = [
    "이 작품에서 어떤 감정이 느껴지나요?",
    "이 장면의 분위기는 어떤가요?",
    "이 작품에서 가장 눈에 띄는 요소는 무엇인가요?"
]

def get_vts_question():
    """ 랜덤한 VTS 질문 반환 """
    return random.choice(VTS_QUESTIONS)
