from app.ai_models.llm_model import generate_vlm_description_qwen

def generate_vlm_description(image_path: str) -> str:
    """ Qwen2.5-VL을 사용하여 이미지 설명 생성 """
    return generate_vlm_description_qwen(image_path)
