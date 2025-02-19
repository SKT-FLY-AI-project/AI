import torch
import asyncio
from PIL import Image
from app.ai_models.qwen_model import get_vlm_model, get_coder_model
from app.utils.opencv_utils import extract_dominant_colors, detect_edges, get_color_name
from langchain.prompts import PromptTemplate

# ✅ Qwen2.5-VL 모델 가져오기
vlm_model, vlm_processor = get_vlm_model()

async def generate_vlm_description_qwen(image_path):
    """ Qwen2.5-VL 모델을 사용해 비동기 이미지 설명 생성 """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    # ✅ OpenCV 활용 색감 분석
    dominant_colors = extract_dominant_colors(image_path)
    edges_detected = detect_edges(image_path)

    # ✅ LLM 프롬프트 생성
    prompt = f"이 이미지를 보고 장면, 색채, 구도, 분위기, 주요 특징을 설명하세요.\n" \
             f"주요 색상: {', '.join(dominant_colors)}\n" \
             f"경계 감지 결과: {'명확함' if edges_detected else '불명확함'}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm_processor(text=[text_input], images=image, return_tensors="pt", padding=True).to(vlm_model.device)

    # ✅ 비동기 실행 (토큰 생성 속도 최적화)
    with torch.no_grad():
        outputs = await asyncio.to_thread(vlm_model.generate, **inputs, max_new_tokens=256)

    return vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

def generate_rich_description(title, vlm_desc, dominant_colors, edges):
    """ AI가 생성한 기본 설명을 기반으로 보다 풍부한 그림 설명을 생성하는 함수. """
    coder_model, coder_tokenizer = get_coder_model()
    
    color_names = [get_color_name(c) for c in dominant_colors[:5]]
    dominant_colors_text = ", ".join(color_names)
    edges_detected = "명확히 탐지됨" if edges.sum() > 10000 else "불명확하게 탐지됨"
    
    prompt_template = PromptTemplate(
        input_variables=["title", "vlm_desc", "dominant_colors", "edges_detected"],
        template=f"""
        당신은 그림 설명 전문가입니다.  
        다음 그림에 대해 상세한 설명을 생성해주세요.
        시각장애인에게 설명할 수 있도록 자세하게 작성해 주세요.
        **단, 200자 ~ 500자 사이의 길이로만 생성해야 합니다!**

        - **제목:** "{{title}}"  
        - **VLM 기반 기본 설명:** "{{vlm_desc}}"   

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
    
    inputs = coder_tokenizer(formatted_prompt, return_tensors="pt").to(coder_model.device)
    with torch.no_grad():
        output = coder_model.generate(**inputs, max_new_tokens=1024)
    
    return coder_tokenizer.decode(output[0], skip_special_tokens=True).strip()