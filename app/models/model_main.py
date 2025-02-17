# PS C:\Users\007\Documents\TEAM3_GITHUB\AI> venv\Scripts\activate
# (venv) PS C:\Users\007\Documents\TEAM3_GITHUB\AI> python app/models/model_main.py "app/models/one_imageDetection/London_CourtauldGallery_Manet'sABar.jpg"

# 패키지 설치가 안 되어있다면, 아래 코드 중 필요한 패키지 설치.
# pip install opencv-python tensorflow numpy matplotlib
# pip install openai
# pip install requests
# pip install python-dotenv
# pip install langchain langchain-openai openai
# pip install python-dotenv groq
# pip install gTTS

# pip install --upgrade torch accelerate
# pip install qwen-vl-utils[decord]==0.0.8
# pip install --upgrade transformers
# pip show qwen-vl-utils
# pip install bitsandbytes # 이건 결국 못썼음.
# pip install git+https://github.com/huggingface/transformers accelerate


#pip install fastapi
#pip install uvicorn
#pip install python-multipart


# PS C:\Users\007\Documents\TEAM3_GITHUB\AI> venv\Scripts\activate
# (venv) PS C:\Users\007\Documents\TEAM3_GITHUB\AI> python app/models/model_main.py "app/models/one_imageDetection/London_CourtauldGallery_Manet'sABar.jpg"

# python main.py "app/models/London_CourtauldGallery_Manet'sABar.jpg"
# image_path = "app\models\Van Gogh's The Starry Night.png"  # 사용자가 제공한 이미지 경로
# image_path = "app\models\London_CourtauldGallery_Cezanne's.png"  # 사용자가 제공한 이미지 경로
# image_path = "app\models\London_CourtauldGallery_Manet'sABar.jpg"

# main.py
# model_main.py

# fast API 실험용
import sys
import os
from app.models.one_imageDetection.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors
from app.models.three_llm.llm import generate_vlm_description_qwen, generate_rich_description, text_to_speech

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def process_image(image_path):
    """
    이미지를 받아서 분석 후 결과를 JSON 형태로 반환하는 함수.
    """
    # 🔹 이미지 전처리 및 OpenCV 분석
    image = load_and_preprocess_image(image_path)
    edges = detect_edges(image)
    dominant_colors = extract_dominant_colors(image)

    # ✅ Qwen2.5-VL 실행하여 설명 생성
    vlm_description = generate_vlm_description_qwen(image_path)

    if isinstance(vlm_description, str):
        vlm_description = [vlm_description]

    # ✅ 설명이 없을 경우 기본값 설정
    if not vlm_description:
        vlm_description = ["설명을 생성할 수 없습니다."]

    # 🔹 LLM을 활용한 설명 생성
    rich_description = generate_rich_description("분석된 그림", vlm_description[0], dominant_colors, edges)

    # 🔹 음성 변환 실행 (음성 파일 저장)
    audio_filename = f"output_{os.path.basename(image_path)}.mp3"
    audio_path = f"uploads/{audio_filename}"
    text_to_speech(rich_description, output_file=audio_path)

    # 🔹 결과 JSON 반환
    return {
        "image_path": image_path,
        "vlm_description": vlm_description[0],
        "rich_description": rich_description,
        "dominant_colors": dominant_colors.tolist(),
        "edges_detected": "명확히 탐지됨" if edges.sum() > 10000 else "불명확",
        "audio_url": f"/static/{audio_filename}"  # 프론트엔드에서 음성 파일 접근 가능하도록 URL 제공
    }


# 테스트용 코드
# import os
# from one_imageDetection.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors, display_results
# from three_llm.llm import generate_vlm_description_qwen, generate_rich_description, text_to_speech, answer_user_question, start_vts_conversation


# if __name__ == "__main__":
#     # 🔹 테스트할 이미지 리스트
#     test_images = [
#         "app/models/one_imageDetection/London_CourtauldGallery_Cezanne's.png",
#         "app/models/one_imageDetection/London_CourtauldGallery_Manet'sABar.jpg",
#         "app/models/one_imageDetection/Van Gogh's The Starry Night.png",
#     ]

#     for image_path in test_images:
#         print(f"\n🔎 테스트 중: {image_path}")
#         display_results(image_path)

#         # 🔹 OpenCV 분석 실행
#         image = load_and_preprocess_image(image_path)
#         edges = detect_edges(image)
#         dominant_colors = extract_dominant_colors(image)

#        # ✅ Qwen2.5-VL 실행
#         print("\n🎨 Qwen2.5-VL 모델 실행 중...")
#         vlm_descriptions = generate_vlm_description_qwen(image_path)

#         # ✅ 결과가 문자열이면 리스트로 변환
#         if isinstance(vlm_descriptions, str):
#             vlm_descriptions = [vlm_descriptions]

#         # ✅ 결과가 None이면 기본값 설정
#         if vlm_descriptions is None:
#             vlm_descriptions = ["설명을 생성할 수 없습니다."]

#         # ✅ 리스트를 줄바꿈으로 연결하여 출력
#         print("\n".join(vlm_descriptions))

#         # 🔹 LLM을 활용한 설명 생성
#         rich_description = generate_rich_description("테스트 그림", vlm_descriptions[0], dominant_colors, edges)
#         print("\n📜 생성된 설명:")
#         print(rich_description)

#         # 🔹 음성 변환 실행
#         text_to_speech(rich_description, output_file=f"output_{os.path.basename(image_path)}.mp3")
        
#         ################################# 여기는 추후 상황에 따라 밑의 함수를 돌릴 수 있도록 해야 한다고 생각함. ##########################
        
#         # 🔹 4번: 사용자 질문 답변 처리
#         answer_user_question("테스트 그림", vlm_descriptions[0], dominant_colors, edges)

#         # 🔹 5번: VTS 방식 감상 지원
#         start_vts_conversation("테스트 그림", vlm_descriptions[0])
