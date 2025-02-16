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


# PS C:\Users\007\Documents\TEAM3_GITHUB\AI> venv\Scripts\activate
# (venv) PS C:\Users\007\Documents\TEAM3_GITHUB\AI> python app/models/model_main.py "app/models/one_imageDetection/London_CourtauldGallery_Manet'sABar.jpg"

# python main.py "app/models/London_CourtauldGallery_Manet'sABar.jpg"
# image_path = "app\models\Van Gogh's The Starry Night.png"  # 사용자가 제공한 이미지 경로
# image_path = "app\models\London_CourtauldGallery_Cezanne's.png"  # 사용자가 제공한 이미지 경로
# image_path = "app\models\London_CourtauldGallery_Manet'sABar.jpg"

# main.py
# model_main.py
import sys
import os
from one_imageDetection.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors, display_results
from three_llm.llm import generate_vlm_description_qwen, generate_rich_description, text_to_speech


if __name__ == "__main__":
    # 🔹 테스트할 이미지 리스트
    test_images = [
        "app/models/one_imageDetection/London_CourtauldGallery_Cezanne's.png",
        "app/models/one_imageDetection/London_CourtauldGallery_Manet'sABar.jpg",
        "app/models/one_imageDetection/Van Gogh's The Starry Night.png",
    ]

    for image_path in test_images:
        print(f"\n🔎 테스트 중: {image_path}")
        display_results(image_path)

        # 🔹 OpenCV 분석 실행
        image = load_and_preprocess_image(image_path)
        edges = detect_edges(image)
        dominant_colors = extract_dominant_colors(image)

       # ✅ Qwen2.5-VL 실행
        print("\n🎨 Qwen2.5-VL 모델 실행 중...")
        vlm_descriptions = generate_vlm_description_qwen(image_path)

        # ✅ 결과가 문자열이면 리스트로 변환
        if isinstance(vlm_descriptions, str):
            vlm_descriptions = [vlm_descriptions]

        # ✅ 결과가 None이면 기본값 설정
        if vlm_descriptions is None:
            vlm_descriptions = ["설명을 생성할 수 없습니다."]

        # ✅ 리스트를 줄바꿈으로 연결하여 출력
        print("\n".join(vlm_descriptions))

        # 🔹 LLM을 활용한 설명 생성
        rich_description = generate_rich_description("테스트 그림", vlm_descriptions[0], dominant_colors, edges)
        print("\n📜 생성된 설명:")
        print(rich_description)

        # 🔹 음성 변환 실행
        text_to_speech(rich_description, output_file=f"output_{os.path.basename(image_path)}.mp3")
