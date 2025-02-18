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

# pip install qwen-vl-utils
# pip install bitsandbytes # 이건 결국 못썼음.
# pip install git+https://github.com/huggingface/transformers accelerate
# pip install torchvision

# python -c "import torch; print(torch.cuda.is_available())" # False 이면, GPU 사용 설정 해줘야한다.
# 설치 방법 : https://doraemin.tistory.com/131 



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

from one_imageDetection.opencv_utils import load_and_preprocess_image, detect_painting_region, detect_edges, extract_dominant_colors, display_results
from two_cnn.cnn_test import predict_image
from three_llm.llm import generate_vlm_description_qwen, generate_rich_description, text_to_speech, answer_user_question, start_vts_conversation


from one_imageDetection.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors, display_results
from three_llm.llm import generate_vlm_description_qwen, generate_rich_description, text_to_speech, answer_user_question, start_vts_conversation


if __name__ == "__main__":
    # 🔹 테스트할 이미지 리스트
    test_images = [

        # "app/models/one_imageDetection/London_CourtauldGallery_Cezanne's.png",      # Unknown title
        # "app/models/one_imageDetection/London_CourtauldGallery_Manet'sABar.jpg",    # Unknown title
        # "app/models/one_imageDetection/Van Gogh's The Starry Night.png",            # Unknown title
        # r"app\models\two_cnn\data\cnn_test_data\test_ViewofToledo_ElGreco.png",     # View of Toledo
        # r"app\models\two_cnn\data\cnn_test_data\test_GardenatSainte-Adresse_monet.png", # Garden at Sainte-Adresse
        # r"app\models\one_imageDetection\GardenatSainte-Adresse_monet.png", # GardenatSainte-Adresse_monet - Detection 해야함
        r"app\models\one_imageDetection\GardenatSainte-Adresse_monet.png"           # Garden at Sainte-Adresse

    ]

    for image_path in test_images:
        print(f"\n🔎 테스트 중: {image_path}")
        display_results(image_path)

        # 🔹 OpenCV 분석 실행
        image_pre = load_and_preprocess_image(image_path)
        image = detect_painting_region(image_pre)
        edges = detect_edges(image)
        dominant_colors = extract_dominant_colors(image)

       # ✅ Qwen2.5-VL 실행
        print("\n🎨 Qwen2.5-VL 모델 실행 중...")

        vlm_descriptions = generate_vlm_description_qwen(image)

        # ✅ 결과가 문자열이면 리스트로 변환
        if isinstance(vlm_descriptions, str):
            vlm_descriptions = [vlm_descriptions]

        # ✅ 결과가 None이면 기본값 설정
        if vlm_descriptions is None:
            vlm_descriptions = ["설명을 생성할 수 없습니다."]

        # ✅ 리스트를 줄바꿈으로 연결하여 출력
        print("\n".join(vlm_descriptions))


        # 해당 작품이 AI가 학습한 것이면, 제목이 return "{class_name}"
        # 해당 작품이 AI가 학습한 것이 아니면, return "Unknown Title"
        title = predict_image(image)
        if isinstance(title, set):
            title = list(title)[0]  # set을 리스트로 변환 후 첫 번째 값 가져오기

        print("작품 제목 추출 결과입니다.", title)

        # 🔹 LLM을 활용한 설명 생성
        rich_description = generate_rich_description(title, vlm_descriptions[0], dominant_colors, edges)

        print("\n📜 생성된 설명:")
        print(rich_description)

        # 🔹 음성 변환 실행
        text_to_speech(rich_description, output_file=f"output_{os.path.basename(image_path)}.mp3")

        
        ################################# 여기는 추후 상황에 따라 밑의 함수를 돌릴 수 있도록 해야 한다고 생각함. ##########################
        
        # 🔹 4번: 사용자 질문 답변 처리
        answer_user_question(title, rich_description[0], dominant_colors, edges)

        # 🔹 5번: VTS 방식 감상 지원
        start_vts_conversation(title, rich_description[0], dominant_colors, edges)