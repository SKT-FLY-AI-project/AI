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
import sys
import os

from one_imageDetection.opencv_utils import load_and_preprocess_image, detect_edges, extract_dominant_colors, display_results
from three_llm.llm import generate_vlm_description_qwen, generate_rich_description, text_to_speech # generate_blip_description

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python main.py <이미지 경로>")
        sys.exit(1)
        
    painting_title = "폴리 베르제르 바" # 임시로 지정했다 치고
    
    # one : openCV 실행
    image_path = sys.argv[1]  # 터미널에서 입력받은 이미지 경로
    display_results(image_path)

    # three : LLM으로 넘기기
    image = load_and_preprocess_image(image_path)
    edges = detect_edges(image)
    dominant_colors = extract_dominant_colors(image)
    
    
    blip_desc = generate_vlm_description_qwen(image_path)
    edges, dominant_colors = display_results(image_path)
    rich_description = generate_rich_description(painting_title, blip_desc, dominant_colors, edges)

    print("생성된 설명:", rich_description)

    # 텍스트를 음성으로 변환 및 실행
    text_to_speech(rich_description, output_file="description_audio.mp3")
    
    