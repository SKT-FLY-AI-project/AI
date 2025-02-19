import os
import shutil
from fastapi import UploadFile
from app.data_models.image_model import ImageAnalysisResult
from app.ai_models.llm_model import generate_vlm_description_qwen, generate_rich_description
from app.utils.opencv_utils import extract_dominant_colors, detect_edges

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def process_uploaded_image(file: UploadFile) -> ImageAnalysisResult:
    """ 업로드된 이미지를 저장하고 분석을 수행 """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # ✅ OpenCV 분석
    dominant_colors = extract_dominant_colors(file_path)
    edges_detected = detect_edges(file_path)

    # ✅ 기본 VLM 설명 생성
    vlm_description = await generate_vlm_description_qwen(file_path)

    # ✅ LLM 기반 풍부한 설명 생성
    rich_description = await generate_rich_description(file.filename, vlm_description, dominant_colors, edges_detected)

    return ImageAnalysisResult(
        filename=file.filename,
        description=rich_description
    )