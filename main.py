from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from app.models.model_main import process_image  # model_main.py에서 이미지 분석 함수 불러옴
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS 설정 (Flutter 연동 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (보안 필요시 수정)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 'uploads' 폴더를 Static 파일 경로로 설정
app.mount("/static", StaticFiles(directory="uploads"), name="static")

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    """
    이미지를 업로드하면 분석을 수행하고 결과를 반환하는 API 엔드포인트.
    """
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    
    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔹 이미지 분석 실행
    result = process_image(file_path)

    return result  # JSON 응답 반환

@app.get("/")
def read_root():
    return {"message": "FastAPI 서버 실행 중"}
