from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.image_router import router as image_router # router 객체를 직접 임포트
from app.routers.vts_router import router as vts_router  # router 객체를 직접 임포트
from app.services import process_uploaded_image
from app.ai_models import predict_image


app = FastAPI(title="VTS AI Backend", version="1.0")

# CORS 설정 (Flutter 연동 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(image_router, prefix="/image", tags=["Image Analysis"])
app.include_router(vts_router, prefix="/vts", tags=["VTS AI"])

@app.get("/")
def root():
    return {"message": "VTS AI Backend is running!"}
