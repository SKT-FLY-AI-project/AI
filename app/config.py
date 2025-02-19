import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

CORS_ORIGINS = ["*"]
UPLOAD_FOLDER = "uploads"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
