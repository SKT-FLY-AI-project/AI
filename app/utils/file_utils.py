import os
import shutil
import uuid
from fastapi import UploadFile
from pathlib import Path

# ✅ 업로드 디렉토리 설정
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성

# ✅ 최대 파일 크기 제한 (예: 10MB)
MAX_FILE_SIZE_MB = 10

def get_file_size(file: UploadFile) -> int:
    """파일 크기를 바이트 단위로 반환"""
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    return size

def is_valid_file_size(file: UploadFile) -> bool:
    """파일 크기가 제한을 초과하는지 확인"""
    file_size_mb = get_file_size(file) / (1024 * 1024)
    return file_size_mb <= MAX_FILE_SIZE_MB

def generate_unique_filename(filename: str) -> str:
    """UUID를 이용해 고유한 파일 이름 생성"""
    file_ext = filename.split(".")[-1]
    unique_name = f"{uuid.uuid4().hex}.{file_ext}"
    return unique_name

async def save_uploaded_file(file: UploadFile) -> str:
    """
    업로드된 파일을 저장하고 저장된 경로를 반환
    """
    if not is_valid_file_size(file):
        raise ValueError(f"파일 크기가 {MAX_FILE_SIZE_MB}MB를 초과했습니다.")

    unique_filename = generate_unique_filename(file.filename)
    file_path = UPLOAD_DIR / unique_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return str(file_path)

def cleanup_old_files(max_files: int = 50):
    """
    업로드 폴더에서 오래된 파일을 정리 (최대 파일 수 유지)
    """
    files = sorted(UPLOAD_DIR.glob("*"), key=lambda f: f.stat().st_mtime)  # 생성 날짜 기준 정렬
    if len(files) > max_files:
        for file in files[:-max_files]:  # 초과된 파일 삭제
            file.unlink()

def get_uploaded_files() -> list:
    """업로드된 파일 목록을 반환"""
    return [str(file) for file in UPLOAD_DIR.glob("*")]

