"""
FastAPI 라우터 모듈 초기화 파일
"""
from .image_router import router as image_router
from .vts_router import router as vts_router

__all__ = ["image_router", "vts_router"]
