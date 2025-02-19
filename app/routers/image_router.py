from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.image_service import process_uploaded_image
from app.data_models.response_model import BaseResponse

router = APIRouter()

@router.post("/analyze/", response_model=BaseResponse)
async def analyze_image(file: UploadFile = File(...)):
    """ 이미지를 업로드하고 분석 결과를 반환하는 API """
    try:
        result = await process_uploaded_image(file)
        return BaseResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))