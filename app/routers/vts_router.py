from fastapi import APIRouter
from app.services.vts_service import get_vts_question
from app.data_models.response_model import BaseResponse

router = APIRouter()

@router.get("/get_question/", response_model=BaseResponse)
async def get_vts():
    """ VTS 질문을 랜덤으로 가져오는 API """
    question = get_vts_question()
    return BaseResponse(success=True, data=question)
