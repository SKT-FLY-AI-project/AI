from pydantic import BaseModel
from typing import Any, Optional

class BaseResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
