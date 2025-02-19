from pydantic import BaseModel

class ImageAnalysisResult(BaseModel):
    filename: str
    description: str
