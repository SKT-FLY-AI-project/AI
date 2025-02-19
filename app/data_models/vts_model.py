from pydantic import BaseModel

class VTSQuestion(BaseModel):
    question: str
