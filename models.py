from typing import List

from pydantic import BaseModel


class Emotion(BaseModel):
    face_num: int
    emotion: str


class EmotionsResponse(BaseModel):
    emotions: List[Emotion]
