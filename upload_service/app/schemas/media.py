from pydantic import BaseModel
from typing import List, Optional
import uuid
import enum

class ContentType(str, enum.Enum):
    PHOTO = "PHOTO"
    PHOTO_SOUND = "PHOTO_SOUND"
    VIDEO = "VIDEO"

class PostBase(BaseModel):
    body: str
    hashtags: Optional[List[str]] = None

class PostCreate(PostBase):
    pass

class PostRead(PostBase):
    id: uuid.UUID
    content_id: uuid.UUID

    class Config:
        from_attributes = True

class ContentRead(BaseModel):
    id: uuid.UUID
    content_type: ContentType
    author_id: uuid.UUID
    prompt: Optional[str] = None
    posts: List[PostRead] = []

    class Config:
        from_attributes = True
