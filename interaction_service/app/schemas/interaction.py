import uuid
from pydantic import BaseModel
from typing import Optional

class InteractionBase(BaseModel):
    post_id: uuid.UUID

class WatchTimeCreate(InteractionBase):
    duration_seconds: float

class LikeToggleResponse(BaseModel):
    liked: bool

class SaveToggleResponse(BaseModel):
    saved: bool

class InteractionCount(BaseModel):
    post_id: uuid.UUID
    likes: int = 0
    saves: int = 0
    views: int = 0
    avg_watch_time: float = 0.0
