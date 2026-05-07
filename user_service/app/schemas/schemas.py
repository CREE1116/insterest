from pydantic import BaseModel, ConfigDict, Field
import uuid
from typing import Optional, List
from datetime import datetime

class UserProfileRead(BaseModel):
    id: uuid.UUID = Field(..., alias="user_id")
    user_id: uuid.UUID
    nickname: Optional[str] = None
    profile_image: Optional[str] = None
    bio: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

class UserProfileUpdate(BaseModel):
    nickname: Optional[str] = None
    bio: Optional[str] = None

class CollectionCreate(BaseModel):
    name: str

class CollectionRead(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime
    item_count: int = 0
    collection_images: List[str] = []
    
    model_config = ConfigDict(from_attributes=True)

class CollectionItemCreate(BaseModel):
    post_id: uuid.UUID
