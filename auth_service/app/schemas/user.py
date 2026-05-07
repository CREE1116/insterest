import uuid
from typing import Optional
from pydantic import BaseModel, EmailStr

# User Base
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    nickname: Optional[str] = None
    profile_image: Optional[str] = None
    is_active: Optional[bool] = True

# User Create
class UserCreate(UserBase):
    email: EmailStr
    password: str
    nickname: str

# User Update
class UserUpdate(BaseModel):
    nickname: Optional[str] = None
    profile_image: Optional[str] = None
    password: Optional[str] = None

# User Response
class User(UserBase):
    id: uuid.UUID
    is_verified: bool
    is_superuser: bool

    class Config:
        from_attributes = True

# Token
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

# Token Payload
class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None
    type: Optional[str] = None
