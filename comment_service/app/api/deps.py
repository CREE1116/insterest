import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from jose import jwt, JWTError
import os

SECRET_KEY = os.getenv("SECRET_KEY", "k8s-local-secret-key")
ALGORITHM = "HS256"

async def get_current_user_id(request: Request) -> uuid.UUID:
    auth_header = request.headers.get("Authorization")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    if not token:
        token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
        return uuid.UUID(user_id)
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="토큰 검증에 실패했습니다.")
