import uuid
from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from jose import jwt, JWTError

from app.core import security
from app.core.config import settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import UserCreate, User as UserSchema, Token
from app.services.email_service import send_verification_email
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/register", response_model=UserSchema)
async def register_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: UserCreate
) -> Any:
    """새로운 사용자를 등록합니다."""
    # 이메일 중복 확인
    result = await db.execute(select(User).filter(User.email == user_in.email))
    user = result.scalars().first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="이미 등록된 이메일입니다.",
        )
    
    # 사용자 생성 (ID 명시적 할당)
    db_obj = User(
        id=uuid.uuid4(),
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        nickname=user_in.nickname,
        is_active=True,
        is_verified=False,
    )
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    
    # 이메일 인증 토큰 생성 및 발송
    verify_token = security.create_access_token(str(db_obj.id), expires_delta=timedelta(hours=24))
    await send_verification_email(db_obj.email, verify_token)
    
    return db_obj

@router.post("/login", response_model=Token)
async def login_access_token(
    response: Response,
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """OAuth2 호환 로그인 토큰 발급."""
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalars().first()
    
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="이메일 또는 비밀번호가 틀렸습니다.")
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="비활성화된 계정입니다.")
        
    access_token = security.create_access_token(subject=str(user.id))
    refresh_token = security.create_refresh_token(subject=str(user.id))
    
    # 쿠키 설정 (경로를 /로 명시하여 모든 서비스에서 공유 가능하게 함)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=False,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        path="/",
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }

@router.get("/verify-email")
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """이메일 인증 링크 확인."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=400, detail="잘못된 토큰이거나 만료되었습니다.")
    
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    user.is_verified = True
    await db.commit()
    
    return {"msg": "이메일 인증이 완료되었습니다."}

@router.post("/validate", response_model=UserSchema)
async def validate_token(
    current_user: User = Depends(get_current_user)
):
    """타 서비스에서 호출할 수 있는 토큰 검증 엔드포인트."""
    return current_user

@router.post("/logout")
async def logout(response: Response):
    """로그아웃: 쿠키를 삭제합니다."""
    response.delete_cookie(
        key="access_token",
        path="/",
        httponly=True, # 로그인 시 설정과 동일하게 맞춤
        samesite="lax",
    )
    # 구형 브라우저 대응을 위해 한 번 더 수동 설정
    response.set_cookie(
        key="access_token",
        value="",
        max_age=0,
        expires=0,
        path="/",
        httponly=True,
        samesite="lax",
    )
    return {"msg": "Successfully logged out"}
