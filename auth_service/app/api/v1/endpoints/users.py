from typing import Any, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import User as UserSchema, UserUpdate
import uuid

router = APIRouter()

@router.get("/me", response_model=UserSchema)
def read_user_me(
    current_user: User = Depends(get_current_user),
) -> Any:
    """현재 로그인한 사용자의 정보를 반환합니다."""
    return current_user

@router.put("/me", response_model=UserSchema)
async def update_user_me(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """사용자 정보를 업데이트합니다."""
    if user_in.nickname:
        current_user.nickname = user_in.nickname
    if user_in.profile_image:
        current_user.profile_image = user_in.profile_image
    if user_in.password:
        from app.core.security import get_password_hash
        current_user.hashed_password = get_password_hash(user_in.password)
    
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user

@router.get("/batch", response_model=List[UserSchema])
async def get_users_batch(
    *,
    db: AsyncSession = Depends(get_db),
    user_ids: List[uuid.UUID] = Query(...),
) -> Any:
    """여러 유저의 정보를 한 번에 조회합니다. (피드 닉네임 표시용)"""
    result = await db.execute(select(User).where(User.id.in_(user_ids)))
    return result.scalars().all()
