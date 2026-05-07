from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.api.deps import get_current_user_id
from app.models.models import UserProfile, AuthUser
from app.schemas.schemas import UserProfileRead, UserProfileUpdate
from app.core.config import settings
import uuid
import os
import aiofiles
from typing import List

router = APIRouter()

import logging

logger = logging.getLogger(__name__)

@router.get("/me", response_model=UserProfileRead)
async def get_my_profile(
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    logger.info(f"🔍 Fetching profile for user_id: {user_id}")
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
    profile = result.scalar_one_or_none()
    
    if not profile or not profile.nickname:
        logger.info(f"ℹ️ Profile or nickname not found for {user_id}, attempting to sync from AuthUser")
        # 1. auth-service의 닉네임 조회
        auth_res = await db.execute(select(AuthUser.nickname).where(AuthUser.id == user_id))
        auth_nickname = auth_res.scalar_one_or_none()
        
        if not profile:
            profile = UserProfile(
                user_id=user_id, 
                nickname=auth_nickname or f"User_{str(user_id)[:6]}"
            )
            db.add(profile)
        else:
            profile.nickname = auth_nickname or profile.nickname or f"User_{str(user_id)[:6]}"
            
        await db.commit()
        await db.refresh(profile)
        logger.info(f"✅ Synced profile with nickname: {profile.nickname}")
    
    return profile

@router.put("/me", response_model=UserProfileRead)
async def update_my_profile(
    profile_in: UserProfileUpdate,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
    profile = result.scalar_one_or_none()
    if not profile:
        profile = UserProfile(user_id=user_id, nickname=profile_in.nickname)
        db.add(profile)
    else:
        if profile_in.nickname:
            profile.nickname = profile_in.nickname
        if profile_in.bio is not None:
            profile.bio = profile_in.bio
    await db.commit()
    await db.refresh(profile)
    return profile

@router.post("/me/image", response_model=UserProfileRead)
async def upload_profile_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_ext = file.filename.split(".")[-1]
    file_name = f"{user_id}_{uuid.uuid4().hex[:8]}.{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, file_name)
    
    async with aiofiles.open(file_path, "wb") as out_file:
        await out_file.write(await file.read())
        
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
    profile = result.scalar_one_or_none()
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
    
    profile.profile_image = f"/profiles/{file_name}"
    await db.commit()
    await db.refresh(profile)
    return profile

@router.get("/batch", response_model=List[UserProfileRead])
async def get_users_batch(
    user_ids: List[uuid.UUID] = Query(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. 기존 프로필 조회
    result = await db.execute(select(UserProfile).where(UserProfile.user_id.in_(user_ids)))
    existing_profiles = result.scalars().all()
    existing_user_ids = {p.user_id for p in existing_profiles}
    
    # 2. 누락된 유저들에 대해 기본 프로필 생성
    missing_user_ids = [uid for uid in user_ids if uid not in existing_user_ids]
    if missing_user_ids:
        # auth-service에서 닉네임 일괄 조회
        auth_res = await db.execute(select(AuthUser).where(AuthUser.id.in_(missing_user_ids)))
        auth_users = {u.id: u.nickname for u in auth_res.scalars().all()}
        
        for uid in missing_user_ids:
            new_profile = UserProfile(
                user_id=uid, 
                nickname=auth_users.get(uid) or f"User_{str(uid)[:6]}"
            )
            db.add(new_profile)
        await db.commit()
        
        # 전체 다시 조회
        result = await db.execute(select(UserProfile).where(UserProfile.user_id.in_(user_ids)))
        return result.scalars().all()
        
    return existing_profiles
