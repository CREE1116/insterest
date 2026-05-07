from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, desc
from app.db.session import get_db
from app.api.deps import get_current_user_id, get_optional_user_id
from app.models.interaction import Like, Save, View, WatchTime, Ignore
from app.schemas.interaction import WatchTimeCreate, LikeToggleResponse, SaveToggleResponse, InteractionCount
import uuid
from typing import List, Optional
import logging

logger = logging.getLogger("interaction_api")

router = APIRouter()

@router.post("/{post_id}/like", response_model=LikeToggleResponse)
async def toggle_like(post_id: uuid.UUID, db: AsyncSession = Depends(get_db), user_id: uuid.UUID = Depends(get_current_user_id)):
    logger.info(f"👍 Toggle Like request: user={user_id}, post={post_id}")
    # Use select(Like.post_id, Like.user_id) to avoid fetching non-existent 'id' column
    query = select(Like).where(Like.post_id == post_id, Like.user_id == user_id)
    result = await db.execute(query)
    existing = result.scalars().first()
    
    if existing:
        logger.info(f"🗑 Removing like for post {post_id}")
        await db.delete(existing); await db.commit()
        return {"liked": False}
    
    logger.info(f"➕ Adding like for post {post_id}")
    db.add(Like(post_id=post_id, user_id=user_id))
    await db.commit()
    return {"liked": True}

@router.post("/{post_id}/save", response_model=SaveToggleResponse)
async def toggle_save(post_id: uuid.UUID, db: AsyncSession = Depends(get_db), user_id: uuid.UUID = Depends(get_current_user_id)):
    query = select(Save).where(Save.post_id == post_id, Save.user_id == user_id)
    result = await db.execute(query)
    existing = result.scalars().first()
    
    if existing:
        await db.delete(existing); await db.commit()
        return {"saved": False}
    db.add(Save(post_id=post_id, user_id=user_id))
    await db.commit()
    return {"saved": True}

@router.get("/me/liked", response_model=List[uuid.UUID])
async def get_my_liked_posts(
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    logger.info(f"🔍 Fetching liked posts for user: {user_id}")
    # Explicitly select only post_id to avoid any 'id' column issues
    result = await db.execute(select(Like.post_id).where(Like.user_id == user_id))
    liked_ids = result.scalars().all()
    logger.info(f"✅ Found {len(liked_ids)} liked posts for user {user_id}")
    return liked_ids

@router.post("/{post_id}/view")
async def record_view(post_id: uuid.UUID, db: AsyncSession = Depends(get_db), user_id: uuid.UUID = Depends(get_current_user_id)):
    db.add(View(post_id=post_id, user_id=user_id))
    await db.commit()
    return {"status": "recorded"}

@router.post("/watch-time")
async def record_watch_time(wt: WatchTimeCreate, db: AsyncSession = Depends(get_db), user_id: uuid.UUID = Depends(get_current_user_id)):
    db.add(WatchTime(post_id=wt.post_id, user_id=user_id, duration_seconds=wt.duration_seconds))
    await db.commit()
    return {"status": "recorded"}

@router.get("/stats/{post_id}", response_model=dict)
async def get_post_stats(
    post_id: uuid.UUID, 
    db: AsyncSession = Depends(get_db),
    user_id: Optional[uuid.UUID] = Depends(get_optional_user_id)
):
    l_count = await db.scalar(select(func.count(Like.user_id)).where(Like.post_id == post_id)) or 0
    s_count = await db.scalar(select(func.count(Save.user_id)).where(Save.post_id == post_id)) or 0
    v_count = await db.scalar(select(func.count(View.id)).where(View.post_id == post_id)) or 0
    avg_wt = await db.scalar(select(func.avg(WatchTime.duration_seconds)).where(WatchTime.post_id == post_id)) or 0.0
    
    is_liked = False
    if user_id:
        liked_exists = await db.scalar(select(Like.post_id).where(Like.post_id == post_id, Like.user_id == user_id).limit(1))
        is_liked = True if liked_exists else False

    return {
        "post_id": str(post_id),
        "likes": l_count,
        "saves": s_count,
        "views": v_count,
        "avg_watch_time": float(avg_wt),
        "is_liked": is_liked
    }

@router.get("/recent")
async def get_recent_interactions(
    limit: int = Query(200),
    db: AsyncSession = Depends(get_db)
):
    stmt = select(Like).order_by(desc(Like.created_at)).limit(limit)
    result = await db.execute(stmt)
    likes = result.scalars().all()
    
    return [{"user_id": str(l.user_id), "post_id": str(l.post_id), "type": "like"} for l in likes]

@router.get("/user/{user_id}")
async def get_user_interaction_history(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    likes = (await db.execute(select(Like.post_id).where(Like.user_id == user_id))).scalars().all()
    saves = (await db.execute(select(Save.post_id).where(Save.user_id == user_id))).scalars().all()
    
    return {
        "likes": [{"post_id": str(pid)} for pid in likes],
        "saves": [{"post_id": str(pid)} for pid in saves]
    }

@router.get("/all")
async def get_all_interactions(db: AsyncSession = Depends(get_db)):
    likes = (await db.execute(select(Like.user_id, Like.post_id))).all()
    saves = (await db.execute(select(Save.user_id, Save.post_id))).all()
    views = (await db.execute(select(View.user_id, View.post_id))).all()
    wts = (await db.execute(select(WatchTime.user_id, WatchTime.post_id, WatchTime.duration_seconds))).all()
    
    return {
        "likes": [{"user_id": str(r[0]), "post_id": str(r[1])} for r in likes],
        "saves": [{"user_id": str(r[0]), "post_id": str(r[1])} for r in saves],
        "views": [{"user_id": str(r[0]), "post_id": str(r[1])} for r in views],
        "watch_times": [{"user_id": str(r[0]), "post_id": str(r[1]), "duration": r[2]} for r in wts]
    }
