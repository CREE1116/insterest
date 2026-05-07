from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.db.session import get_db
from app.api.deps import get_current_user_id
from app.models.models import Comment
from pydantic import BaseModel
import uuid
import logging
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter()

class CommentCreate(BaseModel):
    post_id: uuid.UUID
    content: str

@router.post("/")
@router.post("")
async def create_comment(
    comment_in: CommentCreate,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    new_comment = Comment(
        post_id=comment_in.post_id,
        user_id=user_id,
        content=comment_in.content
    )
    db.add(new_comment)
    await db.commit()
    await db.refresh(new_comment)
    return new_comment

@router.get("/{post_id}")
async def list_comments(
    post_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Comment)
        .where(Comment.post_id == post_id)
        .order_by(desc(Comment.created_at))
    )
    comments = result.scalars().all()
    # 유저 정보는 프론트엔드에서 batch API로 처리하므로 id만 반환
    return comments

class CommentBatchRequest(BaseModel):
    post_ids: List[uuid.UUID]

@router.post("/batch")
async def list_comments_batch(
    request: CommentBatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """여러 포스트 ID에 대해 댓글 목록을 맵 형태로 반환합니다."""
    result = await db.execute(
        select(Comment)
        .where(Comment.post_id.in_(request.post_ids))
        .order_by(desc(Comment.created_at))
    )
    comments = result.scalars().all()
    
    # post_id별로 그룹화
    res_map = {}
    for c in comments:
        pid = str(c.post_id)
        if pid not in res_map:
            res_map[pid] = []
        res_map[pid].append(c)
        
    return res_map
