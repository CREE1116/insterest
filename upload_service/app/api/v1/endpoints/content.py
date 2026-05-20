from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, text
from sqlalchemy.orm import selectinload
from app.api.deps import get_db, get_current_user_id, get_current_user_id_optional
from app.models.media import Content, Post, Media, MediaType, ContentType, Hashtag, GenerationMeta
from app.services.kafka_producer import kafka_producer
import uuid
import asyncio
import httpx
import logging
from fastapi import BackgroundTasks
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Schemas ---
class MediaRead(BaseModel):
    id: uuid.UUID
    type: str
    url: str
    model_config = ConfigDict(from_attributes=True)

class HashtagRead(BaseModel):
    tag: str
    model_config = ConfigDict(from_attributes=True)

class ContentRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    content_type: str
    created_at: datetime
    media_list: List[MediaRead] = []
    metadata_info: Optional[Dict[str, Any]] = None # 추가된 메타데이터 필드
    
    # 레거시 지원 및 편의를 위한 프롬프트 필드 (metadata_info에서 추출 가능)
    image_prompt: Optional[str] = None
    audio_prompt: Optional[str] = None
    is_ai: bool = False
    
    model_config = ConfigDict(from_attributes=True)

class PostRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    content_id: uuid.UUID
    caption: Optional[str] = None
    created_at: datetime
    like_count: int
    view_count: int
    is_liked: bool = False
    content: ContentRead
    hashtags: List[HashtagRead] = []
    model_config = ConfigDict(from_attributes=True)

class PostCreate(BaseModel):
    content_id: Optional[uuid.UUID] = None # 일반 업로드시 필수, AI 생성시 선택
    metadata_info: Optional[Dict[str, Any]] = None # AI 생성시 직접 전달
    caption: Optional[str] = None
    hashtags: List[str] = []

class PostUpdate(BaseModel):
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None

# --- Helper Functions ---
def enrich_content_read(c: Content) -> ContentRead:
    """Content 객체에서 metadata_info와 generation_meta를 활용해 ContentRead 스키마를 채웁니다."""
    # media_list를 포함하여 명시적으로 필드를 채웁니다.
    media_data = []
    if c.media_list:
        for m in c.media_list:
            # MediaType Enum 처리
            m_type = m.type.value if hasattr(m.type, 'value') else str(m.type)
            media_data.append(MediaRead(id=m.id, type=m_type, url=m.url))

    c_read = ContentRead(
        id=c.id,
        user_id=c.user_id,
        content_type=c.content_type.value if hasattr(c.content_type, 'value') else str(c.content_type),
        created_at=c.created_at,
        media_list=media_data,
        metadata_info=c.metadata_info,
        is_ai=c.is_ai
    )
    
    # 1. metadata_info에서 정보 추출
    if c.metadata_info:
        c_read.image_prompt = c.metadata_info.get("image_prompt")
        c_read.audio_prompt = c.metadata_info.get("music_prompt") or c.metadata_info.get("audio_prompt")
    
    # 2. generation_meta가 있다면 우선순위 적용 (또는 보완)
    if hasattr(c, 'generation_meta') and c.generation_meta:
        c_read.image_prompt = c_read.image_prompt or c.generation_meta.image_prompt
        c_read.audio_prompt = c_read.audio_prompt or c.generation_meta.audio_prompt
        
    return c_read

# --- Endpoints ---

# 1. 게시물 생성 (Post)
@router.post("", response_model=dict, include_in_schema=True)
@router.post("/", response_model=dict, include_in_schema=False)
async def create_post(
    *,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    post_in: PostCreate,
    background_tasks: BackgroundTasks
):
    content = None
    
    # 1. content_id가 제공된 경우 (일반 업로드 혹은 이미 생성된 AI 컨텐츠)
    if post_in.content_id:
        result = await db.execute(
            select(Content).where(Content.id == post_in.content_id)
        )
        content = result.scalar_one_or_none()
    
    # 2. content_id가 없고 metadata_info가 있는 경우 (즉시 포스팅하는 AI 컨텐츠)
    elif post_in.metadata_info:
        content = Content(
            user_id=current_user_id,
            content_type=ContentType.PHOTO_SOUND, # AI 기본값
            metadata_info=post_in.metadata_info,
            is_ai=True
        )
        db.add(content)
        await db.flush()
        
        # Media 객체 생성 (URL이 있는 경우에만 추가)
        media_items = []
        if post_in.metadata_info.get("image_url"):
            media_items.append(Media(
                user_id=current_user_id,
                content_id=content.id,
                type=MediaType.IMAGE,
                url=post_in.metadata_info.get("image_url")
            ))
        if post_in.metadata_info.get("audio_url"):
            media_items.append(Media(
                user_id=current_user_id,
                content_id=content.id,
                type=MediaType.AUDIO,
                url=post_in.metadata_info.get("audio_url")
            ))
        elif post_in.metadata_info.get("music_url"): # music_url 필드도 호환성 위해 체크
             media_items.append(Media(
                user_id=current_user_id,
                content_id=content.id,
                type=MediaType.AUDIO,
                url=post_in.metadata_info.get("music_url")
            ))
            
        if media_items:
            db.add_all(media_items)
        
        # GenerationMeta 생성
        gen_meta = GenerationMeta(
            content_id=content.id,
            user_id=current_user_id,
            image_prompt=post_in.metadata_info.get("prompt") or post_in.metadata_info.get("image_prompt"),
            audio_prompt=post_in.metadata_info.get("music_prompt") or post_in.metadata_info.get("audio_prompt"),
            model="StableDiffusion + MusicGen"
        )
        db.add(gen_meta)

    if not content:
        raise HTTPException(status_code=400, detail="Invalid content reference")

    # 3. Post 생성 (user_id 누락 방지)
    new_post = Post(
        content_id=content.id,
        user_id=current_user_id,
        caption=post_in.caption,
        hashtags=[] # Initialize to prevent lazy loading error in async
    )
    db.add(new_post)
    await db.flush()

    # 4. 해시태그 처리
    if post_in.hashtags:
        for tag_name in post_in.hashtags:
            stmt = select(Hashtag).where(Hashtag.tag == tag_name)
            result = await db.execute(stmt)
            hashtag = result.scalar_one_or_none()
            if not hashtag:
                hashtag = Hashtag(tag=tag_name)
                db.add(hashtag)
                await db.flush()
            
            # 중간 테이블 관계 추가
            new_post.hashtags.append(hashtag)

    await db.commit()
    
    # 5. 추천 시스템 연동 (Kafka 전송)
    try:
        # 추천 엔진을 위한 이벤트 발행
        background_tasks.add_task(kafka_producer.send_post_created, new_post.id, content.id, current_user_id)
        logger.info(f"Kafka event queued for post: {new_post.id}")
    except Exception as e:
        logger.error(f"Kafka error: {e}")

    return {"status": "success", "post_id": str(new_post.id)}

# 2. 전체 피드 조회 (Post 기준)
@router.get("/feed", response_model=List[PostRead])
async def get_feed(
    *,
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[uuid.UUID] = Depends(get_current_user_id_optional),
    skip: int = 0,
    limit: int = 50
):
    result = await db.execute(
        select(Post)
        .options(
            selectinload(Post.hashtags),
            selectinload(Post.content).selectinload(Content.media_list),
            selectinload(Post.content).selectinload(Content.generation_meta)
        )
        .where(Post.is_deleted == False)
        .order_by(desc(Post.created_at))
        .offset(skip)
        .limit(limit)
    )
    posts = result.scalars().all()

    # 1. 좋아요 여부 (is_liked) 및 좋아요 수 (like_count) 조회
    liked_post_ids = set()
    like_counts = {}
    
    post_ids = [p.id for p in posts]
    if post_ids:
        # 좋아요 여부 확인
        if current_user_id:
            liked_res = await db.execute(
                text("SELECT post_id FROM interaction.likes WHERE user_id = :uid AND post_id = ANY(:pids)")
                .bindparams(uid=current_user_id, pids=post_ids)
            )
            liked_post_ids = {row[0] for row in liked_res.all()}
        
        # 실제 좋아요 수 집계 (interaction.likes 테이블 기준)
        count_res = await db.execute(
            text("SELECT post_id, COUNT(*) FROM interaction.likes WHERE post_id = ANY(:pids) GROUP BY post_id")
            .bindparams(pids=post_ids)
        )
        like_counts = {row[0]: row[1] for row in count_res.all()}

    response_data = []
    for p in posts:
        p_read = PostRead.model_validate(p)
        p_read.content = enrich_content_read(p.content)
        p_read.is_liked = p.id in liked_post_ids
        p_read.like_count = like_counts.get(p.id, 0) # 실시간 집계값 적용
        response_data.append(p_read)

    return response_data

@router.get("/posts/all", response_model=List[PostRead])
async def get_all_posts(
    db: AsyncSession = Depends(get_db)
):
    """모든 활성 포스트를 조회합니다. (내부 서비스용)"""
    result = await db.execute(
        select(Post)
        .options(
            selectinload(Post.hashtags),
            selectinload(Post.content).selectinload(Content.media_list),
            selectinload(Post.content).selectinload(Content.generation_meta)
        )
        .where(Post.is_deleted == False)
        .order_by(desc(Post.created_at))
    )
    posts = result.scalars().all()
    
    response_data = []
    for p in posts:
        p_read = PostRead.model_validate(p)
        p_read.content = enrich_content_read(p.content)
        response_data.append(p_read)
        
    return response_data

class PostBatchRequest(BaseModel):
    post_ids: List[uuid.UUID]

@router.get("/hashtags")
async def get_hashtags(
    q: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """전체 해시태그 목록을 반환합니다. q 파라미터로 prefix 필터링 가능."""
    from sqlalchemy import func
    stmt = (
        select(Hashtag.tag, func.count(Hashtag.id).label("count"))
        .join(Hashtag.posts)
        .where(Post.is_deleted == False)
        .group_by(Hashtag.tag)
        .order_by(desc("count"))
        .limit(limit)
    )
    if q:
        stmt = stmt.where(Hashtag.tag.ilike(f"{q}%"))
    result = await db.execute(stmt)
    return [{"tag": row.tag, "count": row.count} for row in result.all()]


@router.get("/by-hashtag/{tag}", response_model=List[PostRead])
async def get_posts_by_hashtag(
    tag: str,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[uuid.UUID] = Depends(get_current_user_id_optional)
):
    """특정 해시태그가 달린 포스트를 정확히 검색합니다."""
    result = await db.execute(
        select(Post)
        .options(
            selectinload(Post.hashtags),
            selectinload(Post.content).selectinload(Content.media_list),
            selectinload(Post.content).selectinload(Content.generation_meta)
        )
        .join(Post.hashtags)
        .where(Hashtag.tag == tag.lstrip("#"), Post.is_deleted == False)
        .order_by(desc(Post.created_at))
        .offset(skip)
        .limit(limit)
    )
    posts = result.scalars().all()
    if not posts:
        return []

    post_ids = [p.id for p in posts]
    liked_post_ids: set = set()
    like_counts: dict = {}
    if current_user_id:
        liked_res = await db.execute(
            text("SELECT post_id FROM interaction.likes WHERE user_id = :uid AND post_id = ANY(:pids)")
            .bindparams(uid=current_user_id, pids=post_ids)
        )
        liked_post_ids = {row[0] for row in liked_res.all()}
    count_res = await db.execute(
        text("SELECT post_id, COUNT(*) FROM interaction.likes WHERE post_id = ANY(:pids) GROUP BY post_id")
        .bindparams(pids=post_ids)
    )
    like_counts = {row[0]: row[1] for row in count_res.all()}

    response_data = []
    for p in posts:
        p_read = PostRead.model_validate(p)
        p_read.content = enrich_content_read(p.content)
        p_read.is_liked = p.id in liked_post_ids
        p_read.like_count = like_counts.get(p.id, 0)
        response_data.append(p_read)
    return response_data


@router.post("/batch", response_model=List[PostRead])
async def get_posts_batch(
    request: PostBatchRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[uuid.UUID] = Depends(get_current_user_id_optional)
):
    """여러 포스트의 상세 정보를 한 번에 조회합니다."""
    if not request.post_ids:
        return []
        
    result = await db.execute(
        select(Post)
        .options(
            selectinload(Post.hashtags),
            selectinload(Post.content).selectinload(Content.media_list),
            selectinload(Post.content).selectinload(Content.generation_meta)
        )
        .where(Post.id.in_(request.post_ids))
        .where(Post.is_deleted == False)
    )
    posts = result.scalars().all()

    # 1. 좋아요 정보 조회 (is_liked 및 like_count)
    liked_post_ids = set()
    like_counts = {}

    pids = [p.id for p in posts]
    if pids:
        # 좋아요 여부
        if current_user_id:
            liked_res = await db.execute(
                text("SELECT post_id FROM interaction.likes WHERE user_id = :uid AND post_id = ANY(:pids)")
                .bindparams(uid=current_user_id, pids=pids)
            )
            liked_post_ids = {row[0] for row in liked_res.all()}

        # 실제 좋아요 수 집계
        count_res = await db.execute(
            text("SELECT post_id, COUNT(*) FROM interaction.likes WHERE post_id = ANY(:pids) GROUP BY post_id")
            .bindparams(pids=pids)
        )
        like_counts = {row[0]: row[1] for row in count_res.all()}

    # 요청받은 ID 순서대로 정렬 (추천 엔진의 순서 유지)
    post_map = {p.id: p for p in posts}
    ordered_posts = [post_map[pid] for pid in request.post_ids if pid in post_map]

    response_data = []
    for p in ordered_posts:
        p_read = PostRead.model_validate(p)
        p_read.content = enrich_content_read(p.content)
        p_read.is_liked = p.id in liked_post_ids
        p_read.like_count = like_counts.get(p.id, 0)
        response_data.append(p_read)

    return response_data
@router.get("/users/{user_id}/posts", response_model=List[PostRead])
async def get_user_posts(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[uuid.UUID] = Depends(get_current_user_id_optional),
    skip: int = 0,
    limit: int = 500
):
    """특정 사용자의 포스트를 조회합니다."""
    result = await db.execute(
        select(Post)
        .options(
            selectinload(Post.hashtags),
            selectinload(Post.content).selectinload(Content.media_list),
            selectinload(Post.content).selectinload(Content.generation_meta)
        )
        .where(Post.user_id == user_id)
        .where(Post.is_deleted == False)
        .order_by(desc(Post.created_at))
        .offset(skip)
        .limit(limit)
    )
    posts = result.scalars().all()

    # 1. 좋아요 정보 조회 (is_liked 및 like_count)
    liked_post_ids = set()
    like_counts = {}
    
    pids = [p.id for p in posts]
    if pids:
        # 좋아요 여부
        if current_user_id:
            liked_res = await db.execute(
                text("SELECT post_id FROM interaction.likes WHERE user_id = :uid AND post_id = ANY(:pids)")
                .bindparams(uid=current_user_id, pids=pids)
            )
            liked_post_ids = {row[0] for row in liked_res.all()}
        
        # 실제 좋아요 수 집계
        count_res = await db.execute(
            text("SELECT post_id, COUNT(*) FROM interaction.likes WHERE post_id = ANY(:pids) GROUP BY post_id")
            .bindparams(pids=pids)
        )
        like_counts = {row[0]: row[1] for row in count_res.all()}
    
    response_data = []
    for p in posts:
        p_read = PostRead.model_validate(p)
        p_read.content = enrich_content_read(p.content)
        p_read.is_liked = p.id in liked_post_ids
        p_read.like_count = like_counts.get(p.id, 0)
        response_data.append(p_read)
        
    return response_data

# 3. 특정 게시물 상세 조회
@router.get("/{post_id}", response_model=PostRead)
async def get_post(
    post_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: Optional[uuid.UUID] = Depends(get_current_user_id_optional)
):
    result = await db.execute(
        select(Post)
        .options(
            selectinload(Post.hashtags),
            selectinload(Post.content).selectinload(Content.media_list),
            selectinload(Post.content).selectinload(Content.generation_meta)
        )
        .where(Post.id == post_id)
        .where(Post.is_deleted == False)
    )
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="Post not found")
        
    is_liked = False
    like_count = 0
    
    # 1. 좋아요 여부 및 총 좋아요 수 조회
    if current_user_id:
        liked_res = await db.execute(
            text("SELECT 1 FROM interaction.likes WHERE user_id = :uid AND post_id = :pid")
            .bindparams(uid=current_user_id, pid=post_id)
        )
        is_liked = liked_res.scalar() is not None
        
    count_res = await db.execute(
        text("SELECT COUNT(*) FROM interaction.likes WHERE post_id = :pid")
        .bindparams(pid=post_id)
    )
    like_count = count_res.scalar() or 0

    p_read = PostRead.model_validate(p)
    p_read.content = enrich_content_read(p.content)
    p_read.is_liked = is_liked
    p_read.like_count = like_count
    return p_read

# 4. 게시물 삭제
@router.delete("/{post_id}")
async def delete_post(
    post_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user_id)
):
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this post")
        
    post.is_deleted = True
    await db.commit()
    return {"status": "success"}

# 5. 게시물 수정 (Edit)
@router.patch("/{post_id}", response_model=PostRead)
async def update_post(
    post_id: uuid.UUID,
    post_update: PostUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user_id)
):
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.hashtags), selectinload(Post.content).selectinload(Content.media_list))
        .where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this post")
        
    if post_update.caption is not None:
        post.caption = post_update.caption
        
    if post_update.hashtags is not None:
        # 기존 해시태그 제거 (중간 테이블 관계 끊기)
        post.hashtags = []
        await db.flush()
        
        # 새 해시태그 추가
        for tag_name in post_update.hashtags:
            stmt = select(Hashtag).where(Hashtag.tag == tag_name)
            res = await db.execute(stmt)
            hashtag = res.scalar_one_or_none()
            if not hashtag:
                hashtag = Hashtag(tag=tag_name)
                db.add(hashtag)
                await db.flush()
            post.hashtags.append(hashtag)
            
    await db.commit()
    await db.refresh(post)
    
    # 추천 시스템 재동기화 트리거
    try:
        background_tasks.add_task(kafka_producer.send_post_updated, post.id, post.content_id, current_user_id)
        logger.info(f"Kafka update event queued for post: {post.id}")
    except Exception as e:
        logger.error(f"Kafka error during update: {e}")

    # 리턴 스키마 구성
    p_read = PostRead.model_validate(post)
    p_read.content = enrich_content_read(post.content)
    return p_read
