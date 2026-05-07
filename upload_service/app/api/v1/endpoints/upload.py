from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.models.media import Content, Post, ContentType
import uuid
import json
from typing import List, Optional

router = APIRouter()

@router.post("/")
async def create_upload(
    *,
    db: AsyncSession = Depends(get_db),
    image_file: Optional[UploadFile] = File(None),
    sound_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    author_id: uuid.UUID = Form(...),
    body: str = Form(...),
    hashtags: Optional[str] = Form(None), # JSON list or comma separated
    prompt: Optional[str] = Form(None),
):
    # 1. Validation: 소리는 반드시 이미지와 함께 있어야 함
    if sound_file and not image_file:
        raise HTTPException(status_code=400, detail="Sound file must be uploaded with an image file.")

    # 2. Determine Content Type
    content_type = ContentType.PHOTO
    if video_file:
        content_type = ContentType.VIDEO
    elif image_file and sound_file:
        content_type = ContentType.PHOTO_SOUND
    elif image_file:
        content_type = ContentType.PHOTO
    else:
        raise HTTPException(status_code=400, detail="At least one media file (image or video) is required.")

    # 3. Save Content Metadata
    new_content = Content(
        content_type=content_type,
        author_id=author_id,
        prompt=prompt
    )
    db.add(new_content)
    await db.flush() # ID 생성을 위해 flush

    # 4. Handle Hashtags
    tag_list = []
    if hashtags:
        try:
            tag_list = json.loads(hashtags)
        except json.JSONDecodeError:
            tag_list = [tag.strip() for tag in hashtags.split(",")]

    # 5. Save Post Metadata
    new_post = Post(
        content_id=new_content.id,
        body=body,
        hashtags=tag_list
    )
    db.add(new_post)
    await db.commit()
    await db.refresh(new_post)
    
    # 6. Return Result (실제 파일 저장 로직은 서비스 레이어에서 처리 가능)
    return {
        "status": "success",
        "content_id": str(new_content.id),
        "post_id": str(new_post.id),
        "content_type": content_type
    }
