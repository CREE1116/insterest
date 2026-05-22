import os
import asyncio
import aiofiles
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db, get_current_user_id
from app.models.media import Media, MediaType
import uuid
from typing import Optional
import logging

from sqlalchemy import text
import shutil
from app.db.session import engine
from app.core.config import settings

router = APIRouter(redirect_slashes=True)
logger = logging.getLogger(__name__)

@router.post("/reset")
async def reset_database():
    """DB 테이블 초기화 및 업로드 파일 삭제"""
    async with engine.begin() as conn:
        # 정확한 테이블 목록 (최신 스키마 기준)
        tables = ["generation_meta", "post_hashtag", "media", "post", "content", "hashtag"]
        for table in tables:
            try:
                await conn.execute(text(f"TRUNCATE TABLE {settings.POSTGRES_SCHEMA}.{table} RESTART IDENTITY CASCADE"))
            except Exception as e:
                logger.warning(f"Failed to truncate {table}: {str(e)}")
    
    # 업로드 파일 삭제
    if os.path.exists(settings.UPLOAD_DIR):
        for filename in os.listdir(settings.UPLOAD_DIR):
            if filename == ".gitkeep": continue
            file_path = os.path.join(settings.UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    await asyncio.to_thread(os.unlink, file_path)
                elif os.path.isdir(file_path):
                    await asyncio.to_thread(shutil.rmtree, file_path)
            except: pass
                
    return {"status": "success", "message": "Database and files reset successfully"}

from app.models.media import Content, Media, MediaType, ContentType

from pydub import AudioSegment
import io

@router.post("/upload-combined", include_in_schema=True)
@router.post("/upload-combined/", include_in_schema=False)
async def upload_combined(
    *,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    image_file: UploadFile = File(...), # 이미지는 필수
    audio_file: Optional[UploadFile] = File(None), # 소리는 선택
    start_time: Optional[float] = Form(0),
    end_time: Optional[float] = Form(10),
):
    """이미지와 소리를 한 번에 업로드하여 하나의 Content로 묶어 보관함에 저장합니다. 오디오는 서버에서 10초로 트리밍됩니다."""
    logger.info(f"Combined upload started for user {current_user_id}")
    try:
        # 1. Content 생성 (그릇)
        content_id = uuid.uuid4()
        new_content = Content(
            id=content_id,
            user_id=current_user_id,
            content_type=ContentType.PHOTO_SOUND if audio_file else ContentType.PHOTO,
            metadata_info={
                "ai_generated": False,
                "start_time": 0, # 서버에서 자르므로 저장된 파일은 항상 0초부터 시작
                "end_time": 10
            }
        )
        db.add(new_content)
        await db.flush()
        logger.info(f"Created content {content_id}")

        # 2. 이미지 파일 저장 및 Media 생성
        image_ext = image_file.filename.split(".")[-1] if "." in image_file.filename else "jpg"
        image_name = f"{uuid.uuid4()}.{image_ext}"
        image_path = os.path.join(settings.UPLOAD_DIR, image_name)
        
        async with aiofiles.open(image_path, "wb") as out_file:
            await out_file.write(await image_file.read())
            
        new_image = Media(
            user_id=current_user_id,
            content_id=content_id,
            type=MediaType.IMAGE,
            url=f"/uploads/{image_name}",
            metadata_info={"filename": image_file.filename}
        )
        db.add(new_image)
        logger.info(f"Saved image to {image_path}")

        # 3. 오디오 파일 저장 및 트리밍 (있는 경우)
        if audio_file:
            audio_content = await audio_file.read()
            if len(audio_content) > 0:
                logger.info(f"Processing audio file: {audio_file.filename} ({len(audio_content)} bytes)")
                audio_format = audio_file.filename.split(".")[-1].lower() if "." in audio_file.filename else "mp3"
                if audio_format == "m4a": audio_format = "mp4" # pydub m4a 지원 대응
                
                try:
                    # pydub으로 오디오 로드 및 트리밍 (asyncio.to_thread)
                    audio_name = f"{uuid.uuid4()}.mp3"
                    audio_path = os.path.join(settings.UPLOAD_DIR, audio_name)
                    
                    def _process_audio_sync():
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_content), format=audio_format)
                        start_ms = int(start_time * 1000)
                        end_ms = start_ms + 10000 # 10초
                        trimmed_audio = audio_segment[start_ms:end_ms]
                        trimmed_audio.export(audio_path, format="mp3", bitrate="128k")
                        
                    await asyncio.to_thread(_process_audio_sync)
                        
                    new_audio = Media(
                        user_id=current_user_id,
                        content_id=content_id,
                        type=MediaType.AUDIO,
                        url=f"/uploads/{audio_name}",
                        duration=10,
                        metadata_info={
                            "filename": f"{audio_file.filename.split('.')[0]}_trimmed.mp3",
                            "original_start_time": start_time
                        }
                    )
                    db.add(new_audio)
                    logger.info(f"Saved trimmed audio to {audio_path}")
                except Exception as se:
                    logger.error(f"Failed to decode/process audio: {str(se)}")
                    # 오디오 처리에 실패해도 이미지는 저장하고 싶을 수 있으나, 
                    # combined upload의 의도상 400 에러를 반환하는 것이 맞을 수 있음.
                    # 여기서는 기존 로직대로 에러를 발생시킴.
                    raise HTTPException(status_code=400, detail=f"소리 파일을 해독할 수 없습니다: {str(se)}")
            else:
                logger.warning("Empty audio file received")

        # get_db()에서 commit을 해주지만 명시적으로 commit 하고 싶은 경우
        # await db.commit() 
        # return {"content_id": str(content_id), "status": "success"}
        
        # 명시적으로 커밋하여 다음 요청(POST /content)에서 데이터를 조회할 수 있도록 함
        await db.commit()
        logger.info(f"Successfully committed and completed combined upload for content {content_id}")
        return {"content_id": str(content_id), "status": "success"}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Combined upload failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/upload")
async def upload_media(
    *,
    db: AsyncSession = Depends(get_db),
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    file: UploadFile = File(...),
    media_type: str = Form(...),
    duration: Optional[int] = Form(None),
):
    try:
        # 1. 미디어 타입 처리
        try:
            m_type = MediaType(media_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid media type: {media_type}")

        # 2. 파일명 생성 및 저장
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "bin"
        file_name = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(settings.UPLOAD_DIR, file_name)

        # 3. 비동기 파일 쓰기
        async with aiofiles.open(file_path, "wb") as out_file:
            content_data = await file.read()
            await out_file.write(content_data)

        # 4. Content 생성 (이 미디어를 담을 그릇)
        c_type = ContentType.PHOTO
        if m_type == MediaType.AUDIO: c_type = ContentType.PHOTO_SOUND # 임시
        elif m_type == MediaType.VIDEO: c_type = ContentType.VIDEO

        new_content = Content(
            user_id=current_user_id,
            content_type=c_type,
            metadata_info={"ai_generated": False}
        )
        db.add(new_content)
        await db.flush()

        # 5. Media 생성 및 연결
        new_media = Media(
            user_id=current_user_id,
            content_id=new_content.id,
            type=m_type,
            url=f"/uploads/{file_name}",
            duration=duration,
            metadata_info={"filename": file.filename, "content_type": file.content_type}
        )

        db.add(new_media)
        await db.commit()
        await db.refresh(new_media)

        return {
            "media_id": str(new_media.id),
            "content_id": str(new_content.id), # 컨텐츠 ID 반환
            "url": new_media.url,
            "type": new_media.type
        }
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during upload: {str(e)}")
