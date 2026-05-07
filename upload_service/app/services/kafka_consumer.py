import asyncio
import json
import logging
import uuid
import httpx
import os
import aiofiles
from aiokafka import AIOKafkaConsumer
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import SessionLocal
from app.models.media import Content, Media, GenerationMeta, MediaType, ContentType
from app.core.config import settings

logger = logging.getLogger(__name__)

async def fetch_file_data(url: str) -> bytes:
    """Generation Service로부터 실제 파일 데이터를 가져옵니다."""
    # 만약 url이 '/outputs/...'로 시작한다면, generation-service의 8002 포트에서 가져와야 함
    full_url = f"{settings.GENERATION_SERVICE_URL}{url}" if url.startswith("/") else url
    logger.info(f"Fetching file data from: {full_url}")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        try:
            response = await client.get(full_url)
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to fetch file from {full_url}: Status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching file from {full_url}: {e}")
            return None

async def consume_generation_completed():
    """Kafka에서 생성 완료 메시지를 소비하여 파일 저장 및 DB 등록"""
    logger.info(f"Initializing Kafka Consumer... Servers: {settings.KAFKA_BOOTSTRAP_SERVERS}")
    
    # 소비자 생성 (v6 그룹으로 업데이트하여 새로운 컨슈머가 처음부터 읽도록 유도)
    consumer = AIOKafkaConsumer(
        "generation.completed",
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="upload-service-group-v6", 
        auto_offset_reset='earliest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    
    try:
        await consumer.start()
        logger.info("✅ Kafka Consumer started. Monitoring 'generation.completed'...")
    except Exception as e:
        logger.error(f"❌ Failed to start Kafka Consumer: {e}")
        return

    try:
        async for msg in consumer:
            data = msg.value
            logger.info(f"📥 Received AI content event: {data.get('title')}")
            
            image_url = data.get("image_url")
            music_url = data.get("music_url")
            
            # 1. 파일 다운로드 (generation-service -> upload-service)
            # image_url: /outputs/abc.png -> http://generation-service:8002/outputs/abc.png
            image_data, music_data = await asyncio.gather(
                fetch_file_data(image_url),
                fetch_file_data(music_url)
            )

            if not image_data or not music_data:
                logger.error("❌ Failed to download media data, skipping.")
                continue

            try:
                # 2. 로컬 파일 시스템(/uploads)에 저장하여 영구 보관
                image_filename = f"ai_img_{uuid.uuid4()}.png"
                music_filename = f"ai_aud_{uuid.uuid4()}.wav"
                
                image_path = os.path.join(settings.UPLOAD_DIR, image_filename)
                music_path = os.path.join(settings.UPLOAD_DIR, music_filename)
                
                async with aiofiles.open(image_path, "wb") as f:
                    await f.write(image_data)
                async with aiofiles.open(music_path, "wb") as f:
                    await f.write(music_data)

                async with SessionLocal() as db:
                    user_id_str = data.get("user_id")
                    user_id = uuid.UUID(user_id_str) if isinstance(user_id_str, str) else user_id_str
                    content_id = uuid.uuid4()
                    
                    # 3. Content 생성 (그릇)
                    content = Content(
                        id=content_id,
                        user_id=user_id,
                        content_type=ContentType.PHOTO_SOUND,
                        is_ai=True,
                        metadata_info={
                            "title": data.get("title"),
                            "mood": data.get("mood"),
                            "ai_generated": True
                        }
                    )
                    db.add(content)
                    
                    # 4. Media 생성 (DB에는 경로만 저장, file_data는 제거)
                    db.add(Media(
                        user_id=user_id,
                        content_id=content_id,
                        type=MediaType.IMAGE,
                        url=f"/uploads/{image_filename}",
                        metadata_info={"original_filename": data.get("title")}
                    ))
                    db.add(Media(
                        user_id=user_id,
                        content_id=content_id,
                        type=MediaType.AUDIO,
                        url=f"/uploads/{music_filename}",
                        metadata_info={"original_filename": data.get("title")}
                    ))
                    
                    # 5. GenerationMeta 생성 (프롬프트 기록)
                    db.add(GenerationMeta(
                        content_id=content_id,
                        user_id=user_id,
                        image_prompt=data.get("image_prompt"),
                        audio_prompt=data.get("music_prompt"),
                        model="MusicGen/Flux"
                    ))
                    
                    await db.commit()
                    logger.info(f"✅ Successfully persisted AI content to /uploads and DB: {content_id}")
                    
            except Exception as e:
                logger.error(f"❌ Error during persistence: {e}")
                
    except Exception as e:
        logger.error(f"❌ Consumer loop error: {e}")
    finally:
        await consumer.stop()
