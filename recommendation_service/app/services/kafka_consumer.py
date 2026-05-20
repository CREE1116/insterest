import asyncio
import json
import logging
import uuid
import httpx
import numpy as np
import torch
from aiokafka import AIOKafkaConsumer
from app.core.config import settings
from app.services.intelligence import intel_service
from app.db.session import AsyncSessionLocal
from app.ml.nlp import nlp_embedder

logger = logging.getLogger(__name__)

async def consume_post_created():
    """Kafka에서 포스트 생성/수정 메시지를 소비하여 CLIP 512-dim 통합 벡터로 인덱싱"""
    logger.info(f"Initializing CLIP-unified Kafka Consumer... Servers: {settings.KAFKA_BOOTSTRAP_SERVERS}")
    
    consumer = AIOKafkaConsumer(
        "recommendation.post_created",
        "recommendation.post_updated",
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="recommendation-group",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    
    try:
        await consumer.start()
        logger.info("✅ MLP-based Multi-modal Kafka Consumer started.")
        
        async for msg in consumer:
            try:
                data = msg.value
                post_id = uuid.UUID(data["post_id"])
                
                async with httpx.AsyncClient(timeout=20.0) as client:
                    # 1. 포스트 상세 정보 조회
                    res = await client.get(f"{settings.UPLOAD_SERVICE_URL}/api/v1/upload/content/{post_id}")
                    if res.status_code != 200: continue
                    post_data = res.json()
                    
                    # 2. 무드 벡터 생성 (CLIP 512-dim)
                    caption = post_data.get("caption") or ""
                    gen_meta = post_data.get("content", {}).get("generation_meta", {}) or {}
                    image_prompt = gen_meta.get("image_prompt") or ""
                    music_prompt = gen_meta.get("music_prompt") or ""
                    # 무드 = 캡션 + 이미지 프롬프트 + 음악 프롬프트 통합 텍스트
                    mood_text = f"{caption} {image_prompt} {music_prompt}".strip()

                    hashtags = post_data.get("hashtags", [])
                    hashtag_texts = [h["tag"] if isinstance(h, dict) else str(h) for h in hashtags]

                    # (A) 무드 텍스트 벡터 (CLIP 512-dim)
                    c_vec = nlp_embedder.embed_text_clip(mood_text).cpu().numpy() if mood_text else np.zeros(512, dtype=np.float32)

                    # (B) 해시태그 평균 벡터 (CLIP 512-dim, DB 저장용)
                    if hashtag_texts:
                        h_vecs = nlp_embedder.embed_batch_clip(hashtag_texts)
                        h_vec = torch.mean(h_vecs, dim=0).cpu().numpy()
                    else:
                        h_vec = np.zeros(512, dtype=np.float32)

                    # (C) 이미지 벡터 (CLIP 512-dim)
                    i_vec = np.zeros(512, dtype=np.float32)
                    media_list = post_data.get("content", {}).get("media_list", [])
                    image_url = next((m["url"] for m in media_list if m["type"] == "image"), None)

                    if image_url:
                        try:
                            full_image_url = f"{settings.UPLOAD_SERVICE_URL}{image_url}" if image_url.startswith("/") else image_url
                            img_res = await client.get(full_image_url)
                            if img_res.status_code == 200:
                                i_vec = nlp_embedder.embed_image(img_res.content).cpu().numpy()
                        except Exception as img_e:
                            logger.error(f"❌ Image fetch error: {img_e}")

                    # 3. CLIP 512-dim 통합 인덱싱
                    async with AsyncSessionLocal() as db:
                        await intel_service.index_post(
                            db, post_id, c_vec, h_vec, i_vec,
                            metadata={"caption": caption, "image_prompt": image_prompt, "music_prompt": music_prompt}
                        )
                        logger.info(f"✅ CLIP-unified indexing complete for {post_id}")
                        
            except Exception as e:
                logger.error(f"❌ Processing error: {e}", exc_info=True)
                
    except Exception as e:
        logger.error(f"❌ Kafka Consumer error: {e}")
    finally:
        await consumer.stop()
