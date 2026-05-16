import json
import logging
from aiokafka import AIOKafkaProducer
from app.core.config import settings

logger = logging.getLogger(__name__)

class KafkaProducer:
    def __init__(self):
        self.producer = None

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self.producer.start()
        logger.info("✅ Kafka Producer started.")

    async def stop(self):
        if self.producer:
            await self.producer.stop()
            logger.info("✅ Kafka Producer stopped.")

    async def send_event(self, topic: str, data: dict):
        if not self.producer:
            await self.start()
        try:
            await self.producer.send_and_wait(topic, data)
            logger.info(f"📤 Sent Kafka event to {topic}: {data}")
        except Exception as e:
            logger.error(f"❌ Failed to send Kafka event: {e}")

    async def send_post_created(self, post_id, content_id, user_id):
        """추천 시스템 연동을 위한 포스트 생성 이벤트 발행"""
        data = {
            "event_type": "post.created",
            "post_id": str(post_id),
            "content_id": str(content_id),
            "user_id": str(user_id)
        }
        await self.send_event("recommendation.post_created", data)

    async def send_post_updated(self, post_id, content_id, user_id):
        """추천 시스템 연동을 위한 포스트 수정 이벤트 발행 (재인덱싱 트리거)"""
        data = {
            "event_type": "post.updated",
            "post_id": str(post_id),
            "content_id": str(content_id),
            "user_id": str(user_id)
        }
        await self.send_event("recommendation.post_updated", data)

kafka_producer = KafkaProducer()
