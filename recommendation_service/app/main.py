from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from app.api.v1.api import api_router
from app.core.config import settings
from app.db.session import engine, Base
from app.ml.vector_store import vector_store
from app.entities.models import PostVector  # noqa: F401 — needed for metadata
import logging
import os
import asyncio

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "message": "Internal Server Error"},
    )


# Serve benchmark CIFAR images: /benchmark/cifar100/{class}_{idx}.jpg
_CIFAR_IMG_DIR = "/data/cifar_images"
os.makedirs(_CIFAR_IMG_DIR, exist_ok=True)
app.mount("/benchmark/cifar100", StaticFiles(directory=_CIFAR_IMG_DIR), name="benchmark_cifar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _scheduled_daily_training():
    """24시간마다 UserTower 재학습."""
    from app.services.intelligence import intel_service
    from app.db.session import AsyncSessionLocal
    while True:
        await asyncio.sleep(24 * 3600)
        logger.info("⏰ Starting scheduled daily training...")
        try:
            async with AsyncSessionLocal() as db:
                await intel_service.train_discovery(db)
            logger.info("✅ Scheduled daily training complete.")
        except Exception as e:
            logger.error(f"❌ Scheduled training failed: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Unified Intelligence Service...")

    # 0. Initialize Database
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.POSTGRES_SCHEMA}"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database Schema Ready")

    # 1. Initialize Redis Vector Index (재시작 시 기존 인덱스·데이터 유지)
    vector_store.create_index()
    logger.info("✅ Redis Index Ready")

    # 2. 스마트 Backfill: DB 포스트 수 > Redis 벡터 수일 때만 실행
    from app.services.intelligence import intel_service
    from app.db.session import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as db:
            res = await db.execute(text("SELECT COUNT(*) FROM search.post_vectors"))
            db_count = res.scalar() or 0
        redis_count = vector_store.count()
        if redis_count < db_count:
            logger.info(f"🔄 Redis ({redis_count}) < DB ({db_count}) — backfilling...")
            async with AsyncSessionLocal() as db:
                asyncio.create_task(intel_service.backfill_all_posts(db))
        else:
            logger.info(f"✅ Redis index up to date ({redis_count} vectors).")
    except Exception as e:
        logger.warning(f"⚠️ Startup backfill check failed: {e}")

    # 3. 일일 자동 재학습 스케줄러
    asyncio.create_task(_scheduled_daily_training())
    logger.info("⏰ Daily training scheduler started (runs every 24h).")

    # 4. Kafka Consumer
    from app.services.kafka_consumer import consume_post_created
    asyncio.create_task(consume_post_created())
    logger.info("📡 Kafka Consumer Task Started")

    # 5. Benchmark image directory — create if missing
    os.makedirs("/data/cifar_images", exist_ok=True)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "recommendation-service"}


app.include_router(api_router, prefix=settings.API_V1_STR)
