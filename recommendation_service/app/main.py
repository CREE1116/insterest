from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.api.v1.api import api_router
from app.core.config import settings
from app.db.session import engine, Base
from app.ml.vector_store import vector_store
from app.entities.models import PostVector # Ensure models are imported for metadata
import logging
import os
import asyncio

# Suppress parallelism warning from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "message": "Internal Server Error"},
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Unified Intelligence Service...")
    
    # 0. Initialize Database
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.POSTGRES_SCHEMA}"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database Schema Ready")

    # 1. Initialize Redis Vector Index
    vector_store.create_index()
    logger.info("✅ Discovery Engine Ready")

    # 2. Trigger Initial Backfill (Async)
    from app.services.intelligence import intel_service
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        await intel_service.backfill_all_posts(db)
    logger.info("🔄 Initial Backfill Task Started")

    # 3. Start Kafka Consumer (Async)
    from app.services.kafka_consumer import consume_post_created
    asyncio.create_task(consume_post_created())
    logger.info("📡 Kafka Consumer Task Started")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "recommendation-service"}

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)
