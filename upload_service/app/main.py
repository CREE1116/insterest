from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
import os
import shutil
import asyncio
import logging

from app.api.v1.api import api_router
from app.core.config import settings
from app.models.media import Base
from app.db.session import engine
from app.services.kafka_consumer import consume_generation_completed

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("upload_service")

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
        content={"detail": str(exc), "message": "Internal Server Error in Upload Service"},
    )

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 1. 업로드 디렉토리 생성 (절대 경로 보장 및 로깅 강화)
upload_path = os.path.abspath(settings.UPLOAD_DIR)
try:
    os.makedirs(upload_path, exist_ok=True)
    logger.info(f"📁 Upload directory initialized at: {upload_path}")
    # 쓰기 권한 테스트
    test_file = os.path.join(upload_path, ".write_test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    logger.info(f"✅ Write permission verified for: {upload_path}")
except Exception as e:
    logger.error(f"❌ Failed to initialize upload directory {upload_path}: {e}")

# 2. 정적 파일 서빙 설정
app.mount("/uploads", StaticFiles(directory=upload_path), name="uploads")

@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting upload-service backend...")
    
    # Debug: Print all routes
    for route in app.routes:
        logger.info(f"Route: {route.path} [Methods: {getattr(route, 'methods', 'N/A')}]")

    # 스키마 생성 및 테이블 생성 로직
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.POSTGRES_SCHEMA}"))
        
        # 1. Base table creation
        await conn.run_sync(Base.metadata.create_all)
        
        # 2. Migration: Add 'is_ai' column if missing (Safety check)
        try:
            await conn.execute(text(f"ALTER TABLE {settings.POSTGRES_SCHEMA}.content ADD COLUMN IF NOT EXISTS is_ai BOOLEAN DEFAULT FALSE"))
            logger.info("✅ Migration: Ensured 'is_ai' column exists.")
        except Exception as mig_e:
            logger.warning(f"⚠️ Migration warning (is_ai): {mig_e}")
    
    # Kafka Consumer를 비동기 태스크로 실행 (예외 처리 포함)
    async def run_consumer():
        logger.info("📡 Starting Kafka Consumer task...")
        try:
            await consume_generation_completed()
        except Exception as e:
            logger.error(f"💀 Kafka Consumer task CRASHED: {e}")

    asyncio.create_task(run_consumer())
    logger.info("✅ Startup event completed.")

app.include_router(api_router, prefix=f"{settings.API_V1_STR}/upload")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 404 Debugging Catch-all
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path_name: str):
    logger.warning(f"⚠️ 404 Catch-all: {request.method} {request.url.path} (Path name: {path_name})")
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Not Found",
            "requested_path": request.url.path,
            "method": request.method,
            "message": "This is a custom 404 from Catch-all"
        }
    )





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
