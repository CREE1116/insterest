from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.services.intelligence import intel_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/train")
async def trigger_training(db: AsyncSession = Depends(get_db)):
    """
    추천 모델 학습을 수동으로 트리거합니다 (비동기 실행).
    """
    try:
        # 비동기로 학습 시작 (함수 내에서 asyncio.create_task 사용됨)
        await intel_service.train_daily_async(db)
        return {"status": "success", "message": "Unified Discovery Model training started in background."}
    except Exception as e:
        logger.error(f"❌ Failed to trigger training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backfill")
async def trigger_backfill(db: AsyncSession = Depends(get_db)):
    """모든 포스트를 CLIP 512-dim으로 재인덱싱합니다."""
    try:
        await intel_service.backfill_all_posts(db)
        return {"status": "success", "message": "CLIP 512-dim backfill complete."}
    except Exception as e:
        logger.error(f"❌ Failed to trigger backfill: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_training_status():
    """현재 학습 상태 및 Redis 인덱스 정보 반환."""
    from app.ml.vector_store import vector_store
    return {
        "status": "ready",
        "engine": "CLIP-unified UnifiedDiscoveryEngine",
        "redis_vector_count": vector_store.count(),
    }

@router.get("")
async def run_offline_eval(db: AsyncSession = Depends(get_db)):
    """오프라인 NDCG/Recall 평가를 실행합니다."""
    try:
        results = await intel_service.evaluate_offline(db)
        return results
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
