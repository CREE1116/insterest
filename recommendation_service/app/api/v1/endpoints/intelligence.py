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
    """
    모든 포스트를 다시 벡터화(Re-indexing)하도록 요청합니다.
    """
    try:
        await intel_service.backfill_all_posts(db)
        return {"status": "success", "message": "128-dim Backfill task started in background."}
    except Exception as e:
        logger.error(f"❌ Failed to trigger backfill: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_training_status():
    """
    현재 학습 상태 및 마지막 성공 시간을 반환합니다. (현재는 스켈레톤 상태)
    """
    return {
        "status": "ready",
        "last_trained": "Not tracked yet",
        "engine": "UnifiedDiscoveryEngine"
    }

@router.get("")
async def run_benchmark(db: AsyncSession = Depends(get_db)):
    """
    정량적 벤치마크(Recall, NDCG)를 실행하고 결과를 반환합니다.
    """
    try:
        results = await intel_service.run_quantitative_benchmark(db)
        return results
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
