from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.services.intelligence import intel_service
from typing import List, Optional
import uuid

router = APIRouter()

@router.get("/recommend", response_model=List[uuid.UUID])
async def discovery_feed(
    user_id: Optional[str] = Query(None), 
    query: Optional[str] = Query(None),
    skip: int = Query(0),
    limit: int = Query(50),
    db: AsyncSession = Depends(get_db)
):
    # '0'이거나 유효하지 않은 UUID인 경우 None으로 처리하여 Cold Start 유도
    processed_user_id = None
    if user_id and user_id != "0" and user_id != "undefined":
        try:
            processed_user_id = uuid.UUID(user_id)
        except (ValueError, AttributeError):
            processed_user_id = None

    results = await intel_service.discover(db, user_id=processed_user_id, query_text=query, skip=skip, limit=limit)
    return [r["id"] for r in results]

@router.get("", response_model=List[uuid.UUID])
async def discovery_search(
    query: str = Query(...),
    user_id: Optional[uuid.UUID] = Query(None),
    skip: int = Query(0),
    limit: int = Query(50),
    db: AsyncSession = Depends(get_db)
):
    """General Search: 순수하게 검색어(Query)에만 집중하여 결과를 반환합니다."""
    results = await intel_service.discover(
        db, 
        user_id=user_id, 
        query_text=query, 
        skip=skip, 
        limit=limit, 
        use_personalization=False  # 검색어 타워만 사용
    )
    return [r["id"] for r in results]

@router.post("/sync")
async def sync_index(
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger manual re-indexing and projection for vectors (Backfill)
    """
    await intel_service.backfill_all_posts(db)
    return {"status": "sync_completed"}

@router.get("/metrics")
async def get_discovery_metrics(
    db: AsyncSession = Depends(get_db)
):
    """
    정량적 성능 지표(NDCG, Recall)를 측정하여 반환합니다. (Postman 확인용)
    """
    metrics = await intel_service.evaluate_offline(db)
    if not metrics:
        return {"message": "No data available for evaluation"}
    return metrics

@router.post("/train")
async def trigger_training(
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger manual model training asynchronously
    """
    await intel_service.train_daily_async(db)
    return {"status": "training_started"}
