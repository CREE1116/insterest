import logging
from fastapi import APIRouter, Depends, Query, BackgroundTasks, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db, AsyncSessionLocal
from app.services.intelligence import intel_service
from typing import List, Optional

logger = logging.getLogger(__name__)

import uuid

router = APIRouter()

@router.post("/image", response_model=List[uuid.UUID])
async def discovery_image_search(
    file: UploadFile = File(...),
    user_id: Optional[str] = Query(None),
    skip: int = Query(0),
    limit: int = Query(50),
    db: AsyncSession = Depends(get_db)
):
    """Image Search: 이미지를 CLIP으로 벡터화하고 투영하여 유사한 아이템을 검색합니다."""
    # user_id 파싱 처리 (유효하지 않은 UUID는 None 처리)
    processed_user_id = None
    if user_id and user_id != "0" and user_id != "undefined":
        try:
            processed_user_id = uuid.UUID(user_id)
        except (ValueError, AttributeError):
            processed_user_id = None

    image_bytes = await file.read()
    results = await intel_service.discover_by_image(
        db,
        image_bytes=image_bytes,
        user_id=processed_user_id,
        skip=skip,
        limit=limit
    )
    return [r["id"] for r in results]

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
    background_tasks: BackgroundTasks
):
    """
    Trigger manual model training.
    독립적인 DB 세션을 생성하여 HTTP 요청 세션과 수명주기를 분리합니다.
    """
    async def run_training():
        async with AsyncSessionLocal() as session:
            await intel_service.train_discovery(session)

    background_tasks.add_task(run_training)
    return {"status": "training_started"}

@router.get("/benchmark/synthetic")
async def run_synthetic_benchmark():
    """
    합성 데이터셋을 실시간 Pillow로 그려서 CLIP/SBERT의 Retrieval 성능을 평가합니다.
    """
    from app.services.benchmark import benchmark_service
    return await benchmark_service.run_synthetic_benchmark()

@router.post("/benchmark/custom")
async def run_custom_benchmark(
    file: UploadFile = File(...)
):
    """
    사용자가 업로드한 ZIP 파일(dataset.json + 이미지들)을 사용하여 검색 정확도를 평가합니다.
    """
    from app.services.benchmark import benchmark_service
    contents = await file.read()
    return await benchmark_service.run_custom_benchmark(contents)

@router.get("/benchmark/animal")
async def run_animal_benchmark(
    db: AsyncSession = Depends(get_db)
):
    """
    동물 이미지 데이터셋을 DB에 실시간 주입하고 다대다 교차 매칭(Recall, NDCG) 정확도를 평가합니다.
    """
    from app.services.benchmark import benchmark_service
    return await benchmark_service.run_animal_db_benchmark(db)

@router.post("/benchmark/seed-gemini")
async def seed_gemini_personas_interactions(
    db: AsyncSession = Depends(get_db)
):
    """
    제미나이 API를 활용하여 가상 유저 페르소나 및 고품질 시뮬레이션 좋아요를 DB에 일괄 주입합니다.
    """
    from app.services.seed_gemini_interactions import seed_gemini_interactions_service
    return await seed_gemini_interactions_service(db)

@router.get("/debug/users")
async def get_debug_active_users(
    db: AsyncSession = Depends(get_db)
):
    """
    추천 시뮬레이터에서 사용할 실활동 테스트용 사용자 목록(좋아요 누적 유저)을 반환합니다.
    """
    from sqlalchemy import text
    try:
        # 최근 좋아요를 가장 많이 누른 상위 10명의 유저 정보
        res = await db.execute(text(
            "SELECT user_id, COUNT(*) as cnt FROM interaction.likes GROUP BY user_id ORDER BY cnt DESC LIMIT 10"
        ))
        rows = res.all()
        # 간단한 닉네임과 함께 유저 목록 반환
        users_list = []
        for r in rows:
            uid_str = str(r[0])
            users_list.append({
                "id": uid_str,
                "likes_count": r[1],
                "label": f"활동 유저 (좋아요 {r[1]}개) - {uid_str[:8]}"
            })
        return users_list
    except Exception as e:
        return []

