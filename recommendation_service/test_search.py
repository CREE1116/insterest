import asyncio
import uuid
from sqlalchemy import select
from app.services.intelligence import intel_service
from app.db.session import AsyncSessionLocal
from app.entities.models import PostVector

async def test():
    async with AsyncSessionLocal() as db:
        print("🔍 검색 테스트: '고양이'")
        results = await intel_service.discover(db, query_text="고양이", limit=5)
        if results:
            pids = [uuid.UUID(r["id"]) for r in results]
            res = await db.execute(select(PostVector).where(PostVector.post_id.in_(pids)))
            p_map = {str(pv.post_id): (pv.content_text.get("caption", "") if pv.content_text else "") for pv in res.scalars().all()}
            for r in results:
                caption = p_map.get(r["id"], "")
                print(f"  - 점수: {r['score']:.4f} | 내용: {caption}")
            
        print("\n🔍 검색 테스트: '토끼'")
        results = await intel_service.discover(db, query_text="토끼", limit=5)
        if results:
            pids = [uuid.UUID(r["id"]) for r in results]
            res = await db.execute(select(PostVector).where(PostVector.post_id.in_(pids)))
            p_map = {str(pv.post_id): (pv.content_text.get("caption", "") if pv.content_text else "") for pv in res.scalars().all()}
            for r in results:
                caption = p_map.get(r["id"], "")
                print(f"  - 점수: {r['score']:.4f} | 내용: {caption}")

if __name__ == "__main__":
    asyncio.run(test())
