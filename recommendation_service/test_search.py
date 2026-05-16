import asyncio
from app.services.intelligence import intel_service
from app.db.session import AsyncSessionLocal

async def test():
    async with AsyncSessionLocal() as db:
        print("🔍 검색 테스트: '고양이'")
        results = await intel_service.discover(db, query_text="고양이", limit=5)
        for r in results:
            print(f"  - 점수: {r['score']:.4f} | 내용: {r['caption']}")
            
        print("\n🔍 검색 테스트: '토끼'")
        results = await intel_service.discover(db, query_text="토끼", limit=5)
        for r in results:
            print(f"  - 점수: {r['score']:.4f} | 내용: {r['caption']}")

if __name__ == "__main__":
    asyncio.run(test())
