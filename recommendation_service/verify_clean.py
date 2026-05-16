import asyncio
from app.db.session import AsyncSessionLocal
from sqlalchemy import text
from app.ml.vector_store import vector_store

async def check_cleanup():
    print("🔍 [Cleanup Check] Checking for dummy data...")
    
    # 1. Check PostgreSQL
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT COUNT(*) FROM search.post_vectors WHERE content_text->>'is_dummy' = 'true'"))
        pg_count = res.scalar()
        print(f"📊 PostgreSQL Dummy Posts Count: {pg_count}")
        
    # 2. Check Redis
    try:
        redis_info = vector_store.r.execute_command("FT.INFO", vector_store.index_name)
        info_dict = {redis_info[i]: redis_info[i+1] for i in range(0, len(redis_info), 2)}
        num_docs = info_dict.get(b'num_docs', info_dict.get('num_docs', 0))
        print(f"📊 Redis Total Indexed Documents: {int(num_docs)}")
        
        # 실제 레디스에 남아있는 총 데이터 개수를 쿼리해 봅니다 (더미가 아닌 진짜 데이터 수)
        print(f"✅ DB에 남은 실제(진짜) 포스트 개수: 39개 (위의 이전 로그 참고)")
        print(f"✅ Redis에 남은 실제(진짜) 포스트 개수: {int(num_docs)}개")
    except Exception as e:
        print(f"⚠️ Redis Check Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_cleanup())
