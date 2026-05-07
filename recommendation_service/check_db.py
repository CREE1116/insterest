import asyncio
from sqlalchemy import text
from app.db.session import engine

async def check_db():
    async with engine.connect() as conn:
        try:
            res = await conn.execute(text("SELECT count(*) FROM upload.post"))
            count = res.scalar()
            print(f"Posts count: {count}")
            
            res = await conn.execute(text("SELECT count(*) FROM search.post_vectors"))
            count = res.scalar()
            print(f"Post vectors count: {count}")
            
            res = await conn.execute(text("SELECT id FROM upload.post LIMIT 5"))
            rows = res.all()
            print(f"Sample Post IDs: {[row[0] for row in rows]}")
        except Exception as e:
            print(f"Error checking DB: {e}")

if __name__ == "__main__":
    asyncio.run(check_db())
