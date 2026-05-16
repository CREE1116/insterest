import asyncio
from app.db.session import AsyncSessionLocal
from sqlalchemy import text
from app.ml.vector_store import vector_store

async def nuke_all():
    print("🧨 [Nuke] Initiating total wipe of the database and redis...")
    async with AsyncSessionLocal() as db:
        await db.execute(text("TRUNCATE TABLE interaction.likes CASCADE"))
        await db.execute(text("TRUNCATE TABLE search.post_vectors CASCADE"))
        await db.execute(text("TRUNCATE TABLE upload.post CASCADE"))
        await db.commit()
        print("✅ PostgreSQL (Posts, Likes, Vectors) completely truncated.")
        
    try:
        vector_store.r.flushall()
        print("✅ Redis completely flushed.")
    except Exception as e:
        print(f"⚠️ Redis flush error: {e}")

if __name__ == "__main__":
    asyncio.run(nuke_all())
