import asyncio
from app.db.session import AsyncSessionLocal
from sqlalchemy import text

async def check():
    async with AsyncSessionLocal() as db:
        res = await db.execute(text("SELECT id, caption FROM upload.post WHERE caption LIKE '%토끼%'"))
        rabbits = res.all()
        print(f"🐰 토끼 포스트 개수: {len(rabbits)}")
        for r in rabbits:
            print(f"  {r[1]} (ID: {r[0]})")

if __name__ == "__main__":
    asyncio.run(check())
