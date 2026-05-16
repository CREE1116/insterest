import asyncio
from app.db.session import AsyncSessionLocal
from sqlalchemy import text

async def check():
    async with AsyncSessionLocal() as db:
        print("🔍 [Real Data Check] Posts:")
        res = await db.execute(text("SELECT id, caption FROM upload.post ORDER BY created_at DESC LIMIT 15"))
        for row in res.all():
            print(f"  Post: {row[1]} (ID: {row[0]})")
            
        print("\n🔍 [Real Data Check] Likes:")
        res = await db.execute(text("SELECT user_id, post_id FROM interaction.likes ORDER BY created_at DESC LIMIT 15"))
        for row in res.all():
            print(f"  Like: User {row[0]} -> Post {row[1]}")

if __name__ == "__main__":
    asyncio.run(check())
