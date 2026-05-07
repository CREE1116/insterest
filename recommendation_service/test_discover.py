import asyncio
import uuid
import numpy as np
from app.services.intelligence import intel_service
from app.db.session import AsyncSessionLocal

async def test():
    print("Starting discovery test...")
    async with AsyncSessionLocal() as db:
        # Test 1: Direct discover call
        try:
            results = await intel_service.discover(db, user_id=None, limit=10)
            print(f"Discovery results (User: None): {results}")
        except Exception as e:
            print(f"Discovery test failed: {e}")

        # Test 2: SQL Fallback check
        from sqlalchemy import text
        try:
            stmt = text("SELECT id FROM upload.post WHERE is_deleted = FALSE ORDER BY created_at DESC LIMIT 5")
            res = await db.execute(stmt)
            fallback_ids = [row[0] for row in res.all()]
            print(f"SQL Fallback results: {fallback_ids}")
        except Exception as e:
            print(f"SQL Fallback failed: {e}")

if __name__ == "__main__":
    asyncio.run(test())
