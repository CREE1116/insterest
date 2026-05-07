import asyncio
import uuid
import numpy as np
import torch
from app.services.intelligence import intel_service
from app.db.session import AsyncSessionLocal

async def test():
    print("Starting discovery test for SEARCH...")
    async with AsyncSessionLocal() as db:
        # Test 1: Search for '떡볶이'
        try:
            results_1 = await intel_service.discover(db, user_id=None, query_text="떡볶이", limit=5)
            print(f"Search results for '떡볶이': {results_1}")
        except Exception as e:
            print(f"Search '떡볶이' failed: {e}")

        # Test 2: Search for '강아지'
        try:
            results_2 = await intel_service.discover(db, user_id=None, query_text="강아지", limit=5)
            print(f"Search results for '강아지': {results_2}")
        except Exception as e:
            print(f"Search '강아지' failed: {e}")
            
        if results_1 == results_2:
            print("⚠️ WARNING: Search results are IDENTICAL for different queries!")
        else:
            print("✅ SUCCESS: Search results are DIFFERENT.")

if __name__ == "__main__":
    asyncio.run(test())
