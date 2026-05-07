import asyncio
from app.services.intelligence import intel_service
from app.db.session import AsyncSessionLocal
import logging

logging.basicConfig(level=logging.INFO)

async def trigger_training():
    print("🚀 Triggering manual training to initialize UserTower...")
    async with AsyncSessionLocal() as db:
        await intel_service.train_daily(db)
    print("✅ Training sequence completed.")

if __name__ == "__main__":
    asyncio.run(trigger_training())
