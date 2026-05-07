from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import MetaData
from app.core.config import settings

engine = create_async_engine(settings.async_database_url, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# 모든 테이블이 'comment' 스키마를 사용하도록 설정
metadata = MetaData(schema=settings.POSTGRES_SCHEMA)
Base = declarative_base(metadata=metadata)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
