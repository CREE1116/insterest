import uuid
from sqlalchemy import String, Column, DateTime, func, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from app.db.session import Base
from app.core.config import settings

class AuthUser(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "auth"}  # auth-service의 스키마

    id = Column(UUID(as_uuid=True), primary_key=True)
    nickname = Column(String)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    __table_args__ = {"schema": settings.POSTGRES_SCHEMA}

    user_id = Column(UUID(as_uuid=True), primary_key=True) # auth-service의 user_id와 동일
    nickname = Column(String, index=True, nullable=True)
    profile_image = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Collection(Base):
    __tablename__ = "collections"
    __table_args__ = {"schema": settings.POSTGRES_SCHEMA}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CollectionItem(Base):
    __tablename__ = "collection_items"
    __table_args__ = {"schema": settings.POSTGRES_SCHEMA}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.POSTGRES_SCHEMA}.collections.id", ondelete="CASCADE"), nullable=False)
    post_id = Column(UUID(as_uuid=True), nullable=False) # upload-service의 post_id
    created_at = Column(DateTime(timezone=True), server_default=func.now())
