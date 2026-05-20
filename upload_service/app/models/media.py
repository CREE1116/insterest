from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.schema import MetaData, Table, Column
from sqlalchemy import String, Text, ForeignKey, Integer, DateTime, Boolean, JSON, Enum, LargeBinary
from datetime import datetime
import enum
import uuid
from typing import List, Optional
from app.core.config import settings

class Base(DeclarativeBase):
    metadata = MetaData(schema=settings.POSTGRES_SCHEMA)

class MediaType(str, enum.Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class ContentType(str, enum.Enum):
    PHOTO = "PHOTO"
    PHOTO_SOUND = "PHOTO_SOUND"
    VIDEO = "VIDEO"

# 중간 테이블: Post <-> Hashtag
post_hashtag = Table(
    "post_hashtag",
    Base.metadata,
    Column("post_id", ForeignKey("post.id"), primary_key=True),
    Column("hashtag_id", ForeignKey("hashtag.id"), primary_key=True)
)

class Content(Base):
    __tablename__ = "content"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(index=True) # 제작자
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType))

    # 메타데이터 (프롬프트, AI 생성 정보 등)
    metadata_info: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default={})

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    is_deleted: Mapped[bool] = mapped_column(default=False)
    is_ai: Mapped[bool] = mapped_column(default=False)

    # Relationships
    media_list: Mapped[List["Media"]] = relationship(back_populates="content", lazy="selectin")
    generation_meta: Mapped[Optional["GenerationMeta"]] = relationship(back_populates="content", lazy="selectin")
    posts: Mapped[List["Post"]] = relationship(back_populates="content", lazy="selectin")

class Post(Base):
    __tablename__ = "post"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    content_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("content.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(index=True) # 포스팅한 사람

    caption: Mapped[Optional[str]] = mapped_column(Text) # 포스트 내용

    like_count: Mapped[int] = mapped_column(default=0)
    view_count: Mapped[int] = mapped_column(default=0)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(onupdate=datetime.utcnow)

    is_deleted: Mapped[bool] = mapped_column(default=False)

    # Relationships
    content: Mapped["Content"] = relationship(back_populates="posts", lazy="selectin")
    hashtags: Mapped[List["Hashtag"]] = relationship(
        secondary=post_hashtag, back_populates="posts", lazy="selectin"
    )


class Media(Base):
    __tablename__ = "media"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    content_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("content.id"), index=True)

    type: Mapped[MediaType] = mapped_column(Enum(MediaType))
    url: Mapped[str] = mapped_column(String(512))
    
    duration: Mapped[Optional[int]] = mapped_column(Integer)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    metadata_info: Mapped[Optional[dict]] = mapped_column(JSON)

    content: Mapped[Optional["Content"]] = relationship(back_populates="media_list", lazy="selectin")

class Hashtag(Base):
    __tablename__ = "hashtag"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tag: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    posts: Mapped[List["Post"]] = relationship(
        secondary=post_hashtag, back_populates="hashtags", lazy="selectin"
    )

class GenerationMeta(Base):
    __tablename__ = "generation_meta"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("content.id"), unique=True)
    user_id: Mapped[uuid.UUID] = mapped_column(index=True)

    image_prompt: Mapped[Optional[str]] = mapped_column(Text)
    audio_prompt: Mapped[Optional[str]] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    content: Mapped["Content"] = relationship(back_populates="generation_meta", lazy="selectin")
