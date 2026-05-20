from sqlalchemy import Column, String, DateTime, JSON, LargeBinary, func, Index
from sqlalchemy.dialects.postgresql import UUID
from app.db.session import Base
import uuid

class PostVector(Base):
    __tablename__ = "post_vectors"
    __table_args__ = {"schema": "search"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id = Column(UUID(as_uuid=True), nullable=False, unique=True)
    
    # CLIP 512-dim vectors (text & image in unified CLIP space)
    caption_vector = Column(LargeBinary, nullable=False)   # 512-dim CLIP text (mood = caption+image_prompt+music_prompt)
    hashtag_vector = Column(LargeBinary, nullable=True)    # 512-dim CLIP text (hashtags)
    image_vector = Column(LargeBinary, nullable=True)      # 512-dim CLIP image

    # Metadata: { "caption": "...", "image_prompt": "...", "music_prompt": "..." }
    content_text = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Index for fast post_id lookups
    __table_args__ = (
        Index("idx_post_id", "post_id"),
        {"schema": "search"}
    )
