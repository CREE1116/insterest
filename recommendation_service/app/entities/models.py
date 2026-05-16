from sqlalchemy import Column, String, DateTime, JSON, LargeBinary, func, Index
from sqlalchemy.dialects.postgresql import UUID
from app.db.session import Base
import uuid

class PostVector(Base):
    __tablename__ = "post_vectors"
    __table_args__ = {"schema": "search"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id = Column(UUID(as_uuid=True), nullable=False, unique=True)
    
    # Pre-trained LLM Vectors
    # Stored as binary for efficiency
    caption_vector = Column(LargeBinary, nullable=False)   # 768-dim
    hashtag_vector = Column(LargeBinary, nullable=True)    # 768-dim
    image_vector = Column(LargeBinary, nullable=True)      # 512-dim (CLIP)
    
    # Metadata for filtering and debugging
    content_text = Column(JSON, nullable=True) # { "caption": "...", "tags": "..." }
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Index for fast post_id lookups
    __table_args__ = (
        Index("idx_post_id", "post_id"),
        {"schema": "search"}
    )
