import uuid
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, func, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class InteractionBase:
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    post_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Like(Base, InteractionBase):
    __tablename__ = "likes"
    # likes has composite PK: (user_id, post_id)
    user_id = Column(UUID(as_uuid=True), primary_key=True, index=True, nullable=False)
    post_id = Column(UUID(as_uuid=True), primary_key=True, index=True, nullable=False)
    __table_args__ = (UniqueConstraint('user_id', 'post_id', name='_user_post_like_uc'), {"schema": "interaction"})

class Save(Base, InteractionBase):
    __tablename__ = "saves"
    # saves has composite PK: (user_id, post_id)
    user_id = Column(UUID(as_uuid=True), primary_key=True, index=True, nullable=False)
    post_id = Column(UUID(as_uuid=True), primary_key=True, index=True, nullable=False)
    __table_args__ = (UniqueConstraint('user_id', 'post_id', name='_user_post_save_uc'), {"schema": "interaction"})

class View(Base, InteractionBase):
    __tablename__ = "views"
    __table_args__ = {"schema": "interaction"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

class WatchTime(Base):
    __tablename__ = "watch_times"
    __table_args__ = {"schema": "interaction"}
    user_id = Column(UUID(as_uuid=True), primary_key=True, index=True, nullable=False)
    post_id = Column(UUID(as_uuid=True), primary_key=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), primary_key=True, server_default=func.now())
    duration_seconds = Column(Float, nullable=False)

class Ignore(Base, InteractionBase):
    __tablename__ = "ignores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    __table_args__ = (UniqueConstraint('user_id', 'post_id', name='_user_post_ignore_uc'), {"schema": "interaction"})
