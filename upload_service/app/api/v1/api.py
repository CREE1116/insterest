from fastapi import APIRouter
from app.api.v1.endpoints import media, content

api_router = APIRouter()
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(content.router, prefix="/content", tags=["content"])
