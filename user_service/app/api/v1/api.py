from fastapi import APIRouter
from app.api.v1.endpoints import profile, collection

api_router = APIRouter()
api_router.include_router(profile.router, prefix="/users", tags=["users"])
api_router.include_router(collection.router, prefix="/collections", tags=["collections"])
