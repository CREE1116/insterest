from fastapi import APIRouter
from app.api.v1.endpoints import auth, social

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(social.router, prefix="/social", tags=["social"])
# users router removed
