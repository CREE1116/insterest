from fastapi import APIRouter
from app.api.v1.endpoints import discovery, intelligence

api_router = APIRouter()
api_router.include_router(discovery.router, prefix="/discovery", tags=["discovery"])
api_router.include_router(discovery.router, prefix="/search", tags=["search"])
api_router.include_router(intelligence.router, prefix="/intel", tags=["intelligence"])
