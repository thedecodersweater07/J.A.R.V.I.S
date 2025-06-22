from fastapi import APIRouter
from datetime import datetime

api_router = APIRouter()

@api_router.get("/health", response_model=dict)
async def health_check() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }

# Voeg hier meer REST endpoints toe, bijv. authenticatie, user info, etc.
