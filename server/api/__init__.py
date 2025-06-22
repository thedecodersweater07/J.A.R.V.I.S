"""
JARVIS API Package
Contains API routes and handlers for the JARVIS server
"""

# API package
# __all__ verwijderd voor compatibiliteit en om warnings te voorkomen

from fastapi import APIRouter

# Create API router
router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "JARVIS API",
        "version": "0.1.0"
    }

@router.get("/version")
async def get_version():
    """Get API version"""
    return {"version": "0.1.0"}

# Import and include other API routes
try:
    from . import ai, auth, system
    
    # Include AI routes
    router.include_router(ai.router, prefix="/ai", tags=["AI"])
    
    # Include auth routes
    router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    
    # Include system routes
    router.include_router(system.router, prefix="/system", tags=["System"])
    
except ImportError as e:
    import logging
    logging.warning(f"Could not load all API modules: {e}")
