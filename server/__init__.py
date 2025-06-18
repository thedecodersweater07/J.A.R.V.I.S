"""
JARVIS Server Backend
Provides secure API endpoints for UI-AI communication
"""

import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="JARVIS Server",
    description="Backend API for JARVIS AI Assistant",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def include_router_safe(module_path: str, prefix: str = ""):
    """Safely include a router if the module exists"""
    try:
        module_name, router_name = module_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[router_name])
        router = getattr(module, router_name, None)
        if router:
            app.include_router(router, prefix=prefix)
            logger.info(f"Successfully included router: {module_path}")
            return True
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not include router {module_path}: {e}")
    return False

# Include available API routes
try:
    from .api import router as api_router
    app.include_router(api_router, prefix="/api")
    logger.info("Included API router")
except ImportError as e:
    logger.warning(f"Could not include API router: {e}")

# Include WebSocket routes if available
if include_router_safe("server.websocket.router", "/ws"):
    logger.info("Included WebSocket router")

# Include other available routes
routers = [
    ("server.auth.router", "/auth"),
    ("server.ai.router", "/ai"),
    ("server.system.router", "/system"),
]

for module_path, prefix in routers:
    include_router_safe(module_path, prefix)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "JARVIS server is running"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to JARVIS API",
        "docs": "/docs",
        "redoc": "/redoc"
    }
