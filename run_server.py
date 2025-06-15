import uvicorn
import logging
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger("jarvis-server")

# Add parent directory to path for module imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import app after path is set up
from server.app import app, initialize_jarvis_model, get_jarvis_model

# Initialize the model when the server starts
@app.on_event("startup")
async def startup_event():
    """Initialize components when the server starts"""
    try:
        # Initialize the Jarvis model
        model = initialize_jarvis_model()
        if not hasattr(model, 'initialized') or not model.initialized:
            logger.error("Failed to initialize Jarvis model")
            raise RuntimeError("Jarvis model initialization failed")
            
        logger.info("Jarvis model initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Don't raise here to allow the server to start in a degraded mode
        # The endpoints will handle the unavailability of components

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is available
        model = getattr(app.state, "jarvis_model", None)
        if not model or not hasattr(model, 'initialized') or not model.initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not initialized"
            )
            
        return {
            "status": "healthy",
            "model_initialized": True,
            "model_name": getattr(model, 'model_name', 'unknown')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}"
        )

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server with Uvicorn.
    
    Args:
        host: The host to bind to. Defaults to "0.0.0.0".
        port: The port to bind to. Defaults to 8000.
        reload: Whether to enable auto-reload. Defaults to False.
    """
    # Configure Uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info" if os.getenv("DEBUG") else "warning",
        workers=1 if reload else 4,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
    
    server = uvicorn.Server(config)
    
    # Log startup information
    logger.info(f"Starting JARVIS server on {host}:{port}")
    logger.info(f"Environment: {'development' if os.getenv('DEBUG') else 'production'}")
    logger.info(f"Auto-reload: {'enabled' if reload else 'disabled'}")
    
    try:
        # Run the server
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        # Cleanup code can go here
        pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run JARVIS server")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)),
                       help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ["DEBUG"] = "1"
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    run_server(host=args.host, port=args.port, reload=args.reload)
