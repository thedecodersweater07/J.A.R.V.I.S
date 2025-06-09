import os
import sys
import logging
import argparse
import traceback
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import the server app with proper error handling
try:
    from server.app import start_server  # Changed from start_app to start_server
    from server.security.security_manager import SecurityManager
    from server.security.middleware import SecurityMiddleware
    from server.api import auth, ai, system
    
    # Try to import websocket handler
    try:
        from server.websocket_handler import router as websocket_router, websocket_endpoint
        websocket_handler_available = True
    except ImportError as e:
        print(f"WebSocket handler not available: {e}")
        websocket_router = None
        websocket_endpoint = None
        websocket_handler_available = False
        print("Warning: WebSocket modules not found, continuing without WebSocket support")
        
except ImportError as e:
    print(f"Error importing server app: {e}")
    websocket_router = None
    websocket_endpoint = None
    websocket_handler_available = False
    print("Warning: Some modules not found, some features may be unavailable")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'server.log'))
    ]
)

logger = logging.getLogger("jarvis-server-launcher")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    # Load configuration
    config_path = Path("config/server.json")
    config = {}
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            
    # Initialize app
    app = FastAPI(
        title="JARVIS API",
        description="JARVIS AI System API",
        version="1.0.0"
    )
    
    # Serve static files
    from fastapi.staticfiles import StaticFiles
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    
    # Add a route to serve index.html
    from fastapi.responses import FileResponse
    
    @app.get("/")
    async def read_index():
        index_path = web_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            return {"message": "JARVIS API is running", "status": "ok"}
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize security
    security_manager = SecurityManager(config.get("security"))
    app.add_middleware(SecurityMiddleware, security_manager=security_manager)
    
    # Add routers
    app.include_router(auth.router)
    app.include_router(ai.router)
    app.include_router(system.router)
    
    # Configure WebSocket if available
    if websocket_handler_available and websocket_router is not None:
        app.include_router(websocket_router)
        if websocket_endpoint is not None:
            app.add_websocket_route("/ws", websocket_endpoint)
    
    return app

def main():
    """Main entry point for the server launcher"""
    parser = argparse.ArgumentParser(description='JARVIS Server Launcher')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("server-launcher")
    logger.info(f"Starting JARVIS server on {args.host}:{args.port}")

    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create and run the app
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()