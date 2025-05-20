import os
import sys
import logging
import argparse
import traceback
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the server app with proper error handling
try:
    from server.app import start_server
    from .security.security_manager import SecurityManager
    from .security.middleware import SecurityMiddleware
    from .api import auth, ai, system
except ImportError as e:
    print(f"Error importing server app: {e}")
    sys.exit(1)

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
    
    return app

def main():
    """Main entry point for the server launcher"""
    parser = argparse.ArgumentParser(description='JARVIS Server Backend')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info(f"Starting JARVIS server on {args.host}:{args.port}")
    
    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Start the server
        start_server(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
