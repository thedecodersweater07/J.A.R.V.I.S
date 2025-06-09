import os
import sys
import json
import logging
import argparse
import traceback
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Configure logging early
def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'server.log')
        ]
    )

logger = logging.getLogger("jarvis-server-launcher")

# Import modules with comprehensive error handling
def import_with_fallback():
    """Import server modules with fallbacks for missing components"""
    imported_modules = {
        'start_server': None,
        'security_manager': None,
        'security_middleware': None,
        'auth_router': None,
        'ai_router': None,
        'system_router': None,
        'websocket_router': None,
        'websocket_endpoint': None
    }
    
    # Try importing main server app
    try:
        from server.app import start_server
        imported_modules['start_server'] = start_server
        logger.info("Successfully imported server app")
    except ImportError as e:
        logger.warning(f"Could not import server app: {e}")
        # Try alternative import
        try:
            from app import create_app as start_server
            imported_modules['start_server'] = start_server
            logger.info("Successfully imported alternative server app")
        except ImportError:
            logger.warning("No server app found, will create basic app")
    
    # Try importing security components
    try:
        from server.security.security_manager import SecurityManager
        imported_modules['security_manager'] = SecurityManager
        logger.info("Successfully imported SecurityManager")
    except ImportError as e:
        logger.warning(f"Could not import SecurityManager: {e}")
    
    try:
        from server.security.middleware import SecurityMiddleware
        imported_modules['security_middleware'] = SecurityMiddleware
        logger.info("Successfully imported SecurityMiddleware")
    except ImportError as e:
        logger.warning(f"Could not import SecurityMiddleware: {e}")
    
    # Try importing API routers
    try:
        from server.api import auth
        imported_modules['auth_router'] = getattr(auth, 'router', None)
        logger.info("Successfully imported auth router")
    except ImportError as e:
        logger.warning(f"Could not import auth router: {e}")
    
    try:
        from server.api import ai
        imported_modules['ai_router'] = getattr(ai, 'router', None)
        logger.info("Successfully imported ai router")
    except ImportError as e:
        logger.warning(f"Could not import ai router: {e}")
    
    try:
        from server.api import system
        imported_modules['system_router'] = getattr(system, 'router', None)
        logger.info("Successfully imported system router")
    except ImportError as e:
        logger.warning(f"Could not import system router: {e}")
    
    # Try importing WebSocket components
    try:
        from server.websocket_handler import router as websocket_router, websocket_endpoint
        imported_modules['websocket_router'] = websocket_router
        imported_modules['websocket_endpoint'] = websocket_endpoint
        logger.info("Successfully imported WebSocket components")
    except ImportError as e:
        logger.warning(f"WebSocket components not available: {e}")
    
    return imported_modules

# Load configuration
def load_config() -> dict:
    """Load server configuration from file"""
    config_path = Path("config/server.json")
    default_config = {
        "cors_origins": ["*"],
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False,
        "security": {
            "enabled": True,
            "jwt_secret": "your-secret-key-here",
            "jwt_algorithm": "HS256",
            "jwt_expiration": 3600
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            logger.warning(f"Error loading config file: {e}, using defaults")
    else:
        logger.info("Config file not found, using defaults")
        # Create config directory and file
        config_path.parent.mkdir(exist_ok=True)
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file at {config_path}")
        except Exception as e:
            logger.warning(f"Could not create config file: {e}")
    
    return default_config

def create_basic_routes(app: FastAPI):
    """Create basic API routes when full routers are not available"""
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "JARVIS API"}
    
    @app.get("/api/status")
    async def api_status():
        """API status endpoint"""
        return {
            "status": "running",
            "version": "1.0.0",
            "message": "JARVIS API is operational"
        }
    
    @app.post("/api/chat")
    async def chat_fallback():
        """Fallback chat endpoint"""
        return {
            "response": "JARVIS AI system is starting up. Full functionality will be available shortly.",
            "status": "initializing"
        }

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    # Load configuration
    config = load_config()
    
    # Import modules
    modules = import_with_fallback()
    
    # Use imported start_server if available, otherwise create new app
    if modules['start_server']:
        try:
            app = modules['start_server']()
            logger.info("Using imported server app")
        except Exception as e:
            logger.error(f"Error using imported server app: {e}")
            app = FastAPI(
                title="JARVIS API",
                description="JARVIS AI System API",
                version="1.0.0"
            )
    else:
        # Initialize new app
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
    
    # Add security middleware if available
    if modules['security_middleware'] and config.get("security", {}).get("enabled", True):
        try:
            security_manager = modules['security_manager']() if modules['security_manager'] else None
            if security_manager:
                app.add_middleware(modules['security_middleware'], security_manager=security_manager)
                logger.info("Added security middleware")
        except Exception as e:
            logger.warning(f"Could not add security middleware: {e}")
    
    # Include API routers if available
    routers_added = 0
    for router_name, router in [
        ('auth', modules['auth_router']),
        ('ai', modules['ai_router']),
        ('system', modules['system_router'])
    ]:
        if router:
            try:
                app.include_router(router, prefix=f"/api/{router_name}")
                routers_added += 1
                logger.info(f"Added {router_name} router")
            except Exception as e:
                logger.warning(f"Could not add {router_name} router: {e}")
    
    # Add basic routes if no routers were added
    if routers_added == 0:
        create_basic_routes(app)
        logger.info("Added basic fallback routes")
    
    # Configure WebSocket if available
    if modules['websocket_router']:
        try:
            app.include_router(modules['websocket_router'])
            logger.info("Added WebSocket router")
        except Exception as e:
            logger.warning(f"Could not add WebSocket router: {e}")
    
    if modules['websocket_endpoint']:
        try:
            app.add_websocket_route("/ws", modules['websocket_endpoint'])
            logger.info("Added WebSocket endpoint")
        except Exception as e:
            logger.warning(f"Could not add WebSocket endpoint: {e}")
    
    # Serve static files
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        try:
            app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
            logger.info("Mounted static files")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
    
    # Add root route
    @app.get("/")
    async def read_index():
        """Serve index page or API info"""
        index_path = web_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            return {
                "message": "JARVIS API is running",
                "status": "ok",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/health",
                    "api_status": "/api/status",
                    "docs": "/docs",
                    "redoc": "/redoc"
                }
            }
    
    # Add error handlers
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return {"error": "Endpoint not found", "status": 404}
    
    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {exc}")
        return {"error": "Internal server error", "status": 500}
    
    logger.info("FastAPI application created successfully")
    return app

def main():
    """Main entry point for the server launcher"""
    parser = argparse.ArgumentParser(description='JARVIS Server Launcher')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', help='Path to config file')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger.info(f"Starting JARVIS server on {args.host}:{args.port}")

    try:
        # Create and run the app
        app = create_app()
        
        # Override config with command line arguments
        config = load_config()
        host = args.host or config.get('host', '127.0.0.1')
        port = args.port or config.get('port', 8000)
        debug = args.debug or config.get('debug', False)
        
        logger.info(f"Server configuration: host={host}, port={port}, debug={debug}")
        
        # Run server
        uvicorn.run(
            app, 
            host=host, 
            port=port, 
            log_level="debug" if debug else "info",
            reload=debug
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

    """
Simple JARVIS Server
"""
import os
import sys
import json
import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import components
from security.security_manager import SecurityManager
from security.auth import AuthHandler
from security.middleware import SimpleSecurityMiddleware
from security.auth import create_auth_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("jarvis.server")

def create_app() -> FastAPI:
    """Create FastAPI app"""
    # Load config
    config = {}
    config_file = Path("config.json")
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    
    # Create app
    app = FastAPI(
        title="JARVIS Server",
        description="Simple JARVIS AI Server",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize security
    security_manager = SecurityManager(config.get("security"))
    auth_handler = AuthHandler(security_manager)
    
    # Add middleware
    app.add_middleware(SimpleSecurityMiddleware, security_manager=security_manager)
    
    # Static files
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Routes
    app.include_router(create_auth_router(security_manager, auth_handler))
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "JARVIS Server is running", "status": "online"}
    
    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy", "service": "jarvis"}
    
    return app

def main():
    """Main server function"""
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    logger.info("Starting JARVIS Server...")
    
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )

if __name__ == "__main__":
    main()