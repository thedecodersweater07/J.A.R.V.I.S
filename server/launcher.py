import os
import sys
import logging
import uvicorn
from typing import Dict, Any, Optional
from pathlib import Path

# Setup logging
logger = logging.getLogger("jarvis-server.launcher")

def setup_server_logging():
    """Set up server-specific logging"""
    logging.getLogger("uvicorn").handlers = []
    logging.getLogger("uvicorn.access").handlers = []

def start_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
    """Start the FastAPI server with given configuration"""
    try:
        # Configure uvicorn logging
        setup_server_logging()
        
        # Import the FastAPI app
        sys.path.append(str(Path(__file__).parent.parent))
        from server.app import app
        
        # Configure server
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="debug" if debug else "info",
            reload=debug,
            workers=1
        )
        
        # Start server
        logger.info(f"Starting server on {host}:{port}")
        server = uvicorn.Server(config)
        server.run()
        
    except ImportError as e:
        logger.error(f"Failed to import server app: {e}")
        raise
    except Exception as e:
        logger.error(f"Server startup failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JARVIS Server Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    start_server(args.host, args.port, args.debug)
