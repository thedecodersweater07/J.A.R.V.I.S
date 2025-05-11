import os
import sys
import logging
import argparse
import importlib
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("jarvis-server-launcher")

def import_app():
    """Import the server app with error handling"""
    try:
        from server.app import app
        return app
    except ImportError as e:
        logger.error(f"Failed to import server app: {e}")
        return None

def start_server(host="127.0.0.1", port=8000, debug=False):
    """Start the server with the specified configuration"""
    try:
        import uvicorn
        
        # Import the app
        app = import_app()
        if app is None:
            logger.error("Server app not available, cannot start server")
            return False
        
        # Set log level
        log_level = "debug" if debug else "info"
        
        # Start the server
        logger.info(f"Starting JARVIS server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level=log_level)
        return True
        
    except ImportError:
        logger.error("Uvicorn not installed, cannot start server")
        return False
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def main():
    """Main entry point for the server launcher"""
    parser = argparse.ArgumentParser(description='JARVIS Server Backend')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Start the server
    start_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
