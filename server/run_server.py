import os
import sys
import logging
import argparse
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the server app with proper error handling
try:
    from server.app import start_server
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
        start_server(host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
