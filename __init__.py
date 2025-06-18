"""
JARVIS AI System
Root package initialization
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
__version__ = '0.1.0'
__author__ = 'Your Name <your.email@example.com>'

# Ensure the project root is in the Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis.log')
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"JARVIS AI System v{__version__} initialized")

# Set environment variables
os.environ["PYTHONPATH"] = os.pathsep.join(
    [project_root] + os.environ.get("PYTHONPATH", "").split(os.pathsep)
)

# Import key modules to make them available at package level
try:
    from core.logging import setup_logging, get_logger
    from server import app as server_app
    from models.database import init_db, get_db
    from config import settings
    from utils.helpers import setup_environment
    
    # Set up logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Initialize environment
    setup_environment()
    
    # Initialize database
    init_db()
    
    logger.info("JARVIS package initialized successfully")
    
except ImportError as e:
    print(f"Warning: Could not initialize JARVIS package: {e}", file=sys.stderr)

# Version
__version__ = "0.1.0"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__version__ = "0.1.0"
