import logging
import sys
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """Configure logging for the application"""
    # Set up logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path('app.log'))
        ]
    )
    return logging.getLogger('jarvis')