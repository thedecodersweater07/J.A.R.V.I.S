import logging
import sys
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """Configure logging for the application"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Ensure logs directory exists
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'jarvis.log'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    # Reduce verbosity of external libraries
    logging.getLogger('OpenGL').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
