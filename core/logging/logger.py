import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from typing import Optional, Union
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and better formatting"""
    
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        # Add color to log level
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Add timestamp in ISO format
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings"""
    def format(self, record):
        """Format log record as JSON"""
        return json.dumps({
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'line': record.lineno,
            'file': record.filename
        })

def setup_logging(
    level: Union[str, int] = "INFO",
    log_dir: Optional[str] = None,
    config_path: Optional[Path] = None
) -> logging.Logger:
    """Setup application logging with flexible configuration options"""
    try:
        # Handle str/int level
        if isinstance(level, str):
            level = getattr(logging, level.upper())
            
        # Load config if provided
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = None

        # Set up log directory
        log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Format strings
        console_format = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
        file_format = "%(asctime)s | %(levelname)-8s | [%(name)s:%(lineno)d] | %(message)s"

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter(console_format))
        root_logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_dir / "jarvis.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)

        # JSON handler
        json_handler = RotatingFileHandler(
            log_dir / "jarvis_structured.jsonl",
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)

        return root_logger

    except Exception as e:
        # Fallback to basic configuration
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to setup logging: {str(e)}", exc_info=True)
        return logging.getLogger()

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    logger = logging.getLogger(name)
    
    # Only set up handlers if they haven't been set up already
    if not logger.handlers and not logging.getLogger().handlers:
        setup_logging()
        
    return logger