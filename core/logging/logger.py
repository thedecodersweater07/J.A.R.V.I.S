import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from typing import Optional
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

def setup_logging(log_dir: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Setup application logging"""
    
    # Create logs directory
    log_dir = Path(log_dir or Path(__file__).parent.parent.parent / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Format strings
    console_format = "%(timestamp)s | %(levelname)-8s | [%(name)s] %(message)s"
    file_format = "%(timestamp)s | %(levelname)-8s | [%(name)s:%(lineno)d] | %(message)s"

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

    # JSON handler with proper formatter
    json_handler = RotatingFileHandler(
        log_dir / "jarvis_structured.jsonl",
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    json_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(json_handler)

    return root_logger

def setup_logging(config_path: Optional[Path] = None) -> None:
    """Setup logging configuration"""
    try:
        if config_path and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {
                "level": "INFO",
                "format": "%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
                "file": "logs/jarvis.log"
            }

        # Create logs directory
        log_path = Path(config["file"]).parent
        log_path.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=config["level"],
            format=config["format"],
            handlers=[
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    config["file"],
                    maxBytes=10485760,  # 10MB
                    backupCount=5,
                    encoding="utf-8"
                )
            ]
        )

    except Exception as e:
        # Fallback to basic configuration
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to setup logging: {str(e)}", exc_info=True)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    logger = logging.getLogger(name)
    
    # Only set up handlers if they haven't been set up already
    if not logger.handlers and not logging.getLogger().handlers:
        setup_logging()
        
    return logger