import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, log_dir: Optional[str] = None) -> None:
    """Enhanced logging setup with better configuration"""
    # Create log directory if needed
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
    # Base config
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": sys.stdout
            }
        },
        "root": {
            "handlers": ["console"],
            "level": level.upper()
        },
        "loggers": {
            "jarvis": {
                "handlers": ["console"],
                "level": level.upper(),
                "propagate": False
            },
            "jarvis-server": {
                "handlers": ["console"],
                "level": level.upper(),
                "propagate": False
            }
        }
    }
    
    # Add file handler if needed
    if log_file:
        log_path = Path(log_dir) / log_file if log_dir else Path(log_file)
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": str(log_path),
            "maxBytes": 10485760,
            "backupCount": 5
        }
        config["root"]["handlers"].append("file")
        
    # Apply configuration
    logging.config.dictConfig(config)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)

# Setup default logging when module is imported
setup_logging()