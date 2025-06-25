"""
Logging utilities for hyperadvanced_ai.

This module provides a centralized way to configure and access logging
functionality across the hyperadvanced_ai framework.
"""

import logging
import sys
from typing import Optional, Dict, Any

# Default log format
DEFAULT_FORMAT = '[%(asctime)s] %(levelname)-8s | %(name)-30s | %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Configure root logger
def configure_logging(
    level: int = logging.INFO,
    format_str: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    stream = None,
    filename: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Configure the root logger with the specified settings.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_str: Log message format string
        datefmt: Date format string for log timestamps
        stream: Stream to use for logging (default: stderr)
        filename: If specified, log to this file instead of stderr
        **kwargs: Additional arguments to pass to logging.basicConfig
    """
    # Set up handler
    if filename:
        handler: logging.Handler = logging.FileHandler(filename)
    else:
        handler = logging.StreamHandler(stream or sys.stderr)
    
    # Configure formatter
    formatter = logging.Formatter(format_str, datefmt)
    handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[handler],
        **kwargs
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__ of the calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # If this is the root logger or no handlers are configured,
    # set up a default console handler
    if not logger.handlers and (logger.level == logging.NOTSET or not logging.root.handlers):
        configure_logging()
    
    return logger

__all__ = ['get_logger', 'configure_logging']
