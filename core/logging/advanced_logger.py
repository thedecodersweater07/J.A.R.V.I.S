import logging
import json
from datetime import datetime
from typing import Any
from pathlib import Path
import sys
import numpy as np
import torch
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays, torch tensors, and datetime objects"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):  # Add numpy array support
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):  # Add torch tensor support
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)

class AdvancedLogger:
    """Advanced logging system with rich formatting and JSON support"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.console = Console()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=True
        )
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(
            self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
    def get_logger(self):
        return self.logger
        
    @staticmethod
    def serialize_for_json(obj: Any) -> Any:
        """Serialize objects for JSON storage"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
