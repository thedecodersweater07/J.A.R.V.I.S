import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch

logger = logging.getLogger(__name__)

class LLMConfig:
    DEFAULT_CONFIG = {
        "model_name": "gpt2",
        "max_length": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "model_path": "models/gpt2",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 1
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        self._config_path = config_path
        
    def load(self) -> bool:
        """Load and validate config"""
        try:
            if self._config_path and self._config_path.exists():
                with open(self._config_path) as f:
                    loaded_config = json.load(f)
                    
                # Validate and update config
                if self._validate_config(loaded_config):
                    self.config.update(loaded_config)
                    return True
                    
            logger.warning("Invalid llm config, using defaults")
            return False
            
        except Exception as e:
            logger.error(f"Error loading LLM config: {e}")
            return False
            
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration values"""
        required_keys = ["model_name", "max_length", "temperature"]
        
        # Check required keys
        if not all(key in config for key in required_keys):
            return False
            
        # Validate value ranges
        if not (0 < config.get("temperature", 0) <= 1.0):
            return False
            
        if not (0 < config.get("top_p", 0) <= 1.0):
            return False
            
        return True
