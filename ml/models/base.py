import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """Base class for all ML models in JARVIS"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_data = None
        self.metrics = {}
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation"""
        pass
        
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        pass
        
    def prepare_training_data(self, data: Any) -> None:
        """Prepare data for training"""
        self.training_data = data
        
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.state_dict(),
            'config': self.config,
            'metrics': self.metrics,
            'device': self.device
        }
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
        
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state'])
            self.config.update(checkpoint['config'])
            self.metrics.update(checkpoint['metrics'])
            logger.info(f"Model checkpoint loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            
    def to_device(self, device: Optional[str] = None) -> 'BaseModel':
        """Move model to specified device"""
        if device:
            self.device = torch.device(device)
        self.to(self.device)
        return self
        
    def get_parameter_count(self) -> Dict[str, int]:
        """Get model parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params
        }
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage in MB"""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        
        return {
            "parameters_mb": param_size / 1024 / 1024,
            "buffers_mb": buffer_size / 1024 / 1024,
            "total_mb": (param_size + buffer_size) / 1024 / 1024
        }
