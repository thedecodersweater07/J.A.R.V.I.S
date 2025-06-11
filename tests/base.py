from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Base class for all JARVIS models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Convert config to dict if not already
        self.config = config.__dict__ if hasattr(config, '__dict__') else config
        # Ensure config is a dictionary
        if not isinstance(self.config, dict):
            self.config = {"default": self.config}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        pass
        
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt"""
        pass

    def to_device(self, device: Optional[torch.device] = None):
        """Move model to specified device"""
        if device is None:
            device = self.device
        self.to(device)
        return self
