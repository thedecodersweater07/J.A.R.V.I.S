from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Running in limited functionality mode.")
    TORCH_AVAILABLE = False
    
    # Create dummy classes for type checking
    class DummyModule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def to(self, *args, **kwargs):
            return self
        def train(self, mode=True):
            return self
        def eval(self):
            return self
        def parameters(self, recurse=True):
            return []
            
    class DummyTensor:
        def to(self, *args, **kwargs):
            return self
            
    # Create dummy modules
    class DummyDevice:
        def __init__(self, device_str):
            self.device_str = device_str
            
        def __str__(self):
            return self.device_str
            
        def __repr__(self):
            return f"device('{self.device_str}')"
    
    torch = type('torch', (), {
        'Tensor': DummyTensor,
        'device': DummyDevice,
        'cuda': type('cuda', (), {'is_available': lambda: False}),
        '__version__': '1.0.0'
    })()
    torch.device = DummyDevice
    nn = type('nn', (), {'Module': DummyModule})()

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
