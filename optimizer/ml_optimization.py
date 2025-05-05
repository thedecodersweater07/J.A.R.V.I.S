import torch
import numpy as np
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class MLOptimizationConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    memory_efficient: bool = True
    
class MLOptimizer:
    """Optimizes ML models for performance and memory efficiency"""
    
    def __init__(self, config: MLOptimizationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply optimizations to model"""
        model = model.to(self.device)
        if self.config.mixed_precision:
            model = self._enable_mixed_precision(model)
        if self.config.memory_efficient:
            model = self._optimize_memory(model)
        return model

    def _enable_mixed_precision(self, model: torch.nn.Module) -> torch.nn.Module:
        """Enable automatic mixed precision"""
        if self.device.type == "cuda":
            model = torch.cuda.amp.autocast()(model)
        return model
        
    def _optimize_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply memory optimizations"""
        # Enable gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model
