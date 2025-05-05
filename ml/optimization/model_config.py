from typing import Dict, Any
import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for ML model optimization"""
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer_type: str
    scheduler_type: str
    mixed_precision: bool
    gradient_clipping: float

class MLOptimizer:
    """Optimize ML models for better performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_training_config(self) -> ModelConfig:
        """Get optimized training configuration"""
        return ModelConfig(
            batch_size=self._get_optimal_batch_size(),
            learning_rate=1e-4 if self.device.type == "cuda" else 1e-3,
            weight_decay=0.01,
            optimizer_type="AdamW",
            scheduler_type="cosine",
            mixed_precision=torch.cuda.is_available(),
            gradient_clipping=1.0
        )
        
    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on model and hardware"""
        if self.device.type == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            return min(64, max(16, int(gpu_mem / (1024**3) * 8)))
        return 16  # Default CPU batch size
