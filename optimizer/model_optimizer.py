import torch
from typing import Any, Dict
import numpy as np

class ModelOptimizer:
    """Optimizes and compresses AI models for lightweight deployment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_level = config.get("compression_level", "balanced")
        self.target_size = config.get("target_size", "small")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def quantize_model(self, model: Any) -> Any:
        """Quantize model to reduce memory footprint"""
        if self.device == "cpu":
            # INT8 quantization for CPU
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            # Mixed precision for GPU
            return self._apply_mixed_precision(model)
    
    def _apply_mixed_precision(self, model: Any) -> Any:
        """Apply mixed precision training/inference"""
        return torch.cuda.amp.autocast()(model)
    
    def prune_model(self, model: Any, sparsity: float = 0.5) -> Any:
        """Remove unnecessary weights based on magnitude"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Zero out the smallest weights
                threshold = torch.quantile(abs(param.data), sparsity)
                mask = abs(param.data) < threshold
                param.data[mask] = 0
        return model
    
    def optimize_inference(self, model: Any) -> Any:
        """Optimize model for inference"""
        model.eval()  # Set to evaluation mode
        if self.device == "cuda":
            model = model.half()  # FP16 for GPU
        return torch.jit.script(model)  # TorchScript optimization
