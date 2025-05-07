import torch
from typing import Dict, Any
from dataclasses import dataclass
from transformers import PreTrainedModel

@dataclass
class LLMOptimizationConfig:
    attention_slicing: bool = True
    kv_caching: bool = True
    tensor_parallelism: bool = False
    batch_size: int = 1
    max_length: int = 512
    quantization: bool = False
    gradient_checkpointing: bool = False

class LLMOptimizer:
    """Optimizes LLM inference and processing"""
    
    def __init__(self, config: LLMOptimizationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def optimize_inference(self, model: Any) -> Any:
        """Optimize model for inference"""
        model = model.eval()
        if self.config.attention_slicing:
            model = self._enable_attention_slicing(model)
        if self.config.kv_caching:
            model = self._enable_kv_caching(model)
        return model.to(self.device)

    def _enable_attention_slicing(self, model: Any) -> Any:
        """Enable attention slicing for memory efficiency"""
        if hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing()
        return model

    def _enable_kv_caching(self, model: Any) -> Any:
        """Enable key-value caching for faster inference"""
        if hasattr(model, "enable_kv_caching"):
            model.enable_kv_caching()
        return model

    def optimize_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply optimization techniques to the model based on configuration."""
        if self.config.quantization:
            model = self._apply_quantization(model)
            
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        return model
        
    def _apply_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply quantization if supported by the model."""
        if hasattr(model, "quantize_"):
            model = model.quantize_()
        return model
