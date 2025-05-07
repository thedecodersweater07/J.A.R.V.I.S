import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from transformers import PreTrainedModel
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMOptimizationConfig:
    attention_slicing: bool = True
    kv_caching: bool = True
    tensor_parallelism: bool = False
    batch_size: int = 1
    max_length: int = 512
    quantization: bool = False
    gradient_checkpointing: bool = False
    
    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None) -> 'LLMOptimizationConfig':
        if config is None:
            return cls()
        return cls(**{
            k: v for k, v in config.items() 
            if hasattr(cls, k)
        })

class LLMOptimizer:
    """Optimizes LLM inference and processing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimizer with configuration"""
        self.config = LLMOptimizationConfig.from_dict(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LLMOptimizer with device: {self.device}")
        
    def optimize_inference(self, model: PreTrainedModel) -> PreTrainedModel:
        """Optimize model for inference"""
        try:
            logger.debug("Starting model optimization")
            model = model.eval()
            
            if self.config.attention_slicing:
                model = self._enable_attention_slicing(model)
                
            if self.config.kv_caching:
                model = self._enable_kv_caching(model)
                
            if self.config.quantization:
                model = self._apply_quantization(model)
                
            model = model.to(self.device)
            logger.info("Model optimization completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            raise

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
