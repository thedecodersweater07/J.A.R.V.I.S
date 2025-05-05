from typing import Dict, Any
import torch

class LLMOptimizer:
    """Optimize LLM performance and resource usage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            return min(32, max(1, int(gpu_mem / (1024**3) * 4)))
        return 8  # Default CPU batch size
        
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get optimized settings for model inference"""
        return {
            "use_cache": True,
            "fp16": torch.cuda.is_available(),
            "batch_size": self.get_optimal_batch_size(),
            "thread_count": min(4, torch.get_num_threads()),
            "attention_slicing": "auto",
            "gradient_checkpointing": True if torch.cuda.is_available() else False
        }
