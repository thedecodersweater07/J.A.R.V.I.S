from typing import List, Dict, Any
from dataclasses import dataclass
import torch

@dataclass
class NLPOptimizationConfig:
    batch_processing: bool = True
    cache_embeddings: bool = True
    parallel_processing: bool = True
    max_batch_size: int = 32
    use_quantization: bool = True

class NLPPipelineOptimizer:
    """Optimizes NLP processing pipeline"""
    
    def __init__(self, config: NLPOptimizationConfig):
        self.config = config
        self.cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def optimize_pipeline(self, pipeline: Any) -> Any:
        """Apply optimizations to NLP pipeline"""
        if self.config.batch_processing:
            pipeline = self._enable_batch_processing(pipeline)
        if self.config.cache_embeddings:
            pipeline = self._add_embedding_cache(pipeline)
        if self.config.use_quantization:
            pipeline = self._quantize_pipeline(pipeline)
        return pipeline

    def _enable_batch_processing(self, pipeline: Any) -> Any:
        """Enable efficient batch processing"""
        if hasattr(pipeline, "enable_batching"):
            pipeline.enable_batching(batch_size=self.config.max_batch_size)
        return pipeline

    def _add_embedding_cache(self, pipeline: Any) -> Any:
        """Add caching for embeddings"""
        if hasattr(pipeline, "add_embedding_cache"):
            pipeline.add_embedding_cache(self.cache)
        return pipeline

    def _quantize_pipeline(self, pipeline: Any) -> Any:
        """Apply quantization to pipeline components"""
        if self.device.type == "cpu" and hasattr(pipeline, "quantize"):
            pipeline.quantize(dtype=torch.qint8)
        return pipeline
