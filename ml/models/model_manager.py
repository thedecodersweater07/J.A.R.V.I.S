import logging
import gc
import os
import threading
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
import torch
import numpy as np
from functools import lru_cache
from pathlib import Path
from core.ml.model_manager import ModelManager as CoreModelManager

logger = logging.getLogger(__name__)

class ModelManager(CoreModelManager):
    """Extended ModelManager for ML models with memory optimization"""
    def __init__(self, base_path: str = "models", max_loaded_models: int = 5):
        super().__init__(base_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_loaded_models = max_loaded_models
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        self._model_lock = threading.RLock()
        self._model_usage_count: Dict[str, int] = {}
        self._last_accessed: Dict[str, float] = {}
        
        # Create model type directories
        self._init_model_dirs(['classifier', 'regressor', 'clustering'])
        
        # Configure GPU memory management if available
        self._setup_gpu_memory_management()
        
    def _init_model_dirs(self, model_types: List[str]):
        """Initialize model type directories"""
        try:
            for model_type in model_types:
                model_dir = Path(self.base_path) / model_type
                model_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created model directory: {model_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create model directories: {e}")
            
    def _setup_gpu_memory_management(self):
        """Setup GPU memory management"""
        if torch.cuda.is_available():
            self.device = "cuda"
            # Log available GPU memory
            for device in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(device)
                self.logger.info(f"Using GPU {device}: {device_props.name} with {device_props.total_memory / 1e9:.2f} GB memory")
        else:
            self.device = "cpu"
            self.logger.info("Using CPU for models")
            
    def load_model(self, model_name: str, version: str = 'latest', force_reload: bool = False) -> Tuple[Any, Dict]:
        """Load a model with memory optimization"""
        model_key = f"{model_name}_{version}"
        
        with self._model_lock:
            # Check if model is already loaded and not forced to reload
            if model_key in self._loaded_models and not force_reload:
                self.logger.debug(f"Using cached model: {model_key}")
                model_info = self._loaded_models[model_key]
                self._update_model_usage(model_key)
                return model_info['model'], model_info['metadata']
                
            # If we're at max capacity, unload least recently used model
            if len(self._loaded_models) >= self.max_loaded_models:
                self._unload_least_used_model()
                
            # Load the model using parent class method
            try:
                model, metadata = super().load_model(model_name, version)
                
                # Store in loaded models cache
                self._loaded_models[model_key] = {
                    'model': model,
                    'metadata': metadata,
                    'size': self._estimate_model_size(model)
                }
                
                # Update usage statistics
                self._update_model_usage(model_key)
                
                self.logger.info(f"Loaded model {model_key} into memory")
                return model, metadata
            except Exception as e:
                self.logger.error(f"Error loading model {model_key}: {str(e)}")
                raise
                
    def _update_model_usage(self, model_key: str) -> None:
        """Update model usage statistics"""
        import time
        self._last_accessed[model_key] = time.time()
        self._model_usage_count[model_key] = self._model_usage_count.get(model_key, 0) + 1
        
    def _unload_least_used_model(self) -> None:
        """Unload the least recently used model"""
        if not self._loaded_models:
            return
            
        # Find least recently accessed model
        least_recent_key = min(self._last_accessed.items(), key=lambda x: x[1])[0]
        
        # Unload it
        if least_recent_key in self._loaded_models:
            self.logger.info(f"Unloading least recently used model: {least_recent_key}")
            del self._loaded_models[least_recent_key]
            if least_recent_key in self._last_accessed:
                del self._last_accessed[least_recent_key]
            if least_recent_key in self._model_usage_count:
                del self._model_usage_count[least_recent_key]
                
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate memory usage of a model in bytes"""
        if hasattr(model, 'state_dict'):
            # PyTorch model
            return sum(param.nelement() * param.element_size() for param in model.parameters())
        elif hasattr(model, '__sizeof__'):
            # Object with size information
            return model.__sizeof__()
        else:
            # Rough estimate for other types
            import sys
            return sys.getsizeof(model)
            
    def unload_model(self, model_name: str, version: str = 'latest') -> bool:
        """Explicitly unload a model from memory"""
        model_key = f"{model_name}_{version}"
        
        with self._model_lock:
            if model_key in self._loaded_models:
                del self._loaded_models[model_key]
                if model_key in self._last_accessed:
                    del self._last_accessed[model_key]
                if model_key in self._model_usage_count:
                    del self._model_usage_count[model_key]
                    
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                self.logger.info(f"Unloaded model {model_key} from memory")
                return True
            return False
            
    def unload_all_models(self) -> None:
        """Unload all models from memory"""
        with self._model_lock:
            self._loaded_models.clear()
            self._last_accessed.clear()
            self._model_usage_count.clear()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Unloaded all models from memory")
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        usage = {
            'loaded_models': len(self._loaded_models),
            'models': {}
        }
        
        # Add per-model information
        for model_key, model_info in self._loaded_models.items():
            usage['models'][model_key] = {
                'size_bytes': model_info.get('size', 0),
                'access_count': self._model_usage_count.get(model_key, 0)
            }
            
        # Add GPU memory information if available
        if torch.cuda.is_available():
            usage['gpu'] = {
                'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(),
                'max_memory_bytes': torch.cuda.max_memory_allocated()
            }
            
        return usage
        
    def __del__(self):
        """Ensure models are unloaded when the manager is deleted"""
        try:
            self.unload_all_models()
        except:
            pass