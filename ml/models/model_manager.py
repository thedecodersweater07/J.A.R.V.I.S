import logging
import gc
import os
import threading
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
import torch
import numpy as np
import pickle
from functools import lru_cache
from pathlib import Path
from core.ml.model_manager import ModelManager as CoreModelManager

logger = logging.getLogger(__name__)

class ModelManager(CoreModelManager):
    """Extended ModelManager for ML models with memory optimization"""
    def __init__(self, base_path: str = "models", max_loaded_models: int = 5):
        # Initialize logger first
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            super().__init__(base_path)
            
            self.max_loaded_models = max_loaded_models
            self._loaded_models: Dict[str, Dict[str, Any]] = {}
            self._model_lock = threading.RLock()
            self._model_usage_count: Dict[str, int] = {}
            self._last_accessed: Dict[str, float] = {}
            
            # Create model type directories
            self._init_model_dirs(['classifier', 'regressor', 'clustering'])
            
            # Configure GPU memory management if available
            self._setup_gpu_memory_management()
            
            # Add error handling for path initialization
            base_path = Path(base_path).resolve()
            if not base_path.exists():
                self.logger.warning(f"Base path {base_path} does not exist, creating it")
                base_path.mkdir(parents=True, exist_ok=True)
                
            self.base_path = base_path
            
        except Exception as e:
            self.logger.error(f"Error during ModelManager initialization: {e}")
            # Use fallback path
            self.base_path = Path("data/models").resolve()
            self.base_path.mkdir(parents=True, exist_ok=True)
        
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
            
    def load_model(self, model_name: str, version: str = 'latest'):
        """Load model with improved fallback handling"""
        
        # Try loading from model directory
        if version == 'latest':
            model_dirs = sorted(list(self.base_path.glob(f"{model_name}_*")))
            if model_dirs:
                try:
                    model_dir = model_dirs[-1]
                    model_path = model_dir / "model.pkl"
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    return model, self._load_metadata(model_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_name}: {e}")

        # Create dummy model if loading fails
        self.logger.info(f"Creating dummy model for {model_name}")
        model_type = model_name.split('_')[0]
        return self._create_dummy_model(model_type), {"status": "dummy"}
                
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