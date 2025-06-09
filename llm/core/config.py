from typing import Dict, Any
import torch
from transformers import __version__ as transformers_version
from packaging import version

class LLMConfigValidator:
    """Validates and fixes LLM configuration"""
    
    @staticmethod
    def validate_and_fix(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix LLM configuration"""
        if not config:
            config = {}
            
        # Ensure model config exists
        if "model" not in config:
            config["model"] = {}
            
        model_config = config["model"]
        
        # Clean potential duplicate parameters
        for key in ['low_cpu_mem_usage', 'device_map', 'torch_dtype']:
            if key in config and key in model_config:
                del config[key]  # Remove from top level if exists in model_config
        
        # Set safe defaults based on Transformers version
        transformers_ver = version.parse(transformers_version)
        if transformers_ver >= version.parse("4.20.0"):
            # Modern Transformers - use device_map
            model_config.setdefault("device_map", "auto")
            model_config.setdefault("low_cpu_mem_usage", True)
        else:
            # Older Transformers - use simpler config
            model_config.pop("device_map", None)
            model_config.setdefault("low_cpu_mem_usage", True)
        
        # Set appropriate dtype
        if torch.cuda.is_available():
            model_config["torch_dtype"] = torch.float16
        else:
            model_config["torch_dtype"] = torch.float32
            
        # Add memory management config
        if "memory_management" not in config:
            config["memory_management"] = {
                "cache_size": 1000,
                "max_memory_usage": 0.8,
                "enable_gc": True
            }
            
        return config

    @staticmethod
    def get_transformers_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract valid kwargs for transformers.from_pretrained()"""
        valid_keys = {
            "low_cpu_mem_usage", "torch_dtype", "device_map",
            "use_cache", "max_length", "revision"
        }
        
        if "model" in config:
            model_config = config["model"]
        else:
            model_config = config
            
        return {k: v for k, v in model_config.items() if k in valid_keys}
