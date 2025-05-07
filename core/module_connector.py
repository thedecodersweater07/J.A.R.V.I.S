from typing import Dict, Any, Optional
import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModuleConnector:
    """Manages connections between different JARVIS modules"""
    
    def __init__(self):
        self.modules = {}
        self.connections = {}
        self.dependencies = {
            "llm": ["knowledge", "memory"],
            "ml": ["data_processor", "model_manager"],
            "brain": ["cognitive_functions", "memory"]
        }
        
    def connect_modules(self):
        """Connect all required modules"""
        for module, deps in self.dependencies.items():
            try:
                module_instance = self._load_module(module)
                for dep in deps:
                    dep_instance = self._load_module(dep)
                    self.connections[f"{module}_{dep}"] = {
                        "source": module_instance,
                        "target": dep_instance
                    }
            except Exception as e:
                logger.error(f"Failed to connect module {module}: {e}")
                
    def _load_module(self, module_name: str) -> Any:
        """Dynamically load a module"""
        try:
            module_path = f"jarvis.{module_name}"
            return importlib.import_module(module_path)
        except ImportError as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            raise
