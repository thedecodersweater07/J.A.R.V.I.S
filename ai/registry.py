import logging
from typing import Dict, Any, Optional

class ModelRegistry:
    def __init__(self):
        self._models = {}
        self.logger = logging.getLogger(__name__)
        
    def register_model(self, name: str, model: Any) -> None:
        """Register an AI model with the registry"""
        self._models[name] = model
        self.logger.info(f"Registered model: {name}")
        
    def get_model(self, name: str) -> Optional[Any]:
        """Retrieve a model from the registry"""
        return self._models.get(name)
        
    def unregister_model(self, name: str) -> None:
        """Remove a model from the registry"""
        if name in self._models:
            del self._models[name]
            self.logger.info(f"Unregistered model: {name}")
