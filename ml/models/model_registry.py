from typing import Dict, Type, Any
from pathlib import Path
import torch
from .base import BaseModel
from ..data import MLDataManager

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Type[BaseModel]] = {}
        self.data_manager = MLDataManager()
        self.model_dir = Path("data/models")
        
    def register_model(self, name: str, model_class: Type[BaseModel]):
        self.models[name] = model_class
        
    def get_model(self, name: str, config: Dict[str, Any]) -> BaseModel:
        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")
            
        # Load training data
        train_data = self.data_manager.load_training_data(name)
        
        # Initialize model
        model = self.models[name](config)
        model.prepare_training_data(train_data)
        
        return model
