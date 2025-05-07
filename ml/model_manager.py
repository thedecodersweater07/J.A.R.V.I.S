import logging
from typing import Dict, Optional, Any
import torch
from .training.trainer import ModelTrainer
from .optimization.optimizer import ModelOptimizer
import os
import pickle
import json
from pathlib import Path
import numpy as np
from datetime import datetime

class ModelManager:
    def __init__(self, base_path: str = "../data/ai_training_data/models"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        
        # Initialize with empty configurations
        default_config = {"epochs": 10, "batch_size": 32}
        self.trainer = ModelTrainer(config=default_config)
        self.optimizer = ModelOptimizer(config=default_config)
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        """Initialize the model manager and load default models"""
        self.logger.info("Initializing Model Manager...")
        self.load_models()

    def load_models(self):
        """Load all required models"""
        self.logger.info("Loading models...")
        # Add model loading logic here
        pass

    def load_model(self, model_name: str, model_path: str) -> Optional[Any]:
        """Load model and update trainer/optimizer"""
        try:
            model = torch.load(model_path)
            self.models[model_name] = model
            
            # Update trainer and optimizer with loaded model
            self.trainer._set_model(model)
            self.optimizer._set_model(model)
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None

    def train_model(self, data: Any, parameters: Dict[str, Any]) -> Optional[Any]:
        self.logger.info("Starting model training...")
        model = self.trainer.train(data, parameters)
        self.logger.info("Model training completed.")
        return model

    def optimize_model(self, model: Any, parameters: Dict[str, Any]) -> Optional[Any]:
        self.logger.info("Starting model optimization...")
        optimized_model = self.optimizer.optimize(model, parameters)
        self.logger.info("Model optimization completed.")
        return optimized_model

    def save_model(self, model, model_name: str, metadata: dict = None):
        """Save a trained model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.base_path / f"{model_name}_{timestamp}"
        model_dir.mkdir(exist_ok=True)
        
        # Save the model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Save metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'saved_at': timestamp,
            'model_name': model_name
        })
        
        meta_path = model_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load_model(self, model_name: str, version: str = 'latest'):
        """Load a saved model"""
        if version == 'latest':
            model_dirs = list(self.base_path.glob(f"{model_name}_*"))
            if not model_dirs:
                raise FileNotFoundError(f"No models found for {model_name}")
            model_dir = sorted(model_dirs)[-1]
        else:
            model_dir = self.base_path / f"{model_name}_{version}"
            
        model_path = model_dir / "model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        meta_path = model_dir / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)
            
        return model, metadata