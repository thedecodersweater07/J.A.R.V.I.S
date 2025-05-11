from pathlib import Path
import numpy as np
from typing import Any, Dict, Optional
from ml.models.model_manager import ModelManager
from ..data.manager import DataManager

class ModelTrainer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.data_manager = DataManager()
        
    def train_and_save(self, 
                       model: Any,
                       dataset_name: str,
                       model_name: str,
                       params: Dict = None) -> None:
        """Train a model and save it with its learned data"""
        # Load training data
        data = self.data_manager.load_dataset(dataset_name)
        
        # Train the model (generic example)
        if hasattr(model, 'fit'):
            if isinstance(data, dict):
                model.fit(data['X'], data['y'])
            else:
                X = data.drop('target', axis=1)
                y = data['target']
                model.fit(X, y)
        
        # Save model and training info
        metadata = {
            'dataset': dataset_name,
            'parameters': params or {},
            'features': list(X.columns) if hasattr(X, 'columns') else None
        }
        
        self.model_manager.save_model(model, model_name, metadata)
        
    def predict(self,
                model_name: str,
                input_data: Any,
                version: str = 'latest') -> np.ndarray:
        """Load a model and make predictions"""
        model, metadata = self.model_manager.load_model(model_name, version)
        return model.predict(input_data)
