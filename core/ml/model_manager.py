import logging
from typing import Dict, Optional, Any, Tuple
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime

from ..logging.logger import get_logger

logger = get_logger(__name__)

class ModelManager:
    def __init__(self, base_path: str = "models"):
        self.models: Dict[str, Any] = {}
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def initialize(self):
        """Initialize models and start data collection"""
        logger.info("Initializing ModelManager")
        await self.data_collector.start_collection()
        await self.load_default_models()
        
    async def load_default_models(self):
        """Load default ML models"""
        try:
            # Load models implementation
            logger.info("Default models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading default models: {e}")
            
    def save_model_state(self, model_name: str, model: Any):
        """Save model state with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.model_dir / f"{model_name}_{timestamp}.pt"
        
        try:
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model {model_name} saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            
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

    def _get_model_path(self, model_name: str, version: str = 'latest') -> Path:
        """Get path for model file"""
        if version == 'latest':
            pattern = f"{model_name}_*.pt"
            model_files = list(self.model_dir.glob(pattern))
            if not model_files:
                return self.model_dir / f"{model_name}_default.pt"
            return sorted(model_files)[-1]
        return self.model_dir / f"{model_name}_{version}.pt"
