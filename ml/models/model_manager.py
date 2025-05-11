import logging
from typing import Dict, Optional, Any
import torch
from core.ml.model_manager import ModelManager as CoreModelManager

class ModelManager(CoreModelManager):
    """Extended ModelManager for ML models"""
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    # The load_model method is inherited from CoreModelManager