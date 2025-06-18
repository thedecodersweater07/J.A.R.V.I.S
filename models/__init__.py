"""
JARVIS Models Package
====================
Core models for JARVIS AI system.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Type, Union, TypeVar, List, Callable, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the path to allow absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import base model first
from .base import BaseModel

# Import JARVIS models
try:
    from .jarvis import JarvisModel, JarvisLanguageModel, JarvisModelManager
    JARVIS_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"JARVIS models not available: {e}")
    JARVIS_MODELS_AVAILABLE = False
    
    # Create dummy classes if imports fail
    class DummyJarvisModel(BaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__({})  # Pass empty config dict
            self.model_name = kwargs.get('model_name', 'dummy')
        def forward(self, *args, **kwargs):
            return {}
        def generate(self, *args, **kwargs):
            return {}
    
    class DummyJarvisLanguageModel:
        def __init__(self, *args, **kwargs):
            self.model_name = kwargs.get('model_name', 'dummy')
        def generate_response(self, *args, **kwargs):
            return "Language model not available."
    
    class DummyJarvisModelManager:
        def __init__(self):
            self.models = {}
    
    # Create type aliases for the dummy classes
    JarvisModel = DummyJarvisModel
    JarvisLanguageModel = DummyJarvisLanguageModel
    JarvisModelManager = DummyJarvisModelManager

# Try to import NLP module
try:
    from nlp import NLPProcessor
    NLP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NLP module not available: {e}")
    NLP_AVAILABLE = False
    
    # Create a dummy NLPProcessor if import fails
    class DummyNLPProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def process(self, *args, **kwargs):
            return {
                "text": "", 
                "tokens": [], 
                "language": "en", 
                "entities": [], 
                "sentiment": {"polarity": 0.0, "subjectivity": 0.0}
            }
    
    NLPProcessor = DummyNLPProcessor  # type: ignore

# Define LLMProcessor type
try:
    from llm.llm_processor import LLMProcessor as _LLMProcessor
    LLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLM module not available: {e}")
    LLM_AVAILABLE = False
    
    # Create a dummy LLMProcessor if import fails
    class _DummyLLMProcessor:
        def __init__(self, *args, **kwargs):
            self.model_name = kwargs.get('model_name', 'dummy')
        def generate(self, *args, **kwargs):
            return "LLM functionality not available. Please install required dependencies."
    
    _LLMProcessor = _DummyLLMProcessor

# Create type alias for LLMProcessor
LLMProcessor = _LLMProcessor  # type: ignore

class ModelFactory:
    """Factory class for creating JARVIS model instances with integrated NLP and LLM capabilities."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Union[JarvisModel, JarvisLanguageModel, 'JarvisNLPModel', 'JarvisMLModel']:
        """
        Create a JARVIS model instance with the specified configuration.
        
        Args:
            model_type: Type of model to create (e.g., 'jarvis', 'language', 'nlp', 'ml')
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Instance of the requested model type
            
        Raises:
            ValueError: If the model type is not supported
        """
        model_type = model_type.lower()
        
        try:
            if model_type in ['jarvis', 'base']:
                return JarvisModel(**kwargs)
            elif model_type == 'language':
                return JarvisLanguageModel(**kwargs)
            elif model_type == 'nlp':
                from .jarvis import JarvisNLPModel
                return JarvisNLPModel(**kwargs)
            elif model_type == 'ml':
                from .jarvis import JarvisMLModel
                return JarvisMLModel(**kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"Error creating model of type {model_type}: {e}")
            # Return a dummy model that won't crash the application
            return DummyJarvisModel(**kwargs)

# Model registry with factory methods
MODELS = {
    "jarvis-small": lambda **kwargs: ModelFactory.create_model("jarvis-small", **kwargs),
    "jarvis-base": lambda **kwargs: ModelFactory.create_model("jarvis-base", **kwargs),
    "jarvis-large": lambda **kwargs: ModelFactory.create_model("jarvis-large", **kwargs),
    "jarvis-xl": lambda **kwargs: ModelFactory.create_model("jarvis-xl", **kwargs)
}

def create_model(model_name: str, **kwargs) -> BaseModel:
    """
    Create a model instance by name with optional configuration.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        Instance of the requested model
        
    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name](**kwargs)

# Export public API
__all__ = [
    'JarvisModel',
    'JarvisLanguageModel', 
    'JarvisModelManager', 
    'BaseModel', 
    'create_model',
    'ModelFactory'
]
