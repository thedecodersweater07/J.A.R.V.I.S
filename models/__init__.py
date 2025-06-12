from typing import Dict, Any
# Relative import to avoid package installation requirement
from .jarvis import JarvisModel, JarvisLanguageModel
from .base import BaseModel

# Model registry
MODELS = {
    "jarvis-small": lambda: JarvisModel("jarvis-small"),
    "jarvis-base": lambda: JarvisModel("jarvis-base"),
    "jarvis-large": lambda: JarvisModel("jarvis-large"),
    "jarvis-xl": lambda: JarvisModel("jarvis-xl")
}

def create_model(model_name: str) -> BaseModel:
    """Create a model instance by name"""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name]()

__all__ = ['JarvisModel', 'JarvisLanguageModel', 'BaseModel', 'create_model']
