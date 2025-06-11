"""
JARVIS Tests Module
==================

This module provides test classes and functions for the JARVIS system.
"""

from .base import BaseModel
from .config import JarvisConfig, JARVIS_CONFIGS
from .jarvis import (
    JarvisModel, 
    JarvisLanguageModel, 
    JarvisNLPModel, 
    JarvisMLModel,
    JarvisModelManager,
    create_jarvis_model,
    ModelMetrics,
    JarvisError,
    ModelLoadError,
    ProcessingError
)

__all__ = [
    # Base classes
    "BaseModel",
    "JarvisConfig", 
    "JARVIS_CONFIGS",
    
    # Core models
    "JarvisModel",
    "JarvisLanguageModel",
    "JarvisNLPModel", 
    "JarvisMLModel",
    
    # Management
    "JarvisModelManager",
    "create_jarvis_model",
    "ModelMetrics",
    "JarvisError",
    "ModelLoadError",
    "ProcessingError"
]
# Ensure all necessary components are imported for testing
# This allows for easy access to all test-related functionalities
# when using `from tests import *`
# This module serves as a central point for importing all test-related classes and functions
# from the JARVIS system, making it easier to manage and maintain tests.
# The __all__ list defines the public API of this module, allowing users to import