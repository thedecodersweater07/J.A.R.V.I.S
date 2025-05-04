"""Machine Learning module for Jarvis"""

from .training.trainers import ModelTrainer
from .feature_engineering import FeatureExtractor, FeatureSelector, FeatureTransformer
from .model_manager import ModelManager

__all__ = [
    'ModelTrainer',
    'FeatureExtractor',
    'FeatureSelector',
    'FeatureTransformer',
    'ModelManager'
]

"""
LLM (Language Learning Model) module
Provides continuous learning and knowledge integration capabilities
"""

from .core.llm_core import LLMCore
from .learning import LearningManager
from .knowledge import KnowledgeManager
from .inference import InferenceEngine

__all__ = [
    'LLMCore',
    'LearningManager', 
    'KnowledgeManager',
    'InferenceEngine'
]
