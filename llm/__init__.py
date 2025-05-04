"""
LLM (Language Learning Model) module
Provides continuous learning and knowledge integration capabilities
"""

from .core import LLMCore
from .learning import LearningManager
from .knowledge import KnowledgeManager
from .inference import InferenceEngine

__all__ = [
    'LLMCore',
    'LearningManager', 
    'KnowledgeManager',
    'InferenceEngine'
]