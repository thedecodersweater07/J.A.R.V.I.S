"""
LLM Package for JARVIS

This package provides language model processing capabilities for the JARVIS system,
including text generation, completion, and other LLM-related operations.
"""

import logging
import os
import sys
import warnings
from typing import Optional, Dict, Any, List, Union, Type, TypeVar, TypeVar

# Add the project root to the path to allow absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components with error handling
LLM_AVAILABLE = False
LLMProcessor = None
LLMManager = None
LLMConfig = None
LLMError = None

# Define dummy classes first
class DummyLLMProcessor:
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get('model_name', 'dummy')
    def generate(self, *args, **kwargs):
        return "LLM functionality not available. Please install required dependencies."

class DummyLLMManager:
    def __init__(self, *args, **kwargs):
        self.models = {}

class DummyLLMConfig:
    def __init__(self, *args, **kwargs):
        self.config = {}

class DummyLLMError(Exception):
    pass

# Set defaults
LLM_AVAILABLE = False
LLMProcessor = DummyLLMProcessor
LLMManager = DummyLLMManager
LLMConfig = DummyLLMConfig
LLMError = DummyLLMError

# Try to import core components
try:
    from .llm_processor import LLMProcessor as RealLLMProcessor
    from .exceptions import LLMError as RealLLMError
    
    # Only override if imports are successful
    LLMProcessor = RealLLMProcessor
    LLMError = RealLLMError
    LLM_AVAILABLE = True
    logger.info("LLM package initialized successfully")
    
except ImportError as e:
    logger.warning(f"Could not initialize LLM package: {e}")

# Try to import processors
try:
    from llm.processors import (
        DutchTokenizer,
        DutchParser,
        DutchNER,
        SentimentAnalyzer,
        IntentClassifier,
        CustomDutchTokenizer,
        CustomDutchParser,
        CustomDutchSentimentAnalyzer
    )
    
    # Export processors
    __all__ = [
        'LLMProcessor',
        'LLMManager',
        'LLMConfig',
        'LLMError',
        'LLM_AVAILABLE',
        'DutchTokenizer',
        'DutchParser',
        'DutchNER',
        'SentimentAnalyzer',
        'IntentClassifier',
        'CustomDutchTokenizer',
        'CustomDutchParser',
        'CustomDutchSentimentAnalyzer'
    ]
    
except ImportError as e:
    warnings.warn(f"Could not import processors: {e}")
    __all__ = [
        'LLMProcessor',
        'LLMManager',
        'LLMConfig',
        'LLMError',
        'LLM_AVAILABLE'
    ]