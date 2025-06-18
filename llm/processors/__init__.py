"""
Processors Package

This package contains various text processing components for the JARVIS system,
including tokenization, parsing, sentiment analysis, and intent classification.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Type, TypeVar, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core processors
try:
    from .dutch_tokenizer import DutchTokenizer
    from .dutch_parser import DutchParser
    from .dutch_ner import DutchNER
    from .sentiment_analyzer import SentimentAnalyzer
    from .intent_classifier import IntentClassifier
    
    # Import custom NLP implementations if available
    try:
        from .custom_nlp import (
            DutchTokenizer as CustomDutchTokenizer,
            DutchParser as CustomDutchParser,
            DutchSentimentAnalyzer as CustomDutchSentimentAnalyzer
        )
        CUSTOM_NLP_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Custom NLP implementations not available: {e}")
        CUSTOM_NLP_AVAILABLE = False
        
        # Define dummy classes for type checking
        class CustomDutchTokenizer: pass
        class CustomDutchParser: pass
        class CustomDutchSentimentAnalyzer: pass
    
    # Export public API
    __all__ = [
        'DutchTokenizer',
        'DutchParser',
        'DutchNER',
        'SentimentAnalyzer',
        'IntentClassifier',
    ]
    
    # Add custom implementations to __all__ if available
    if CUSTOM_NLP_AVAILABLE:
        __all__.extend([
            'CustomDutchTokenizer',
            'CustomDutchParser',
            'CustomDutchSentimentAnalyzer',
        ])

except ImportError as e:
    logger.error(f"Failed to import core processors: {e}")
    
    # Define dummy classes when imports fail
    class DutchTokenizer: 
        def tokenize(self, text: str) -> List[str]:
            return text.split()
            
    class DutchParser:
        def parse(self, text: str) -> Dict[str, Any]:
            return {
                'tokens': text.split(),
                'sentences': [text],
                'dependencies': [],
                'noun_chunks': []
            }
            
    class DutchNER:
        def extract_entities(self, text: str) -> List[Dict[str, Any]]:
            return []
            
    class SentimentAnalyzer:
        def analyze(self, text: str) -> Dict[str, float]:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
    class IntentClassifier:
        def classify(self, text: str) -> Dict[str, float]:
            return {"intent": "unknown", "confidence": 0.0}
    
    # Dummy custom implementations
    class CustomDutchTokenizer: pass
    class CustomDutchParser: pass
    class CustomDutchSentimentAnalyzer: pass
    
    __all__ = [
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
    import warnings
    warnings.warn(f"Failed to import some processors: {e}")
