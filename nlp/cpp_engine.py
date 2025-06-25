"""Python wrapper for the C++ NLP engine."""

import os
import sys
import platform
import atexit
from typing import Dict, List, Optional, Union, Any, AnyStr, Type, TypeVar, Tuple, Sequence, Mapping

# Type aliases
Language = int
ProcessingMode = int
ProcessingResult = Dict[str, Any]
NLPConfig = Dict[str, Any]

# Try to import the C++ module with proper error handling
try:
    from . import _nlp_engine
    CPP_ENGINE_AVAILABLE = True
    
    # Import types from the C++ module
    Language = _nlp_engine.Language
    ProcessingMode = _nlp_engine.ProcessingMode
    ProcessingResult = _nlp_engine.ProcessingResult
    NLPConfig = _nlp_engine.NLPConfig
    
    # Import constants
    DEFAULT_LANGUAGE = _nlp_engine.DEFAULT_LANGUAGE
    SUPPORTED_LANGUAGES = _nlp_engine.SUPPORTED_LANGUAGES
    
except ImportError as e:
    print(f"Warning: Could not import C++ NLP engine: {e}")
    CPP_ENGINE_AVAILABLE = False
    
    # Define fallback constants
    DEFAULT_LANGUAGE = 0  # Assuming 0 is English
    SUPPORTED_LANGUAGES = ["en"]

# Define stubs for type checking when C++ module is not available
if not CPP_ENGINE_AVAILABLE:
    class _NLEngineStub:
        def __init__(self):
            raise RuntimeError("C++ NLP engine is not available")
            
        def __getattr__(self, name):
            raise RuntimeError("C++ NLP engine is not available")
    
    class _NLPConfigStub(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
    
    # Replace the module with stubs
    _nlp_engine = type('_nlp_engine', (), {
        'NLEngine': _NLEngineStub,
        'NLPConfig': _NLPConfigStub,
        'Language': int,
        'ProcessingMode': int,
        'string_to_language': lambda s: 0,
        'language_to_string': lambda x: "en",
        'DEFAULT_LANGUAGE': DEFAULT_LANGUAGE,
        'SUPPORTED_LANGUAGES': SUPPORTED_LANGUAGES
    })()

class NLPEngine:
    """Python wrapper for the C++ NLP engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NLP engine with optional configuration.
        
        Args:
            config: Optional configuration dictionary or language code string
            
        Raises:
            RuntimeError: If the C++ engine is not available
        """
        if not CPP_ENGINE_AVAILABLE:
            raise RuntimeError("C++ NLP engine is not available")
            
        self._engine = _nlp_engine.NLEngine()
        if config is not None:
            self.initialize(config)
    
    def initialize(self, config: Union[Dict[str, Any], str]) -> bool:
        """Initialize the NLP engine with configuration.
        
        Args:
            config: Either a configuration dictionary or a language code string
            
        Returns:
            bool: True if initialization was successful
        """
        if isinstance(config, str):
            return self._engine.initialize(config)
        else:
            # Convert dict to NLPConfig object
            cfg = _nlp_engine.NLPConfig()
            if 'language' in config:
                if isinstance(config['language'], str):
                    cfg.language = _nlp_engine.string_to_language(config['language'])
                else:
                    cfg.language = config['language']
            if 'mode' in config:
                cfg.mode = config['mode']
            if 'enable_stemming' in config:
                cfg.enable_stemming = bool(config['enable_stemming'])
            if 'enable_entity_extraction' in config:
                cfg.enable_entity_extraction = bool(config['enable_entity_extraction'])
            if 'enable_sentiment_analysis' in config:
                cfg.enable_sentiment_analysis = bool(config['enable_sentiment_analysis'])
            if 'custom_stopwords' in config:
                cfg.custom_stopwords = list(config['custom_stopwords'])
            return self._engine.initialize(cfg)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and return comprehensive results.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing processed text, tokens, entities, etc.
        """
        result = self._engine.process_text(text)
        return {
            'processed_text': result.processed_text,
            'tokens': result.tokens,
            'entities': result.entities,
            'sentiment_scores': dict(result.sentiment_scores),
            'confidence': result.confidence
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self._engine.tokenize(text)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text (lowercase, clean whitespace, etc.).
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        return self._engine.normalize_text(text)
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        return self._engine.extract_entities(text)
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        return _nlp_engine.language_to_string(self._engine.detect_language(text))
    
    def set_language(self, language: str) -> None:
        """Set the processing language.
        
        Args:
            language: Language code (e.g., 'en', 'nl')
        """
        self._engine.set_language(language)
    
    def get_language_name(self) -> str:
        """Get the current language as a string.
        
        Returns:
            Current language name
        """
        return self._engine.get_language_name()
    
    def calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        return self._engine.calculate_sentiment(text)
    
    def get_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        return self._engine.get_keywords(text, max_keywords)
    
    def stem_word(self, word: str) -> str:
        """Stem a single word.
        
        Args:
            word: Input word
            
        Returns:
            Stemmed word
        """
        return self._engine.stem_word(word)
    
    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._engine.is_initialized()
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Current configuration as a dictionary
        """
        cfg = self._engine.get_config()
        return {
            'language': _nlp_engine.language_to_string(cfg.language),
            'mode': cfg.mode,
            'enable_stemming': cfg.enable_stemming,
            'enable_entity_extraction': cfg.enable_entity_extraction,
            'enable_sentiment_analysis': cfg.enable_sentiment_analysis,
            'custom_stopwords': list(cfg.custom_stopwords)
        }
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the engine configuration.
        
        Args:
            config: Configuration dictionary with updated values
        """
        cfg = self._engine.get_config()
        if 'language' in config:
            if isinstance(config['language'], str):
                cfg.language = _nlp_engine.string_to_language(config['language'])
            else:
                cfg.language = config['language']
        if 'mode' in config:
            cfg.mode = config['mode']
        if 'enable_stemming' in config:
            cfg.enable_stemming = bool(config['enable_stemming'])
        if 'enable_entity_extraction' in config:
            cfg.enable_entity_extraction = bool(config['enable_entity_extraction'])
        if 'enable_sentiment_analysis' in config:
            cfg.enable_sentiment_analysis = bool(config['enable_sentiment_analysis'])
        if 'custom_stopwords' in config:
            cfg.custom_stopwords = list(config['custom_stopwords'])
        self._engine.update_config(cfg)
    
    def get_processed_count(self) -> int:
        """Get the number of processed texts."""
        return self._engine.get_processed_count()
    
    def get_average_processing_time(self) -> float:
        """Get the average processing time in milliseconds."""
        return self._engine.get_average_processing_time()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._engine.reset_statistics()

# Create a default instance for convenience
try:
    default_engine: Optional['NLPEngine'] = NLPEngine()
    atexit.register(lambda: default_engine.shutdown() if default_engine else None)
except Exception as e:
    print(f"Warning: Could not initialize default NLP engine: {e}")
    default_engine = None

# Export the most commonly used functions at the module level
# Define wrapper functions that safely handle the case when default_engine is None
def _wrap_method(method_name: str):
    def wrapper(*args, **kwargs):
        if default_engine is None:
            raise RuntimeError("NLP engine is not available")
        return getattr(default_engine, method_name)(*args, **kwargs)
    return wrapper

# Only export methods if we have a working engine
if default_engine is not None:
    process_text = _wrap_method('process_text')
    tokenize = _wrap_method('tokenize')
    normalize_text = _wrap_method('normalize_text')
    extract_entities = _wrap_method('extract_entities')
    detect_language = _wrap_method('detect_language')
    calculate_sentiment = _wrap_method('calculate_sentiment')
    get_keywords = _wrap_method('get_keywords')
    stem_word = _wrap_method('stem_word')
    
    # Cleanup on exit
    def shutdown():
        if default_engine is not None:
            default_engine.shutdown()
else:
    # Create dummy functions that raise an error when called
    def _not_available(*args, **kwargs):
        raise RuntimeError("NLP engine is not available")
    
    process_text = tokenize = normalize_text = extract_entities = _not_available
    detect_language = calculate_sentiment = get_keywords = stem_word = _not_available
    shutdown = lambda: None

# Export enums and constants
Language = _nlp_engine.Language
ProcessingMode = _nlp_engine.ProcessingMode
DEFAULT_LANGUAGE = _nlp_engine.DEFAULT_LANGUAGE
SUPPORTED_LANGUAGES = _nlp_engine.SUPPORTED_LANGUAGES

# Version information
__version__ = getattr(_nlp_engine, "__version__", "0.1.0")

# Clean up
import atexit
atexit.register(lambda: default_engine.shutdown() if hasattr(default_engine, 'shutdown') else None)