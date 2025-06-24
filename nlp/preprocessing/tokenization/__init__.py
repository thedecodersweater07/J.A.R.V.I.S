"""Tokenization module.

This module provides various tokenization utilities for natural language processing.
It includes implementations of different tokenizers that can be used for text processing.
"""

from typing import Dict, Type, Any, Optional, Union, List, Tuple, TypeVar, Generic
import re
import string

# Import tokenizer implementations
try:
    from .word_tokenizer import WordTokenizer
except ImportError:
    from nlp.preprocessing.tokenization.word_tokenizer import WordTokenizer

try:
    from .sentence_tokenizer import SentenceTokenizer
except ImportError:
    from nlp.preprocessing.tokenization.sentence_tokenizer import SentenceTokenizer

try:
    from .regex_tokenizer import RegexTokenizer
except ImportError:
    from nlp.preprocessing.tokenization.regex_tokenizer import RegexTokenizer

try:
    from .whitespace_tokenizer import WhitespaceTokenizer
except ImportError:
    from nlp.preprocessing.tokenization.whitespace_tokenizer import WhitespaceTokenizer

try:
    from .byte_pair_encoding import BPETokenizer
except ImportError:
    from nlp.preprocessing.tokenization.byte_pair_encoding import BPETokenizer

# Type variable for tokenizer classes
T = TypeVar('T')

class Tokenizer(Generic[T]):
    """Base tokenizer class with common functionality."""
    pass

class TokenizerFactory:
    """Factory class for creating tokenizer instances."""
    
    # Map of tokenizer names to their respective classes
    TOKENIZERS = {
        'word': WordTokenizer,
        'sentence': SentenceTokenizer,
        'regex': RegexTokenizer,
        'whitespace': WhitespaceTokenizer,
        'bpe': BPETokenizer,
    }
    
    @classmethod
    def create(cls, 
              tokenizer_type: str = 'word',
              **kwargs) -> Union[WordTokenizer, 
                               SentenceTokenizer, 
                               RegexTokenizer, 
                               WhitespaceTokenizer,
                               BPETokenizer]:
        """Create a tokenizer of the specified type.
        
        Args:
            tokenizer_type: Type of tokenizer to create. Must be one of:
                          - 'word': Word tokenizer (default)
                          - 'sentence': Sentence tokenizer
                          - 'regex': Regex-based tokenizer
                          - 'whitespace': Whitespace tokenizer
                          - 'bpe': Byte Pair Encoding tokenizer
            **kwargs: Additional arguments to pass to the tokenizer constructor
            
        Returns:
            An instance of the specified tokenizer type
            
        Raises:
            ValueError: If an unknown tokenizer type is specified
        """
        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type not in cls.TOKENIZERS:
            raise ValueError(
                f"Unknown tokenizer type: {tokenizer_type}. "
                f"Available types: {', '.join(cls.TOKENIZERS.keys())}"
            )
            
        tokenizer_class = cls.TOKENIZERS[tokenizer_type]
        return tokenizer_class(**kwargs)
    
    @classmethod
    def get_available_tokenizers(cls) -> List[str]:
        """Get a list of available tokenizer types.
        
        Returns:
            List of available tokenizer type names
        """
        return list(cls.TOKENIZERS.keys())


# Create aliases for common tokenizers
word_tokenizer = WordTokenizer
sentence_tokenizer = SentenceTokenizer
regex_tokenizer = RegexTokenizer
whitespace_tokenizer = WhitespaceTokenizer
bpe_tokenizer = BPETokenizer

# Export the most commonly used tokenizers
__all__ = [
    'Tokenizer',
    'TokenizerFactory',
    'WordTokenizer',
    'SentenceTokenizer',
    'RegexTokenizer',
    'WhitespaceTokenizer',
    'BPETokenizer',
    'word_tokenizer',
    'sentence_tokenizer',
    'regex_tokenizer',
    'whitespace_tokenizer',
    'bpe_tokenizer',
]