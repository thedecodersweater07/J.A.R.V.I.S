"""
Dutch Parser Module

This module provides Dutch language parsing functionality using a custom implementation
that doesn't rely on external NLP libraries like spaCy.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)

# Try to import the custom parser
try:
    from .custom_nlp import DutchParser as CustomDutchParser
    CUSTOM_PARSER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Custom Dutch parser not available: {e}")
    CUSTOM_PARSER_AVAILABLE = False

class DutchParser:
    """
    A wrapper around the custom Dutch parser implementation.
    
    This class provides a simple interface for parsing Dutch text using a custom
    implementation that doesn't rely on spaCy or other heavy dependencies.
    """
    
    def __init__(self):
        """Initialize the Dutch parser with a custom implementation."""
        self._available = CUSTOM_PARSER_AVAILABLE
        self.parser = CustomDutchParser() if self._available else None
        
        if not self._available:
            logger.warning("Custom Dutch parser is not available. Using fallback implementation.")
    
    def __call__(self, tokens: List[str]) -> Dict[str, Any]:
        """Parse a list of tokens."""
        text = " ".join(tokens)
        return self.parse(text)
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse the given Dutch text.
        
        Args:
            text: The input text to parse.
            
        Returns:
            A dictionary containing parsed information such as tokens, sentences,
            dependencies, and noun chunks. If the parser is not available, returns
            a dictionary with basic tokenization.
        """
        if not text:
            return self._empty_result()
            
        if not self._available or not self.parser:
            return self._fallback_parse(text)
            
        try:
            result = self.parser.parse(text)
            return {
                'tokens': result.get('tokens', text.split()),
                'sentences': result.get('sentences', [text]),
                'dependencies': [(dep[0], dep[1], dep[2]) for dep in result.get('dependencies', [])],
                'noun_chunks': result.get('noun_chunks', [])
            }
        except Exception as e:
            logger.error(f"Error during text parsing: {e}")
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Fallback implementation when custom parser is not available."""
        # Simple sentence splitting on common sentence boundaries
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            sentences = [text]
            
        # Simple tokenization
        tokens = text.split()
        
        # Very basic noun phrase detection (just simple patterns)
        noun_chunks = []
        words = text.lower().split()
        for i in range(len(words) - 1):
            # Simple pattern: determiner + adjective + noun
            if i < len(words) - 2 and words[i] in {'de', 'het', 'een'} and words[i+1].endswith(('e', 'en')):
                noun_chunks.append(' '.join(words[i:i+3]))
        
        return {
            'tokens': tokens,
            'sentences': sentences,
            'dependencies': [],
            'noun_chunks': noun_chunks
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return an empty parse result."""
        return {
            'tokens': [],
            'sentences': [],
            'dependencies': [],
            'noun_chunks': []
        }
    
    @property
    def is_available(self) -> bool:
        """Check if the Dutch parser is available for use."""
        return self._available
