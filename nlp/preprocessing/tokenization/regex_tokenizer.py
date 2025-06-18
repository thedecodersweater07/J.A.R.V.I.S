"""
Regex Tokenizer Module

This module provides a flexible tokenizer using regular expressions,
allowing for custom tokenization patterns.
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional, Pattern, Union, Callable, Any
import unicodedata

class RegexTokenizer:
    """A flexible tokenizer using regular expressions."""
    
    def __init__(self, 
                 pattern: str = r'\w+', 
                 flags: int = re.UNICODE | re.IGNORECASE,
                 gaps: bool = False,
                 discard_empty: bool = True,
                 lowercase: bool = False,
                 strip_chars: str = None,
                 preserve_case: bool = False):
        """Initialize the regex tokenizer.
        
        Args:
            pattern: Regex pattern for token matching
            flags: Regex flags (e.g., re.IGNORECASE, re.UNICODE)
            gaps: If True, tokenize the spaces between matches
            discard_empty: If True, discard empty tokens
            lowercase: If True, convert tokens to lowercase
            strip_chars: Characters to strip from tokens (or None to keep as is)
            preserve_case: If True, preserve the original case of tokens
        """
        self.pattern = pattern
        self.flags = flags
        self.gaps = gaps
        self.discard_empty = discard_empty
        self.lowercase = lowercase
        self.strip_chars = strip_chars
        self.preserve_case = preserve_case
        
        # Compile the regex pattern
        try:
            self.regex = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        # Default token types for common patterns
        self.token_types = {
            r'\d+': 'NUMBER',
            r'\w+': 'WORD',
            r'\s+': 'WHITESPACE',
            r'[^\w\s]': 'PUNCTUATION'
        }
        
        # Default regex patterns for common token types
        self.patterns = {
            'URL': r'https?://\S+|www\.\S+',
            'EMAIL': r'[\w.+-]+@[\w-]+\.[\w.-]+',
            'HASHTAG': r'#\w+',
            'MENTION': r'@\w+',
            'NUMBER': r'\d+',
            'WORD': r'\w+',
            'WHITESPACE': r'\s+',
            'PUNCTUATION': r'[^\w\s]'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Apply regex
        if self.gaps:
            # Split on the pattern
            tokens = self.regex.split(text)
        else:
            # Find all matches of the pattern
            tokens = self.regex.findall(text)
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            if not token and self.discard_empty:
                continue
                
            # Strip characters if specified
            if self.strip_chars is not None:
                token = token.strip(self.strip_chars)
                if not token and self.discard_empty:
                    continue
            
            # Convert to lowercase if specified
            if self.lowercase and not self.preserve_case:
                token = token.lower()
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        """Tokenize text and return (start, end) spans for each token.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end) tuples for each token
        """
        if not text:
            return []
            
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        spans = []
        for match in self.regex.finditer(text):
            if self.gaps:
                # For gaps, we want the text between matches
                continue
            start, end = match.span()
            spans.append((start, end))
            
        return spans
    
    def tokenize_with_spans(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Tokenize text and return tokens with their spans.
        
        Args:
            text: Input text
            
        Returns:
            List of (token, (start, end)) tuples
        """
        if not text:
            return []
            
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        tokens_with_spans = []
        for match in self.regex.finditer(text):
            if self.gaps:
                # For gaps, we want the text between matches
                continue
                
            token = match.group()
            start, end = match.span()
            
            # Process token
            if self.strip_chars is not None:
                stripped = token.strip(self.strip_chars)
                if not stripped and self.discard_empty:
                    continue
                token = stripped
                
            if self.lowercase and not self.preserve_case:
                token = token.lower()
                
            tokens_with_spans.append((token, (start, end)))
            
        return tokens_with_spans
    
    def tokenize_with_types(self, text: str) -> List[Tuple[str, str]]:
        """Tokenize text and classify each token with a type.
        
        Args:
            text: Input text
            
        Returns:
            List of (token, type) tuples
        """
        if not text:
            return []
            
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        tokens_with_types = []
        
        # First, handle special patterns
        for token_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.UNICODE):
                token = match.group()
                tokens_with_types.append((token, token_type))
                # Replace matched text with spaces to avoid reprocessing
                text = text[:match.start()] + ' ' * len(token) + text[match.end():]
        
        # Then handle remaining text with the main pattern
        for match in self.regex.finditer(text):
            token = match.group()
            if not token.strip() and self.discard_empty:
                continue
                
            # Determine token type
            token_type = 'UNKNOWN'
            for pattern, t_type in self.token_types.items():
                if re.fullmatch(pattern, token, re.UNICODE):
                    token_type = t_type
                    break
            
            tokens_with_types.append((token, token_type))
        
        # Sort by position in original text
        tokens_with_types.sort(key=lambda x: text.find(x[0]))
        
        return tokens_with_types
    
    def add_pattern(self, name: str, pattern: str, token_type: str = None) -> None:
        """Add a custom pattern to the tokenizer.
        
        Args:
            name: Name for the pattern
            pattern: Regex pattern
            token_type: Optional token type for classification
        """
        self.patterns[name] = pattern
        if token_type:
            self.token_types[pattern] = token_type
    
    def set_token_type(self, pattern: str, token_type: str) -> None:
        """Set the token type for a specific pattern.
        
        Args:
            pattern: Regex pattern
            token_type: Token type name
        """
        self.token_types[pattern] = token_type
    
    def __call__(self, text: str) -> List[str]:
        """Alias for tokenize method to make the instance callable."""
        return self.tokenize(text)
