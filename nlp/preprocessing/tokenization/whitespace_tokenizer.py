"""
Whitespace Tokenizer Module

This module provides a simple whitespace-based tokenizer that splits text on whitespace.
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional, Union, Pattern
import unicodedata

class WhitespaceTokenizer:
    """A simple whitespace-based tokenizer with basic preprocessing options."""
    
    def __init__(self, 
                 split_on_newline: bool = True,
                 split_on_whitespace: bool = True,
                 lowercase: bool = False,
                 preserve_case: bool = False,
                 strip_chars: Optional[str] = None,
                 discard_empty: bool = True):
        """Initialize the whitespace tokenizer.
        
        Args:
            split_on_newline: If True, split on newline characters
            split_on_whitespace: If True, split on any whitespace
            lowercase: If True, convert tokens to lowercase
            preserve_case: If True, preserve the original case of tokens
            strip_chars: Characters to strip from tokens (or None to keep as is)
            discard_empty: If True, discard empty tokens
        """
        self.split_on_newline = split_on_newline
        self.split_on_whitespace = split_on_whitespace
        self.lowercase = lowercase
        self.preserve_case = preserve_case
        self.strip_chars = strip_chars
        self.discard_empty = discard_empty
        
        # Compile regex patterns for splitting
        self.patterns = []
        
        if split_on_newline:
            self.patterns.append(r'\n+')
            
        if split_on_whitespace:
            self.patterns.append(r'\s+')
        
        # Default to splitting on any whitespace if no patterns specified
        if not self.patterns:
            self.patterns = [r'\s+']
        
        # Compile the final pattern
        self.pattern = re.compile('|'.join(f'({p})' for p in self.patterns))
    
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
        
        # Split text using the compiled pattern
        tokens = []
        start = 0
        
        for match in self.pattern.finditer(text):
            # Get the text before the match
            if match.start() > start:
                token = text[start:match.start()]
                processed = self._process_token(token)
                if processed is not None:
                    tokens.append(processed)
            
            # Move past this match
            start = match.end()
        
        # Add the remaining text after the last match
        if start < len(text):
            token = text[start:]
            processed = self._process_token(token)
            if processed is not None:
                tokens.append(processed)
        
        return tokens
    
    def _process_token(self, token: str) -> Optional[str]:
        """Process a single token with the configured options.
        
        Args:
            token: Token to process
            
        Returns:
            Processed token or None if it should be discarded
        """
        # Strip characters if specified
        if self.strip_chars is not None:
            token = token.strip(self.strip_chars)
            if not token and self.discard_empty:
                return None
        
        # Convert to lowercase if specified
        if self.lowercase and not self.preserve_case:
            token = token.lower()
        
        return token
    
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
        start = 0
        
        for match in self.pattern.finditer(text):
            # Get the text before the match
            if match.start() > start:
                spans.append((start, match.start()))
            
            # Move past this match
            start = match.end()
        
        # Add the remaining text after the last match
        if start < len(text):
            spans.append((start, len(text)))
            
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
        start = 0
        
        for match in self.pattern.finditer(text):
            # Get the text before the match
            if match.start() > start:
                token = text[start:match.start()]
                processed = self._process_token(token)
                if processed is not None:
                    tokens_with_spans.append((processed, (start, match.start())))
            
            # Move past this match
            start = match.end()
        
        # Add the remaining text after the last match
        if start < len(text):
            token = text[start:]
            processed = self._process_token(token)
            if processed is not None:
                tokens_with_spans.append((processed, (start, len(text))))
        
        return tokens_with_spans
    
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of tokenized texts
        """
        return [self.tokenize(text) for text in texts]
    
    def __call__(self, text: str) -> List[str]:
        """Alias for tokenize method to make the instance callable."""
        return self.tokenize(text)
