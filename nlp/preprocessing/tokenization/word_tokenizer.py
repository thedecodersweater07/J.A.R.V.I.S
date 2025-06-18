"""
Word Tokenizer Module

This module provides functionality for tokenizing text into words and sentences,
handling various edge cases and special characters.
"""

import re
from typing import List, Tuple, Dict, Set, Optional, Union, Pattern
import string
import unicodedata

class WordTokenizer:
    """Tokenizes text into words and sentences with various options."""
    
    def __init__(self, 
                 split_on_whitespace: bool = False,
                 preserve_case: bool = False,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = False,
                 custom_stopwords: Optional[Set[str]] = None):
        """Initialize the word tokenizer.
        
        Args:
            split_on_whitespace: If True, split only on whitespace, otherwise use regex
            preserve_case: If False, convert text to lowercase
            remove_punctuation: If True, remove all punctuation
            remove_numbers: If True, remove all numbers
            remove_stopwords: If True, remove common stopwords
            custom_stopwords: Set of custom stopwords to remove
        """
        self.split_on_whitespace = split_on_whitespace
        self.preserve_case = preserve_case
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        
        # Default English stopwords
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
            "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
            'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
            "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', 
            "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Compile regex patterns
        self.word_tokenize_pattern = re.compile(
            r"\w+[']?\w*|\S"  # Matches words with optional internal apostrophes or single non-whitespace
        )
        
        self.sentence_tokenize_pattern = re.compile(
            r'(?<![A-Z][a-z]\.)(?<=\S[.!?])\s+'  # Matches sentence boundaries
        )
        
        self.punctuation = set(string.punctuation)
        self.digits = set(string.digits)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Normalize unicode and strip whitespace
        text = unicodedata.normalize('NFKC', text).strip()
        
        # Convert to lowercase if needed
        if not self.preserve_case:
            text = text.lower()
        
        # Split into tokens
        if self.split_on_whitespace:
            tokens = text.split()
        else:
            tokens = self.word_tokenize_pattern.findall(text)
        
        # Apply filters
        if self.remove_punctuation or self.remove_numbers or self.remove_stopwords:
            filtered_tokens = []
            for token in tokens:
                # Skip empty tokens
                if not token:
                    continue
                    
                # Check if token is punctuation or number
                if (self.remove_punctuation and all(c in self.punctuation for c in token)) or \
                   (self.remove_numbers and token.isdigit()):
                    continue
                
                # Remove punctuation from start/end of tokens
                if self.remove_punctuation:
                    token = token.strip(''.join(self.punctuation))
                    if not token:  # If token becomes empty after stripping
                        continue
                
                # Skip stopwords if enabled
                if self.remove_stopwords and token.lower() in self.stopwords:
                    continue
                    
                filtered_tokens.append(token)
                
            tokens = filtered_tokens
        
        return tokens
    
    def sent_tokenize(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text to split into sentences
            
        Returns:
            List of sentences
        """
        if not text:
            return []
            
        # Normalize unicode and strip whitespace
        text = unicodedata.normalize('NFKC', text).strip()
        
        # Split into sentences
        sentences = self.sentence_tokenize_pattern.split(text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        """Tokenize text and return (start, end) spans for each token.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end) tuples for each token
        """
        tokens = self.tokenize(text)
        spans = []
        current_pos = 0
        
        for token in tokens:
            # Find the token in the remaining text
            start = text.find(token, current_pos)
            if start == -1:  # Token not found (shouldn't happen with proper tokenization)
                continue
                
            end = start + len(token)
            spans.append((start, end))
            current_pos = end
            
        return spans
    
    def tokenize_sentences(self, text: str) -> List[List[str]]:
        """Tokenize text into sentences and then into words.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences, where each sentence is a list of tokens
        """
        sentences = self.sent_tokenize(text)
        return [self.tokenize(sent) for sent in sentences]
    
    def add_stopwords(self, words: Union[str, List[str], Set[str]]) -> None:
        """Add words to the stopword list.
        
        Args:
            words: Word or list/set of words to add to stopwords
        """
        if isinstance(words, str):
            words = [words]
        self.stopwords.update(words)
    
    def remove_stopword(self, word: str) -> bool:
        """Remove a word from the stopword list.
        
        Args:
            word: Word to remove from stopwords
            
        Returns:
            True if word was removed, False if not found
        """
        if word in self.stopwords:
            self.stopwords.remove(word)
            return True
        return False
    
    def set_language(self, language: str) -> None:
        """Set the language for tokenization (placeholder for future multilingual support).
        
        Args:
            language: Language code (e.g., 'en', 'nl')
        """
        # This is a placeholder for future language-specific tokenization rules
        self.language = language
    
    def __call__(self, text: str) -> List[str]:
        """Alias for tokenize method to make the instance callable."""
        return self.tokenize(text)
