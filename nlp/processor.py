"""
NLP Processor Module for JARVIS

This module provides the main NLP processing capabilities for the JARVIS system,
integrating various NLP components like tokenization, named entity recognition,
sentiment analysis, and more.
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    Main NLP Processor class that handles all NLP-related tasks for JARVIS.
    
    This class serves as the main entry point for all NLP functionality,
    delegating to specialized components as needed.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the NLP Processor.
        
        Args:
            model_name: Name of the NLP model to use
        """
        self.model_name = model_name
        self.components = {}
        logger.info(f"Initialized NLP Processor with model: {model_name}")
    
    def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Process the input text with the configured NLP pipeline.
        
        Args:
            text: Input text to process
            **kwargs: Additional arguments for processing
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Initialize result with basic text info
            result = {
                "text": text,
                "tokens": self.tokenize(text),
                "entities": self.extract_entities(text),
                "sentiment": self.analyze_sentiment(text),
                "intent": self.detect_intent(text),
                "language": self.detect_language(text),
                "success": True
            }
            
            # Add additional metadata
            result["word_count"] = len(result["tokens"])
            result["is_question"] = any(text.strip().endswith(c) for c in ["?", "?"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)
            return {
                "text": text,
                "error": str(e),
                "success": False
            }
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text: Input text to tokenize
            **kwargs: Additional arguments for tokenization
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
            
        # Simple whitespace tokenizer as fallback
        tokens = text.split()
        
        # Remove punctuation from tokens
        import string
        tokens = [token.strip(string.punctuation) for token in tokens if token.strip(string.punctuation)]
        
        return tokens
    
    def analyze_sentiment(self, text: str, **kwargs) -> Dict[str, float]:
        """
        Analyze the sentiment of the input text.
        
        Args:
            text: Input text to analyze
            **kwargs: Additional arguments for sentiment analysis
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {"polarity": 0.0, "subjectivity": 0.0}
            
        # Simple rule-based sentiment analysis
        positive_words = {"good", "great", "excellent", "happy", "awesome", "fantastic"}
        negative_words = {"bad", "terrible", "awful", "sad", "horrible"}
        
        tokens = self.tokenize(text.lower())
        if not tokens:
            return {"polarity": 0.0, "subjectivity": 0.0}
            
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        # Simple polarity score between -1 and 1
        polarity = (positive_count - negative_count) / max(1, len(tokens)) * 2
        polarity = max(-1.0, min(1.0, polarity))  # Clamp between -1 and 1
        
        # Subjectivity based on presence of sentiment words
        subjectivity = (positive_count + negative_count) / max(1, len(tokens))
        
        return {
            "polarity": float(polarity), 
            "subjectivity": float(min(1.0, subjectivity))
        }
    
    def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract named entities from the input text.
        
        Args:
            text: Input text to process
            **kwargs: Additional arguments for entity extraction
            
        Returns:
            List of extracted entities with their types and positions
        """
        if not text:
            return []
            
        entities = []
        tokens = self.tokenize(text)
        
        # Simple rule-based entity extraction
        for i, token in enumerate(tokens):
            # Check for URLs
            if token.startswith(('http://', 'https://', 'www.')):
                entities.append({
                    'text': token,
                    'type': 'URL',
                    'start': text.find(token),
                    'end': text.find(token) + len(token)
                })
            # Check for email addresses
            elif '@' in token and '.' in token.split('@')[-1]:
                entities.append({
                    'text': token,
                    'type': 'EMAIL',
                    'start': text.find(token),
                    'end': text.find(token) + len(token)
                })
            # Check for numbers
            elif token.replace('.', '').isdigit():
                entities.append({
                    'text': token,
                    'type': 'NUMBER',
                    'start': text.find(token),
                    'end': text.find(token) + len(token)
                })
                
        return entities
    
    def detect_language(self, text: str, **kwargs) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to analyze
            **kwargs: Additional arguments for language detection
            
        Returns:
            Detected language code (e.g., 'en', 'nl')
        """
        if not text:
            return 'en'  # Default to English for empty text
            
        # Simple language detection based on common words
        dutch_words = {'de', 'het', 'een', 'en', 'van', 'ik', 'je', 'is', 'zijn', 'te'}
        english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i'}
        
        tokens = set(self.tokenize(text.lower()))
        
        dutch_count = len(tokens & dutch_words)
        english_count = len(tokens & english_words)
        
        if dutch_count > english_count:
            return 'nl'
        return 'en'  # Default to English
        
    def detect_intent(self, text: str, **kwargs) -> str:
        """
        Detect the intent of the input text.
        
        Args:
            text: Input text to analyze
            **kwargs: Additional arguments for intent detection
            
        Returns:
            Detected intent (e.g., 'question', 'statement', 'command')
        """
        if not text:
            return 'unknown'
            
        text = text.strip().lower()
        
        # Check for question
        if text.endswith('?') or text.startswith(('who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'will')):
            return 'question'
            
        # Check for command
        if text.startswith(('go ', 'find ', 'search ', 'show ', 'tell ', 'play ', 'open ', 'close ')) or text.endswith(' please') or text.endswith('!'):
            return 'command'
            
        return 'statement'
