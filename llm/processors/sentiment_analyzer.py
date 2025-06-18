"""
Sentiment Analyzer Module

This module provides sentiment analysis functionality using a custom Dutch sentiment analyzer.
"""

from typing import Dict, Optional, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Try to import the custom analyzer
try:
    from .custom_nlp import DutchSentimentAnalyzer as CustomSentimentAnalyzer
    CUSTOM_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Custom sentiment analyzer not available: {e}")
    CUSTOM_ANALYZER_AVAILABLE = False

class SentimentAnalyzer:
    """
    A wrapper around the custom Dutch sentiment analyzer.
    
    This class provides a simple interface for performing sentiment analysis
    on Dutch text using a custom implementation that doesn't rely on spaCy.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer with a custom Dutch implementation."""
        self._available = CUSTOM_ANALYZER_AVAILABLE
        self.analyzer = CustomSentimentAnalyzer() if self._available else None
        
        if not self._available:
            logger.warning("Custom sentiment analyzer is not available. Using fallback implementation.")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The input text to analyze.
            
        Returns:
            A dictionary containing sentiment scores for positive, negative, and neutral sentiments.
            If the analyzer is not available, returns neutral scores.
        """
        if not text:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
        if not self._available or not self.analyzer:
            # Fallback implementation when custom analyzer is not available
            return self._fallback_analyze(text)
            
        try:
            return self.analyzer.analyze(text)
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            return self._fallback_analyze(text)
    
    def _fallback_analyze(self, text: str) -> Dict[str, float]:
        """Fallback implementation when custom analyzer is not available."""
        # Simple rule-based fallback
        positive_words = {"goed", "mooi", "leuk", "fijn", "blij", "geweldig", "perfect"}
        negative_words = {"slecht", "lelijk", "vervelend", "moeilijk", "verdrietig", "teleurgesteld"}
        
        words = text.lower().split()
        positive = sum(1 for word in words if word in positive_words)
        negative = sum(1 for word in words if word in negative_words)
        total = len(words) or 1
        
        # Normalize scores
        positive_score = min(1.0, positive / total * 2)  # Scale to make scores more pronounced
        negative_score = min(1.0, negative / total * 2)
        neutral_score = max(0, 1 - positive_score - negative_score)
        
        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }
    
    @property
    def is_available(self) -> bool:
        """Check if the sentiment analyzer is available for use."""
        return self._available
