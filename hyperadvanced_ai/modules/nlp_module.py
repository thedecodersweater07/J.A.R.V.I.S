"""
Natural Language Processing module for hyperadvanced_ai.

This module provides NLP capabilities for the JARVIS system, including
text processing, intent recognition, and entity extraction.
"""

from typing import Dict, Any, List, Optional
import logging

from hyperadvanced_ai.core.base_module import BaseAIModule, ModuleConfig

# Default configuration for the NLP module
DEFAULT_NLP_CONFIG = ModuleConfig(
    enabled=True,
    debug=False,
    model_name="gpt-3.5-turbo",
    max_tokens=150,
    temperature=0.7,
    stop_sequences=["\n"],
    timeout=30.0
)

class NLPModule(BaseAIModule):
    """Natural Language Processing module for hyperadvanced_ai.
    
    This module provides NLP capabilities including:
    - Text processing and normalization
    - Intent recognition
    - Entity extraction
    - Sentiment analysis
    - Language detection
    """
    
    # Module metadata
    NAME = "nlp_module"
    VERSION = "1.0.0"
    DESCRIPTION = "Natural Language Processing module for hyperadvanced_ai"
    REQUIREMENTS = {
        "nltk": ">=3.8.1",
        "spacy": ">=3.5.0",
        "transformers": ">=4.25.0"
    }
    
    # Default configuration
    DEFAULT_CONFIG = DEFAULT_NLP_CONFIG
    
    def _initialize_impl(self) -> None:
        """Initialize the NLP module with the given configuration."""
        self.logger.info("Initializing NLP module...")
        
        # Initialize NLP models and resources
        self._load_models()
        
        self.logger.info("NLP module initialized successfully")
    
    def _load_models(self) -> None:
        """Load required NLP models and resources."""
        try:
            # Lazy import of heavy dependencies
            import spacy
            from transformers import pipeline
            
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize transformers pipeline for text generation
            self.text_generator = pipeline(
                "text-generation",
                model=self.config.get("model_name", "gpt2"),
                device=-1  # Use CPU by default
            )
            
        except ImportError as e:
            self.logger.error(f"Failed to load required NLP libraries: {e}")
            if self.config.get("debug", False):
                self.logger.exception("Detailed error:")
            raise
    
    def _shutdown_impl(self) -> None:
        """Clean up resources used by the NLP module."""
        self.logger.info("Shutting down NLP module...")
        # Clean up any resources if needed
        self.nlp = None
        self.text_generator = None
    
    def _health_check_impl(self) -> Dict[str, Any]:
        """Check the health of the NLP module."""
        return {
            "status": "healthy",
            "model": self.config.get("model_name", "unknown"),
            "features": ["tokenization", "ner", "sentiment"]
        }
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process the input text and return NLP analysis.
        
        Args:
            text: The input text to process
            
        Returns:
            Dictionary containing the processed results
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        try:
            # Basic text processing
            doc = self.nlp(text)
            
            # Extract entities
            entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                      for ent in doc.ents]
            
            # Tokenize
            tokens = [{"text": token.text, "lemma": token.lemma_, "pos": token.pos_, "dep": token.dep_}
                     for token in doc]
            
            # Basic sentiment analysis (this is a placeholder - in practice you'd use a proper sentiment model)
            sentiment = self._analyze_sentiment(text)
            
            return {
                "text": text,
                "tokens": tokens,
                "entities": entities,
                "sentiment": sentiment,
                "language": doc.lang_,
                "is_processed": True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            if self.config.get("debug", False):
                self.logger.exception("Detailed error:")
            return {
                "text": text,
                "error": str(e),
                "is_processed": False
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of the given text.
        
        This is a simplified implementation. In a real application, you would
        use a proper sentiment analysis model.
        """
        # This is a placeholder - in practice you'd use a proper sentiment model
        positive_words = {"good", "great", "excellent", "happy", "awesome"}
        negative_words = {"bad", "terrible", "awful", "sad", "horrible"}
        
        words = text.lower().split()
        positive = sum(1 for word in words if word in positive_words)
        negative = sum(1 for word in words if word in negative_words)
        
        total = len(words)
        if total == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
        pos_score = positive / total
        neg_score = negative / total
        neutral = max(0, 1.0 - pos_score - neg_score)
        
        return {
            "positive": pos_score,
            "negative": neg_score,
            "neutral": neutral
        }
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments to pass to the text generator
            
        Returns:
            The generated text
        """
        if not hasattr(self, 'text_generator'):
            raise RuntimeError("Text generator not initialized")
            
        try:
            # Get generation parameters from config or use defaults
            max_tokens = kwargs.pop('max_tokens', self.config.get('max_tokens', 150))
            temperature = kwargs.pop('temperature', self.config.get('temperature', 0.7))
            
            # Generate text
            result = self.text_generator(
                prompt,
                max_length=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract the generated text from the result
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
                return result[0].get('text', '')
            return str(result)
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            if self.config.get("debug", False):
                self.logger.exception("Detailed error:")
            raise
