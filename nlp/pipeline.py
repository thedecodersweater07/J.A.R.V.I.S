"""
Advanced NLP Processing Pipeline for JARVIS

This module implements a comprehensive NLP pipeline that integrates
multiple processing stages including tokenization, entity recognition,
sentiment analysis, and intent classification.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import re
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Enumeration of processing stages."""
    PREPROCESSING = "preprocessing"
    TOKENIZATION = "tokenization"
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    INTENT_CLASSIFICATION = "intent_classification"
    POSTPROCESSING = "postprocessing"

@dataclass
class ProcessingResult:
    """Container for processing results."""
    text: str
    tokens: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Dict[str, float] = field(default_factory=dict)
    intent: Optional[str] = None
    confidence: float = 0.0
    language: str = "en"
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    errors: List[str] = field(default_factory=list)

class Pipeline:
    """
    Advanced NLP Processing Pipeline.
    
    Implements a flexible, configurable pipeline for natural language processing
    with support for multiple processing stages and custom processors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary for pipeline settings
        """
        self.config = config or {}
        self.processors: Dict[ProcessingStage, Callable] = {}
        self.middleware: List[Callable] = []
        self.stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "errors": 0
        }
        
        # Initialize default processors
        self._setup_default_processors()
        logger.info("Pipeline initialized with configuration")
    
    def _setup_default_processors(self):
        """Setup default processing stages."""
        self.register_processor(ProcessingStage.PREPROCESSING, self._preprocess)
        self.register_processor(ProcessingStage.TOKENIZATION, self._tokenize)
        self.register_processor(ProcessingStage.ENTITY_EXTRACTION, self._extract_entities)
        self.register_processor(ProcessingStage.SENTIMENT_ANALYSIS, self._analyze_sentiment)
        self.register_processor(ProcessingStage.INTENT_CLASSIFICATION, self._classify_intent)
        self.register_processor(ProcessingStage.POSTPROCESSING, self._postprocess)
    
    def register_processor(self, stage: ProcessingStage, processor: Callable):
        """
        Register a processor for a specific stage.
        
        Args:
            stage: Processing stage
            processor: Processor function
        """
        self.processors[stage] = processor
        logger.debug(f"Registered processor for stage: {stage.value}")
    
    def add_middleware(self, middleware: Callable):
        """
        Add middleware to the pipeline.
        
        Args:
            middleware: Middleware function
        """
        self.middleware.append(middleware)
        logger.debug("Added middleware to pipeline")
    
    async def process_async(self, text: str, **kwargs) -> ProcessingResult:
        """
        Process text asynchronously through the pipeline.
        
        Args:
            text: Input text to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult object with all processing results
        """
        start_time = time.time()
        result = ProcessingResult(text=text)
        
        try:
            # Apply middleware
            for middleware in self.middleware:
                text = await middleware(text) if asyncio.iscoroutinefunction(middleware) else middleware(text)
            
            # Process through pipeline stages
            stages = kwargs.get('stages', list(ProcessingStage))
            for stage in stages:
                if stage in self.processors:
                    processor = self.processors[stage]
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(result, **kwargs)
                    else:
                        result = processor(result, **kwargs)
                    
                    if not result.success:
                        break
            
            # Update timing and stats
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self._update_stats(processing_time, result.success)
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}", exc_info=True)
            result.success = False
            result.errors.append(str(e))
            self.stats["errors"] += 1
        
        return result
    
    def process(self, text: str, **kwargs) -> ProcessingResult:
        """
        Process text synchronously through the pipeline.
        
        Args:
            text: Input text to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult object with all processing results
        """
        return asyncio.run(self.process_async(text, **kwargs))
    
    def _preprocess(self, result: ProcessingResult, **kwargs) -> ProcessingResult:
        """Preprocess the text."""
        try:
            text = result.text
            
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip()
            
            # Store preprocessed text
            result.text = text
            result.metadata['original_length'] = len(result.text)
            result.metadata['preprocessed_length'] = len(text)
            
        except Exception as e:
            result.errors.append(f"Preprocessing error: {e}")
            logger.error(f"Preprocessing error: {e}")
        
        return result
    
    def _tokenize(self, result: ProcessingResult, **kwargs) -> ProcessingResult:
        """Tokenize the text."""
        try:
            # Simple tokenization (can be replaced with more sophisticated methods)
            tokens = result.text.lower().split()
            
            # Remove punctuation and filter tokens
            tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
            tokens = [token for token in tokens if token.strip()]
            
            result.tokens = tokens
            result.metadata['token_count'] = len(tokens)
            
        except Exception as e:
            result.errors.append(f"Tokenization error: {e}")
            logger.error(f"Tokenization error: {e}")
        
        return result
    
    def _extract_entities(self, result: ProcessingResult, **kwargs) -> ProcessingResult:
        """Extract named entities from text."""
        try:
            # Simple entity extraction (placeholder)
            entities = []
            
            # Basic email detection
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.finditer(email_pattern, result.text)
            for email in emails:
                entities.append({
                    'text': email.group(),
                    'label': 'EMAIL',
                    'start': email.start(),
                    'end': email.end(),
                    'confidence': 0.9
                })
            
            # Basic URL detection
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.finditer(url_pattern, result.text)
            for url in urls:
                entities.append({
                    'text': url.group(),
                    'label': 'URL',
                    'start': url.start(),
                    'end': url.end(),
                    'confidence': 0.95
                })
            
            result.entities = entities
            result.metadata['entity_count'] = len(entities)
            
        except Exception as e:
            result.errors.append(f"Entity extraction error: {e}")
            logger.error(f"Entity extraction error: {e}")
        
        return result
    
    def _analyze_sentiment(self, result: ProcessingResult, **kwargs) -> ProcessingResult:
        """Analyze sentiment of the text."""
        try:
            # Simple sentiment analysis (placeholder)
            text = result.text.lower()
            
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate']
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            # Calculate polarity score
            total_words = len(result.tokens) if result.tokens else 1
            polarity = (pos_count - neg_count) / total_words
            
            # Calculate subjectivity (placeholder)
            subjectivity = min((pos_count + neg_count) / total_words, 1.0)
            
            result.sentiment = {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'positive_words': pos_count,
                'negative_words': neg_count
            }
            
        except Exception as e:
            result.errors.append(f"Sentiment analysis error: {e}")
            logger.error(f"Sentiment analysis error: {e}")
        
        return result
    
    def _classify_intent(self, result: ProcessingResult, **kwargs) -> ProcessingResult:
        """Classify intent of the text."""
        try:
            text = result.text.lower()
            
            # Simple intent classification based on keywords
            intent_patterns = {
                'question': ['what', 'how', 'when', 'where', 'why', 'who', '?'],
                'request': ['please', 'can you', 'could you', 'would you'],
                'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
                'goodbye': ['bye', 'goodbye', 'see you', 'farewell'],
                'command': ['do', 'execute', 'run', 'start', 'stop']
            }
            
            intent_scores = {}
            for intent, patterns in intent_patterns.items():
                score = sum(1 for pattern in patterns if pattern in text)
                if score > 0:
                    intent_scores[intent] = score / len(patterns)
            
            if intent_scores:
                best_intent = max(intent_scores, key=intent_scores.get)
                confidence = intent_scores[best_intent]
                
                result.intent = best_intent
                result.confidence = confidence
                result.metadata['intent_scores'] = intent_scores
            
        except Exception as e:
            result.errors.append(f"Intent classification error: {e}")
            logger.error(f"Intent classification error: {e}")
        
        return result
    
    def _postprocess(self, result: ProcessingResult, **kwargs) -> ProcessingResult:
        """Postprocess the results."""
        try:
            # Add final metadata
            result.metadata['pipeline_version'] = '1.0.0'
            result.metadata['processed_at'] = time.time()
            
            # Validate results
            if not result.tokens and result.text:
                result.errors.append("No tokens generated from text")
            
        except Exception as e:
            result.errors.append(f"Postprocessing error: {e}")
            logger.error(f"Postprocessing error: {e}")
        
        return result
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update pipeline statistics."""
        self.stats["total_processed"] += 1
        self.stats["total_time"] += processing_time
        self.stats["average_time"] = self.stats["total_time"] / self.stats["total_processed"]
        
        if not success:
            self.stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.stats.copy()
    
    def export_config(self) -> Dict[str, Any]:
        """Export current pipeline configuration."""
        return {
            'processors': list(self.processors.keys()),
            'middleware_count': len(self.middleware),
            'config': self.config,
            'stats': self.get_stats()
        }

# Example usage and testing
if __name__ == "__main__":
    # Create pipeline
    pipeline = Pipeline()
    
    # Test processing
    test_texts = [
        "Hello, how are you doing today?",
        "Please send me the report at john@example.com",
        "This is a terrible experience, I hate it!",
        "Can you help me with this task?"
    ]
    
    print("=== NLP Pipeline Test Results ===")
    for text in test_texts:
        result = pipeline.process(text)
        print(f"\nInput: {text}")
        print(f"Tokens: {result.tokens}")
        print(f"Entities: {len(result.entities)} found")
        print(f"Sentiment: {result.sentiment.get('polarity', 0):.2f}")
        print(f"Intent: {result.intent} (confidence: {result.confidence:.2f})")
        print(f"Processing time: {result.processing_time:.4f}s")
        
        if result.errors:
            print(f"Errors: {result.errors}")
    
    # Show pipeline stats
    print(f"\n=== Pipeline Statistics ===")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")