"""
AI Component Adapters Module
Provides adapter classes to make existing components compatible with the new AI architecture.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import traceback

# Import core components
from core.logging import get_logger

# Import components that need adapters
from nlp.processor import NLPProcessor
from ml.model_manager import ModelManager

class NLPProcessorAdapter:
    """
    Adapter for NLPProcessor to make it compatible with the AI Coordinator.
    The adapter handles the different constructor signature of NLPProcessor.
    """
    
    def __init__(self, config: Dict[str, Any] = None, model_name: str = None):
        """
        Initialize the NLPProcessor adapter with configuration.
        
        Args:
            config: Configuration dictionary for NLP processing
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Handle both config and direct model_name initialization
        if model_name:
            self.model = model_name
        else:
            self.model = self.config.get('model', 'nl_core_news_sm')
        
        try:
            self.processor = NLPProcessor(model_name=self.model)
            self.logger.info(f"NLPProcessor adapter initialized with model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLPProcessor: {e}")
            self.processor = self._create_fallback_processor()

    def initialize(self):
        """Initialize the NLP processor"""
        try:
            self.processor = NLPProcessor(model_name=self.model)
            self.logger.info(f"NLPProcessor adapter initialized with model: {self.model}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize NLPProcessor: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text with NLP pipeline.
        
        Args:
            text: Text to process
            context: Additional context for processing
            
        Returns:
            Dictionary with processing results
        """
        if not self.processor:
            return {"error": "NLP processor not initialized", "text": text}
            
        try:
            return self.processor.process(text, context)
        except Exception as e:
            self.logger.error(f"Error in NLP processing: {e}")
            return {"error": str(e), "text": text}
        
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to extract
            
        Returns:
            List of keywords
        """
        if not self.processor:
            return []
            
        try:
            return self.processor.extract_keywords(text, top_n)
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
        
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """
        Classify text into categories.
        
        Args:
            text: Text to classify
            categories: List of categories
            
        Returns:
            Dictionary with category scores
        """
        if not self.processor:
            return {category: 0.0 for category in categories}
            
        try:
            return self.processor.classify_text(text, categories)
        except Exception as e:
            self.logger.error(f"Error classifying text: {e}")
            return {category: 0.0 for category in categories}


class ModelManagerAdapter:
    """
    Adapter for ModelManager to make it compatible with the AI Coordinator.
    The adapter handles the different constructor signature of ModelManager.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelManager adapter with configuration.
        
        Args:
            config: Configuration dictionary for model management
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.manager = None
        
    def initialize(self):
        """Initialize the model manager"""
        try:
            base_path = self.config.get('base_path', 'data/models')
            self.manager = ModelManager(base_path=base_path)
            self.logger.info(f"ModelManager adapter initialized with base path: {base_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelManager: {e}")
            self.logger.error(traceback.format_exc())
            return False
        
    def load_models(self):
        """Load all required models."""
        if not self.manager:
            self.logger.error("Cannot load models: ModelManager not available")
            return {}
            
        try:
            return self.manager.load_models()
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return {}
        
    def load_model(self, model_name: str, model_path: str = None) -> Any:
        """
        Load a specific model.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model
            
        Returns:
            Loaded model
        """
        if not self.manager:
            self.logger.error(f"Cannot load model {model_name}: ModelManager not available")
            return None
            
        try:
            if model_path:
                return self.manager.load_model(model_name, model_path)
            else:
                # Handle case where model_path is not provided
                return self.manager.load_model(model_name)
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None
        
    def get_model(self, model_name: str) -> Any:
        """
        Get a loaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        if not self.manager:
            self.logger.error(f"Cannot get model {model_name}: ModelManager not available")
            return None
            
        try:
            # Check if models dictionary exists
            if hasattr(self.manager, 'models'):
                return self.manager.models.get(model_name)
            return None
        except Exception as e:
            self.logger.error(f"Error getting model {model_name}: {e}")
            return None
        
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save a model.
        
        Args:
            model: Model instance
            model_name: Name of the model
            metadata: Optional metadata
            
        Returns:
            Path to saved model
        """
        if not self.manager:
            self.logger.error(f"Cannot save model {model_name}: ModelManager not available")
            return None
            
        try:
            return self.manager.save_model(model, model_name, metadata)
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
            return None
