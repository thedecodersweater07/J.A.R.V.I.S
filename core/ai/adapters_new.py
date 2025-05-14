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
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the NLPProcessor adapter with configuration.
        
        Args:
            config: Configuration dictionary for NLP processing
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.processor = None
        
        try:
            # Extract model name from config - NLPProcessor takes model_name not config
            model_name = self.config.get("model", "nl_core_news_sm")
            
            # Initialize the actual processor with the correct parameter
            self.processor = NLPProcessor(model_name=model_name)
            self.logger.info(f"NLPProcessor adapter initialized with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLPProcessor: {e}")
            self.logger.error(traceback.format_exc())
            # Create a fallback processor with a basic model
            self.processor = self._create_fallback_processor()
        
    def _create_fallback_processor(self):
        """Create a fallback processor when initialization fails"""
        try:
            # Try with a minimal model
            self.logger.info("Attempting to initialize NLPProcessor with fallback model")
            return NLPProcessor(model_name="nl_core_news_sm")
        except Exception as e:
            self.logger.error(f"Fallback NLPProcessor initialization failed: {e}")
            # If that fails, create a mock processor for graceful degradation
            self.logger.warning("Creating mock NLP processor for graceful degradation")
            from unittest.mock import MagicMock
            mock = MagicMock()
            # Set up basic mock behavior
            mock.process.return_value = {"error": "NLP processor unavailable"}
            mock.extract_keywords.return_value = []
            mock.classify_text.return_value = {}
            return mock
        
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
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ModelManager adapter with configuration.
        
        Args:
            config: Configuration dictionary for model management
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.manager = None
        
        try:
            # Extract base_path from config - ModelManager takes base_path not config
            base_path = self.config.get("base_path", "../data/ai_training_data/models")
            
            # Make sure the path is absolute
            if not os.path.isabs(base_path):
                # Convert to absolute path relative to project root
                base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", base_path))
            
            # Ensure the directory exists
            Path(base_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize the actual manager with the correct parameter
            self.manager = ModelManager(base_path=base_path)
            self.logger.info(f"ModelManager adapter initialized with base path: {base_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelManager: {e}")
            self.logger.error(traceback.format_exc())
            # Create a fallback manager
            self.manager = self._create_fallback_manager()
    
    def _create_fallback_manager(self):
        """Create a fallback model manager when initialization fails"""
        try:
            # Try with a default path
            default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/models"))
            self.logger.info(f"Attempting to initialize ModelManager with fallback path: {default_path}")
            Path(default_path).mkdir(parents=True, exist_ok=True)
            return ModelManager(base_path=default_path)
        except Exception as e:
            self.logger.error(f"Fallback ModelManager initialization failed: {e}")
            # If that fails, create a mock manager for graceful degradation
            self.logger.warning("Creating mock ModelManager for graceful degradation")
            from unittest.mock import MagicMock
            mock = MagicMock()
            # Set up basic mock behavior
            mock.initialize.return_value = None
            mock.load_models.return_value = {}
            mock.load_model.return_value = None
            mock.get_model.return_value = None
            mock.models = {}
            return mock
        
    def initialize(self):
        """Initialize the model manager."""
        if not self.manager:
            self.logger.error("Cannot initialize: ModelManager not available")
            return None
            
        try:
            return self.manager.initialize()
        except Exception as e:
            self.logger.error(f"Error initializing model manager: {e}")
            return None
        
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
