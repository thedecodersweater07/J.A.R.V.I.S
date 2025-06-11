import os
import sys
import logging
import traceback
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import core components
from core.logging import get_logger
from .models.jarvis import JarvisModel  # Import from the models directory


try:
    # Ensure the models directory is in the Python path
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / "models/jarvis"
    if models_dir not in sys.path:
        sys.path.append(str(models_dir))
except Exception as e:
    logging.error(f"Failed to add models directory to path: {e}")
    sys.exit(1)

class MLComponentsManager:
    """Manager class for ML components initialization and management"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.model = None
        self.ai_coordinator = None
        self.model_manager = None
        self.nlp_processor = None
        
        # Initialize components
        self.initialize()
    
    def initialize(self):
        """Initialize all ML components"""
        try:
            self._init_ml_components()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ML components: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _init_ml_components(self):
        """Initialize machine learning components with improved error handling"""
        try:
            # Initialize model with error handling
            try:
                model_name = self.config.get("model", "jarvis-base")
                self.model = JarvisModel(model_name)
                
                # Verify CUDA availability
                if torch.cuda.is_available():
                    try:
                        self.model = self.model.to('cuda')
                        self.logger.info("Using CUDA for model acceleration")
                    except Exception as cuda_error:
                        self.logger.warning(f"Failed to move model to CUDA: {cuda_error}")
                        self.logger.info("Continuing with CPU model")
            except Exception as model_error:
                self.logger.error(f"Failed to initialize main model: {model_error}")
                self.model = None
                
            # Initialize AI Coordinator with improved error handling
            try:
                from core.ai.coordinator import AICoordinator
                
                # Create a comprehensive config for the coordinator
                ai_config = {
                    "llm": self.config.get("llm", {}),
                    "nlp": self.config.get("nlp", {}),
                    "ml": {
                        "base_path": os.path.abspath("data/models")
                    },
                    "resources": self.config.get("resources", {}),
                    "events": self.config.get("events", {}),
                    "models": self.config.get("models", {})
                }
                
                # Initialize the AI coordinator
                self.ai_coordinator = AICoordinator(config=ai_config)
                self.ai_coordinator.initialize()
                
                # Get components from the coordinator
                self.model_manager = self.ai_coordinator.get_component("model_manager")
                self.nlp_processor = self.ai_coordinator.get_component("nlp")
                
                # Verify components were properly initialized
                if self.model_manager and self.nlp_processor:
                    self.logger.info("AI Coordinator initialized successfully with all required components")
                else:
                    missing = []
                    if not self.model_manager:
                        missing.append("model_manager")
                    if not self.nlp_processor:
                        missing.append("nlp")
                    self.logger.warning(f"AI Coordinator initialized but missing components: {', '.join(missing)}")
            except Exception as ai_error:
                self.logger.error(f"AI Coordinator initialization failed: {ai_error}")
                self.ai_coordinator = None
                
                # Fall back to direct initialization if coordinator fails
                try:
                    # Initialize model manager directly with correct parameters
                    from ml.model_manager import ModelManager
                    model_path = os.path.abspath("data/models")
                    # Ensure the model directory exists
                    os.makedirs(model_path, exist_ok=True)
                    self.model_manager = ModelManager(base_path=model_path)
                    self.logger.info(f"ModelManager initialized directly with path: {model_path}")
                except Exception as mm_error:
                    self.logger.error(f"Direct ModelManager initialization failed: {mm_error}")
                    self.logger.error(traceback.format_exc())
                    self.model_manager = None
                
                try:
                    # Initialize NLP processor directly with correct parameters
                    from nlp.processor import NLPProcessor
                    nlp_model = self.config.get("nlp", {}).get("model", "nl_core_news_sm")
                    self.nlp_processor = NLPProcessor(model_name=nlp_model)
                    self.logger.info(f"NLPProcessor initialized directly with model: {nlp_model}")
                except Exception as nlp_error:
                    self.logger.error(f"Direct NLPProcessor initialization failed: {nlp_error}")
                    self.logger.error(traceback.format_exc())
                    # Try with fallback model
                    try:
                        self.logger.info("Attempting to initialize NLPProcessor with fallback model")
                        self.nlp_processor = NLPProcessor(model_name="nl_core_news_sm")
                        self.logger.info("NLPProcessor initialized with fallback model")
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback NLPProcessor initialization also failed: {fallback_error}")
                        self.nlp_processor = None
            
            # Test pipeline if main model is available
            if self.model:
                try:
                    text = "Initialize system check"
                    tasks = ["classification", "generation", "qa"]
                    results = self.model.process_pipeline(text, tasks)
                    self.logger.info("Pipeline test successful")
                except Exception as pipeline_error:
                    self.logger.warning(f"Pipeline test failed: {pipeline_error}, continuing initialization")
            
            # Log initialization status
            components_status = []
            if self.model:
                components_status.append("JarvisModel")
            if self.ai_coordinator:
                components_status.append("AICoordinator")
            if self.model_manager:
                components_status.append("ModelManager")
            if self.nlp_processor:
                components_status.append("NLPProcessor")
                
            if components_status:
                self.logger.info(f"ML components initialized: {', '.join(components_status)}")
            else:
                self.logger.warning("No ML components were successfully initialized")
                
        except Exception as e:
            self.logger.error(f"ML initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            # Don't raise the exception to allow the system to continue with limited functionality
            self.logger.warning("Continuing with limited ML functionality")
    
    def get_model_manager(self):
        """Get the model manager instance"""
        return self.model_manager
    
    def get_nlp_processor(self):
        """Get the NLP processor instance"""
        return self.nlp_processor
    
    def get_model(self):
        """Get the main model instance"""
        return self.model
    
    def process_text(self, text, tasks=None):
        """Process text using available ML components"""
        if not text:
            return {"error": "No text provided"}
            
        results = {}
        
        # Use NLP processor if available
        if self.nlp_processor:
            try:
                nlp_results = self.nlp_processor.process(text)
                results["nlp"] = nlp_results
            except Exception as e:
                self.logger.error(f"Error in NLP processing: {e}")
                results["nlp"] = {"error": str(e)}
        
        # Use main model if available
        if self.model and tasks:
            try:
                model_results = self.model.process_pipeline(text, tasks)
                results["model"] = model_results
            except Exception as e:
                self.logger.error(f"Error in model processing: {e}")
                results["model"] = {"error": str(e)}
        
        return results


# For direct usage
if __name__ == "__main__":
    # Initialize the ML components manager
    ml_manager = MLComponentsManager()
    
    # Test processing
    if ml_manager.nlp_processor:
        print("Testing NLP processor...")
        result = ml_manager.nlp_processor.process("Test the NLP processor functionality")
        print(f"NLP Result: {result}")
    else:
        print("NLP processor not available")
        
    print("\nML Components initialization complete.")

