import torch
import logging
from pathlib import Path
import yaml
import spacy
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class LLMCore:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM Core"""
        try:
            # Load and validate config
            self.config = self._load_config(config)
            self._validate_config()
            
            # Extract configs
            model_config = self.config.get("model", {})
            
            # Initialize components
            self._initialize_model()
            logger.info("LLMCore initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMCore: {str(e)}", exc_info=True)
            raise

    def _validate_config(self) -> None:
        """Validate minimum required configuration"""
        if not self.config:
            raise ValueError("No configuration provided")
            
        if "model" not in self.config:
            raise ValueError("Model configuration missing")
            
        if "name" not in self.config["model"]:
            raise ValueError("Model name not specified in configuration")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "model": {
                "name": "nl_core_news_lg",
                "type": "spacy"
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)
                return {**default_config, **loaded_config}
                
        return default_config

    def _initialize_model(self) -> None:
        """Initialize spaCy model"""
        try:
            model_name = self.config.get("model", {}).get("name", "nl_core_news_lg")
            self.model = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        # Get relevant context from memory
        context_data = self.memory.get_context(prompt) if context is None else context
        
        # Prepare input with context
        enhanced_prompt = self._prepare_prompt(prompt, context_data)
        
        # Generate response
        doc = self.model(enhanced_prompt)
        response = " ".join([token.text for token in doc])
        
        # Store interaction in memory
        self.memory.store_interaction(prompt, response, context_data)
        
        return response

    def _prepare_prompt(self, prompt: str, context: Dict) -> str:
        if not context:
            return prompt
        
        context_str = "\n".join([
            f"Previous [{c['category']}]: {c['text']}"
            for c in context.get("relevant_history", [])
        ])
        
        return f"{context_str}\nCurrent: {prompt}"