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

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration from dictionary or file"""
        default_config = {
            "model": {
                "name": "nl_core_news_lg",
                "type": "spacy"
            }
        }
        
        if isinstance(config, dict):
            return {**default_config, **config}
            
        config_path = Path("config/llm.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    loaded_config = yaml.safe_load(f)
                    return {**default_config, **loaded_config}
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
                
        return default_config

    def _initialize_model(self) -> None:
        """Initialize model with better error handling and transformer support"""
        try:
            model_config = self.config.get("model", {})
            model_name = model_config.get("name", "nl_core_news_sm")
            model_type = model_config.get("type", "spacy")

            if model_type == "spacy":
                try:
                    self.model = spacy.load(model_name)
                except OSError:
                    logger.info(f"Model {model_name} not found, attempting to download...")
                    try:
                        spacy.cli.download(model_name)
                        self.model = spacy.load(model_name)
                    except Exception as e:
                        logger.warning(f"Failed to download {model_name}: {e}")
                        # Try downloading smaller model as fallback
                        fallback_model = "nl_core_news_sm"
                        logger.info(f"Attempting to download fallback model {fallback_model}")
                        spacy.cli.download(fallback_model)
                        self.model = spacy.load(fallback_model)
                        
                logger.info(f"Successfully loaded spaCy model: {self.model.meta['name']}")
                
            elif model_type == "transformer":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    logger.info(f"Successfully loaded transformer model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load transformer {model_name}: {e}")
                    # Fallback to smaller transformer model
                    fallback_model = "gpt2"
                    logger.info(f"Attempting to load fallback model {fallback_model}")
                    self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Could not initialize any suitable model: {str(e)}")

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