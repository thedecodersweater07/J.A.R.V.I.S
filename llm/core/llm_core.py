import torch
import logging
import os
import gc
from pathlib import Path
import yaml
import spacy
from typing import Optional, Dict, Any, Union, Callable
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

class LLMCore:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM Core with optimized memory management"""
        try:
            # Load and validate config
            self.config = self._load_config(config)
            self._validate_config()
            
            # Extract configs
            model_config = self.config.get("model", {})
            
            # Setup memory management
            self._setup_memory_management()
            
            # Initialize components with lazy loading
            self.model = None
            self.tokenizer = None
            self._model_lock = threading.RLock()
            self._model_initialized = False
            
            # Only initialize immediately if auto_load is True
            if self.config.get("auto_load", True):
                self._initialize_model()
                logger.info("LLMCore initialized successfully")
            else:
                logger.info("LLMCore created with lazy loading enabled")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMCore: {str(e)}", exc_info=True)
            raise
            
    def _setup_memory_management(self):
        """Setup memory management parameters"""
        # Get memory management settings from config
        mem_config = self.config.get("memory_management", {})
        
        # Set memory limits
        self.max_memory_usage = mem_config.get("max_memory_usage", 0.8)  # 80% of available memory
        self.cache_size = mem_config.get("cache_size", 32)  # LRU cache size
        
        # Configure GPU memory if available
        if torch.cuda.is_available():
            self.device = "cuda"
            # Limit GPU memory growth
            for device in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(device)
                logger.info(f"Using GPU {device}: {device_props.name} with {device_props.total_memory / 1e9:.2f} GB memory")
        else:
            self.device = "cpu"
            logger.info("Using CPU for inference")

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
        """Initialize model with better error handling, memory optimization and lazy loading"""
        with self._model_lock:
            if self._model_initialized:
                return
                
            try:
                # Clear any existing model from memory
                if self.model is not None:
                    del self.model
                    if self.tokenizer is not None:
                        del self.tokenizer
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                model_config = self.config.get("model", {})
                model_name = model_config.get("name", "nl_core_news_sm")
                model_type = model_config.get("type", "spacy")
                use_gpu = model_config.get("use_gpu", True) and torch.cuda.is_available()
                load_in_8bit = model_config.get("load_in_8bit", False)
                
                # Get memory optimization settings
                optimize_memory = model_config.get("optimize_memory", True)
                
                if model_type == "spacy":
                    try:
                        # Check if model exists before loading
                        if not spacy.util.is_package(model_name):
                            logger.info(f"Model {model_name} not found, attempting to download...")
                            try:
                                spacy.cli.download(model_name)
                            except Exception as e:
                                logger.warning(f"Failed to download {model_name}: {e}")
                                # Try downloading smaller model as fallback
                                fallback_model = "nl_core_news_sm"
                                logger.info(f"Attempting to download fallback model {fallback_model}")
                                spacy.cli.download(fallback_model)
                                model_name = fallback_model
                        
                        # Load with optimized settings
                        exclude = []
                        if optimize_memory:
                            # Exclude components that aren't needed for basic NLP tasks
                            exclude = ["ner", "textcat", "entity_linker", "entity_ruler"]
                            
                        self.model = spacy.load(model_name, exclude=exclude)
                        logger.info(f"Successfully loaded spaCy model: {self.model.meta['name']}")
                    except Exception as e:
                        logger.error(f"Failed to load spaCy model: {e}")
                        raise
                    
                elif model_type == "transformer":
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    try:
                        # Configure loading options for memory optimization
                        load_options = {}
                        
                        if optimize_memory:
                            if use_gpu:
                                # Use more memory-efficient options
                                if load_in_8bit:
                                    load_options["load_in_8bit"] = True
                                else:
                                    load_options["torch_dtype"] = torch.float16
                                    
                                # Use device map for efficient memory usage
                                load_options["device_map"] = "auto"
                            else:
                                # CPU optimizations
                                load_options["low_cpu_mem_usage"] = True
                        
                        # Load tokenizer first (smaller memory footprint)
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        
                        # Load model with optimized settings
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_options)
                        
                        # Move to appropriate device if not using device_map="auto"
                        if use_gpu and "device_map" not in load_options:
                            self.model = self.model.to(self.device)
                            
                        logger.info(f"Successfully loaded transformer model: {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load transformer {model_name}: {e}")
                        # Fallback to smaller transformer model
                        fallback_model = "gpt2"
                        logger.info(f"Attempting to load fallback model {fallback_model}")
                        
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                            self.model = AutoModelForCausalLM.from_pretrained(fallback_model, low_cpu_mem_usage=True)
                            if use_gpu:
                                self.model = self.model.to(self.device)
                        except Exception as fallback_error:
                            logger.error(f"Failed to load fallback model: {fallback_error}")
                            raise
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                    
                self._model_initialized = True
                    
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise RuntimeError(f"Could not initialize any suitable model: {str(e)}")

    @lru_cache(maxsize=32)
    def _cached_tokenize(self, text: str):
        """Cached tokenization to avoid repeated processing of the same text"""
        if not self._model_initialized:
            self._initialize_model()
            
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer(text)
        elif self.tokenizer is not None:
            return self.tokenizer(text, return_tensors="pt")
        else:
            # For spaCy models
            return self.model.tokenizer(text)
    
    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response with optimized memory usage"""
        # Ensure model is loaded
        if not self._model_initialized:
            self._initialize_model()
        
        # Get relevant context from memory
        context_data = self.memory.get_context(prompt) if hasattr(self, 'memory') and context is None else context or {}
        
        # Prepare input with context
        enhanced_prompt = self._prepare_prompt(prompt, context_data)
        
        # Generate response based on model type
        if hasattr(self.model, "meta") and "spacy" in self.model.meta.get("lang", ""):
            # spaCy model
            doc = self.model(enhanced_prompt)
            response = " ".join([token.text for token in doc])
        else:
            # Transformer model
            try:
                # Use cached tokenization
                inputs = self._cached_tokenize(enhanced_prompt)
                
                # Move inputs to the correct device
                if hasattr(inputs, "to") and self.device == "cuda":
                    inputs = inputs.to(self.device)
                
                # Generate with memory optimization
                with torch.no_grad():  # Disable gradient calculation to save memory
                    outputs = self.model.generate(
                        inputs["input_ids"] if isinstance(inputs, dict) else inputs,
                        max_length=self.config.get("max_length", 100),
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode the generated text
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up memory
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"Error generating response: {str(e)}"
        
        # Store interaction in memory if available
        if hasattr(self, 'memory'):
            self.memory.store_interaction(prompt, response, context_data)
        
        return response

    def _prepare_prompt(self, prompt: str, context: Dict) -> str:
        """Prepare prompt with context in an optimized way"""
        if not context:
            return prompt
        
        # More efficient string building with list comprehension and join
        relevant_history = context.get("relevant_history", [])
        if not relevant_history:
            return prompt
            
        # Build context string more efficiently
        context_parts = [f"Previous [{c['category']}]: {c['text']}" for c in relevant_history if 'category' in c and 'text' in c]
        
        if not context_parts:
            return prompt
            
        context_str = "\n".join(context_parts)
        return f"{context_str}\nCurrent: {prompt}"
        
    def unload_model(self):
        """Explicitly unload model to free memory"""
        with self._model_lock:
            if self.model is not None:
                del self.model
                self.model = None
                
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            self._model_initialized = False
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Model unloaded from memory")
            
    def __del__(self):
        """Destructor to ensure memory is freed"""
        try:
            self.unload_model()
        except:
            pass