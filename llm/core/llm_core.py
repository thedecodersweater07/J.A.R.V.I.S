import torch
import logging
import gc
from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from functools import lru_cache
import threading
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class LLMCore:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)  # Changed to use standard logging
        
        # Then continue with rest of initialization
        if hasattr(self, '_initialized'):
            return
            
        try:
            # Define fallback models first
            self.fallback_models = [
                "distilgpt2",  # Smaller model first
                "gpt2",
                "bert-base-uncased"
            ]

            # Load and validate config before initialization
            self.config = self._load_config(config)
            self._validate_config()
            
            # Setup memory management first
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._setup_memory_management()
            
            # Initialize components with lazy loading
            self.model = None
            self.tokenizer = None
            self._model_lock = threading.RLock()
            self._model_initialized = False
            
            # Initialize memory manager
            from llm.memory.enhanced_memory import EnhancedMemoryManager
            self.memory = EnhancedMemoryManager({
                "cache_size": self.config.get("memory_management", {}).get("cache_size", 1000),
                "priority_threshold": 0.5,
                "context_window": 5
            })
            
            # Initialize model if resources available
            if self.config.get("auto_load", True) and self._check_system_resources():
                self._initialize_model()
            else:
                logger.info("LLMCore created with lazy loading enabled")
                
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLMCore: {str(e)}", exc_info=True)
            raise

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration with better validation"""
        default_config = {
            "model": {
                "name": "distilgpt2",
                "type": "transformer",
                "max_length": 2048,  # Increased context length
                "temperature": 0.7,
                "low_cpu_mem_usage": True,
                "offload_folder": "models/offload",  # Add disk offloading
                "load_in_8bit": True,  # Enable 8-bit quantization
                "device_map": "auto"  # Enable auto device mapping
            },
            "memory_management": {
                "cache_size": 2000,  # Increased cache size
                "max_memory_usage": 0.9,  # More aggressive memory usage
                "enable_gc": True,
                "cpu_threshold": 85.0,  # CPU usage threshold
                "memory_threshold": 85.0,  # Memory usage threshold
                "gpu_threshold": 85.0,  # GPU usage threshold
                "throttle_interval": 0.1  # Seconds between throttle checks
            },
            "nlp": {
                "default_model": "en_core_web_sm",
                "fallback_models": ["en_core_web_sm", "en_core_web_md"]
            }
        }

        # Enhanced config merging
        if isinstance(config, dict):
            merged_config = default_config.copy()
            for key, value in config.items():
                if isinstance(value, dict) and key in merged_config:
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            return merged_config

        # Try multiple config paths
        config_paths = [
            Path("config/llm.yaml"),
            Path("config/llm.yml"),
            Path(f"{Path.home()}/.jarvis/config/llm.yaml")
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        loaded_config = yaml.safe_load(f)
                        if not loaded_config:
                            continue
                        merged_config = default_config.copy()
                        for key, value in loaded_config.items():
                            if isinstance(value, dict) and key in merged_config:
                                merged_config[key].update(value)
                            else:
                                merged_config[key] = value
                        return merged_config
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
                    continue

        return default_config

    def _check_system_resources(self) -> bool:
        """Enhanced system resource checking"""
        try:
            # Get thresholds from config
            mem_config = self.config.get("memory_management", {})
            cpu_thresh = mem_config.get("cpu_threshold", 85.0)
            mem_thresh = mem_config.get("memory_threshold", 85.0)
            gpu_thresh = mem_config.get("gpu_threshold", 85.0)

            # Check CPU usage (non-blocking)
            cpu_percent = psutil.cpu_percent(interval=0)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Enhanced GPU memory checking
            gpu_ok = True
            if self.device == "cuda":
                for i in range(torch.cuda.device_count()):
                    gpu_mem = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100
                    if gpu_mem > gpu_thresh:
                        logger.warning(f"GPU {i} memory usage ({gpu_mem:.1f}%) exceeds threshold ({gpu_thresh}%)")
                        gpu_ok = False

            # Log resource usage
            logger.info(f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, "
                       f"Swap: {swap.percent:.1f}%")

            # Return True only if all resources are below thresholds
            resources_ok = (
                cpu_percent < cpu_thresh and
                memory.percent < mem_thresh and
                swap.percent < 90 and
                gpu_ok
            )

            if not resources_ok:
                logger.warning("System resources exceeded thresholds")
                self._cleanup_existing_model()  # Attempt to free resources

            return resources_ok

        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return False

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

    def _initialize_model(self) -> None:
        """Initialize model with better error handling and fallbacks"""
        import importlib
        with self._model_lock:
            if self._model_initialized:
                return

            transformers_kwargs = {
                'torch_dtype': 'auto',
                'device_map': 'auto' if torch.cuda.is_available() else None,
                'low_cpu_mem_usage': True,
            }

            errors = []
            for model_name in [self.config["model"].get("name", "gpt2")] + self.fallback_models:
                try:
                    self.logger.info(f"Attempting to load model: {model_name}")
                    self._cleanup_existing_model()

                    # Detect spaCy models (by name pattern or explicit list)
                    if model_name.startswith("nl_core_news_") or model_name.startswith("en_core_web_"):
                        import spacy
                        self.model = spacy.load(model_name)
                        self.tokenizer = self.model.tokenizer
                        self._model_initialized = True
                        self.logger.info(f"Successfully loaded spaCy model: {model_name}")
                        return
                    else:
                        # Try HuggingFace transformers
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **transformers_kwargs)
                        if torch.cuda.is_available():
                            self.model = self.model.to("cuda")
                        self._model_initialized = True
                        self.logger.info(f"Successfully loaded transformer model: {model_name}")
                        return

                except Exception as e:
                    errors.append(f"Failed to load {model_name}: {str(e)}")
                    continue

            error_details = "\n".join(errors)
            raise RuntimeError(f"Failed to initialize any model. Errors:\n{error_details}")

    def _cleanup_existing_model(self):
        """Clean up existing model and free memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @lru_cache(maxsize=32)
    def _cached_tokenize(self, text: str):
        """Cached tokenization to avoid repeated processing of the same text"""
        from transformers import PreTrainedTokenizer
        if not self._model_initialized:
            self._initialize_model()

        # HuggingFace transformer
        if self.tokenizer is not None and isinstance(self.tokenizer, PreTrainedTokenizer):
            return self.tokenizer(text, return_tensors="pt")
        # spaCy model
        elif self.model is not None and hasattr(self.model, 'tokenizer'):
            return self.model(text)
        else:
            raise RuntimeError("Model or tokenizer not initialized correctly.")

    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response with optimized memory usage"""
        from transformers import PreTrainedModel, PreTrainedTokenizer
        import spacy.language
        import torch
        # Ensure model is loaded
        if not self._model_initialized:
            self._initialize_model()

        if self.model is None:
            return "Error: Model is not initialized."

        # Get relevant context from memory
        context_data = self.memory.get_context(prompt) if hasattr(self, 'memory') and context is None else context or {}
        enhanced_prompt = self._prepare_prompt(prompt, context_data)

        try:
            # HuggingFace transformer: PreTrainedModel and PreTrainedTokenizer
            if (
                self.tokenizer is not None and
                isinstance(self.model, PreTrainedModel) and
                isinstance(self.tokenizer, PreTrainedTokenizer)
            ):
                inputs = self._cached_tokenize(enhanced_prompt)
                if isinstance(inputs, dict) and self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if isinstance(inputs, dict) and "input_ids" in inputs:
                    input_ids = inputs["input_ids"]
                else:
                    raise RuntimeError("Invalid input for HuggingFace model.")
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=self.config.get("max_length", 100),
                        num_return_sequences=1,
                        pad_token_id=getattr(self.tokenizer, 'eos_token_id', None)
                    )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # spaCy model: Language object
            elif isinstance(self.model, spacy.language.Language):
                doc = self.model(enhanced_prompt)
                response = doc.text if hasattr(doc, 'text') else str(doc)
            else:
                response = "Error: Model type not supported."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = f"Error generating response: {str(e)}"

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