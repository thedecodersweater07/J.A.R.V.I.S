import torch
import logging
import gc
from pathlib import Path
import yaml
from typing import Optional, Dict, Any, Union, List, Tuple, cast
from functools import lru_cache
import threading
import psutil

# Import required transformers components
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.generation.utils import GenerationMixin

logger = logging.getLogger(__name__)

class LLMCore:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *_, **__):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
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

                    # Only allow HuggingFace/transformer models
                    if model_name.startswith("gpt2") or model_name.startswith("distilgpt2"):
                        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
                        from transformers import AutoTokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = GPT2LMHeadModel.from_pretrained(model_name, **transformers_kwargs)
                        if torch.cuda.is_available():
                            self.model = self.model.to("cuda")  # type: ignore
                        self._model_initialized = True
                        self.logger.info(f"Successfully loaded GPT2 model: {model_name}")
                        return
                    elif model_name.startswith("bert"):
                        from transformers.models.bert.modeling_bert import BertForMaskedLM
                        from transformers import AutoTokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = BertForMaskedLM.from_pretrained(model_name, **transformers_kwargs)
                        if torch.cuda.is_available():
                            self.model = self.model.to("cuda")  # type: ignore
                        self._model_initialized = True
                        self.logger.info(f"Successfully loaded BERT model: {model_name}")
                        return
                    else:
                        # Try HuggingFace transformers generic fallback
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **transformers_kwargs)
                        if torch.cuda.is_available():
                            self.model = self.model.to("cuda")  # type: ignore
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
        from transformers.tokenization_utils import PreTrainedTokenizer
        
        if not self._model_initialized:
            self._initialize_model()
            
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
            
        try:
            # Ensure text is a non-empty string
            if not text or not isinstance(text, str):
                text = ""
                
            # Tokenize with attention mask and return as dictionary
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_length", 512),
                return_attention_mask=True
            )
            
            # Ensure we have input_ids in the output
            if "input_ids" not in inputs:
                raise ValueError("Tokenizer did not return input_ids")
                
            # Move tensors to device if needed
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            return inputs
            
        except Exception as e:
            self.logger.error(f"Error in _cached_tokenize: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to tokenize input: {str(e)}")

    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response with optimized memory usage and improved type safety"""
        from transformers.modeling_utils import PreTrainedModel
        from transformers.tokenization_utils import PreTrainedTokenizer
        import torch
        
        try:
            # Ensure model is loaded
            if not self._model_initialized:
                self._initialize_model()
                
            if self.model is None:
                self.logger.error("Model is not initialized")
                return "Error: Model is not initialized."
                
            if self.tokenizer is None:
                self.logger.error("Tokenizer is not initialized")
                return "Error: Tokenizer is not initialized."

            # Get relevant context from memory
            context_data = context or {}
            if hasattr(self, 'memory') and context is None:
                try:
                    context_data = self.memory.get_context(prompt) or {}
                except Exception as e:
                    self.logger.warning(f"Failed to get context from memory: {e}")
                    context_data = {}
                    
            enhanced_prompt = self._prepare_prompt(prompt, context_data)
            self.logger.debug(f"Enhanced prompt: {enhanced_prompt}")

            # Ensure we have a valid prompt
            if not enhanced_prompt or not isinstance(enhanced_prompt, str):
                enhanced_prompt = str(enhanced_prompt or "")

            # Tokenize the input
            try:
                inputs = self._cached_tokenize(enhanced_prompt)
                if not isinstance(inputs, dict) or "input_ids" not in inputs:
                    raise ValueError("Invalid tokenizer output format")
                    
                # Move inputs to the correct device
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
            except Exception as e:
                self.logger.error(f"Tokenization failed: {e}", exc_info=True)
                return f"Error: Failed to tokenize input - {str(e)}"

            # Generate response
            try:
                with torch.no_grad():
                    # Get generation parameters from config or use defaults
                    generation_config = {
                        'max_length': min(self.config.get("max_length", 100), 512),  # Cap at 512 tokens
                        'num_return_sequences': 1,
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'do_sample': True,
                        'pad_token_id': getattr(self.tokenizer, 'eos_token_id', None) or 50256,  # Default to GPT2's eos
                        'attention_mask': inputs.get('attention_mask')
                    }
                    
                    # Generate the response - ensure input_ids is a tensor on the correct device
                    input_tensor = inputs['input_ids']
                    if not isinstance(input_tensor, torch.Tensor):
                        input_tensor = torch.tensor(input_tensor, device=self.device)
                    elif input_tensor.device != self.device:
                        input_tensor = input_tensor.to(self.device)
                    
                    # Generate sequences - ensure model is a GenerationMixin
                    if not hasattr(self.model, 'generate') or not callable(self.model.generate):
                        raise RuntimeError("Model does not support text generation")
                        
                    # Cast to GenerationMixin to make type checker happy
                    generation_model = cast(GenerationMixin, self.model)
                    output_sequences = generation_model.generate(
                        input_ids=input_tensor,
                        **{k: v for k, v in generation_config.items() if v is not None}
                    )
                    
                    # Decode the generated text
                    response = self.tokenizer.decode(
                        output_sequences[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Remove any duplicate newlines and strip whitespace
                    response = ' '.join(response.split())
                    
                    self.logger.debug(f"Generated response: {response[:200]}...")
                    return response
                    
            except Exception as e:
                self.logger.error(f"Generation failed: {e}", exc_info=True)
                return f"Error: Failed to generate response - {str(e)}"
                
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_response: {e}", exc_info=True)
            return f"Error: An unexpected error occurred - {str(e)}"

        if hasattr(self, 'memory'):
            self.memory.store_interaction(prompt, response, context_data)
        return response

    def _decode_model_output(self, outputs: Union[torch.Tensor, List, Tuple]) -> str:
        """Safely decode model outputs with proper type checking"""
        if self.tokenizer is None:
            return "Error: Tokenizer not available for decoding."
        
        try:
            # Handle torch.Tensor outputs
            if isinstance(outputs, torch.Tensor):
                # Check if it's a batch of sequences
                if outputs.dim() >= 2 and outputs.size(0) > 0:
                    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    return self.tokenizer.decode(outputs, skip_special_tokens=True)
            
            # Handle list/tuple outputs
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                # Get the first item if it's a batch
                first_output = outputs[0]
                if isinstance(first_output, torch.Tensor):
                    return self.tokenizer.decode(first_output, skip_special_tokens=True)
                else:
                    return self.tokenizer.decode(outputs, skip_special_tokens=True)
            
            # Fallback for other types
            else:
                return self.tokenizer.decode(outputs, skip_special_tokens=True)
                
        except Exception as e:
            logger.error(f"Error decoding model output: {e}")
            return f"Error decoding response: {str(e)}"

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

    def generate(self, prompt: str, context: Optional[dict] = None, **kwargs) -> dict:
        """
        Unified generate method for compatibility with JarvisModel and UI.
        Returns a dict with a 'text' key for the generated response.
        """
        try:
            text = self.generate_response(prompt, context)
            return {"text": text}
        except Exception as e:
            logger.error(f"[LLMCore.generate] Error in AI: {e}", exc_info=True)
            return {"text": f"[Error in AI: {e}]"}