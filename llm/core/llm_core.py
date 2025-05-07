from typing import Optional, Dict, Any
import torch
import logging
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm.memory.enhanced_memory import EnhancedMemoryManager
from llm.optimization.llm_optimizer import LLMOptimizer, LLMOptimizationConfig
from db.sql.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class LLMCore:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM Core"""
        try:
            # Load and validate config
            self.config = config or {}
            self._validate_config()
            
            # Extract configs
            model_config = self.config.get("model", {})
            optimization_config = model_config.get("optimization", {})
            
            # Initialize components
            self.db = DatabaseManager(config=self.config.get("database", {}))
            self.memory = EnhancedMemoryManager(self.config.get("memory", {}))
            self.optimizer = LLMOptimizer(config=optimization_config)
            
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
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "defaults" / "llm.json"
        
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            return {}

    def _initialize_model(self) -> None:
        """Initialize and optimize the model"""
        try:
            model_name = self.config.get("model", {}).get("name", "gpt2")
            logger.info(f"Loading model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Apply optimizations
            self.model = self.optimizer.optimize_inference(self.model)
            logger.debug("Model optimization completed")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
            raise

    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        # Get relevant context from memory
        context_data = self.memory.get_context(prompt) if context is None else context
        
        # Prepare input with context
        enhanced_prompt = self._prepare_prompt(prompt, context_data)
        
        # Generate response
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"].to(self.model.device),
            max_length=self.config["llm"]["model"]["max_length"],
            temperature=self.config["llm"]["model"]["temperature"],
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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