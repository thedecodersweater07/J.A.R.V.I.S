from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from .config import ConfigManager
from .templates import PromptTemplate
from .processor import ResponseProcessor
from data.core.manager import DataManager

logger = logging.getLogger(__name__)

class LLMPipeline:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.active_model = "default"
        self.max_length = 1024
        self.model_path = Path(__file__).parent / "models"
        self.config = ConfigManager()
        self.templates = PromptTemplate()
        self.processor = ResponseProcessor()
        self.data_manager = DataManager()
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        try:
            # Load default model and tokenizer
            model_name = "gpt2"  # Can be configured
            self.tokenizers["default"] = AutoTokenizer.from_pretrained(model_name)
            self.models["default"] = AutoModelForCausalLM.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self.models["default"] = self.models["default"].to("cuda")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM pipeline: {e}")
            
    async def process(self, user_input: str, context: str = "", template: str = "chat"):
        # Format prompt
        prompt = self.templates.format_prompt(
            template,
            user_input=user_input,
            context=context
        )
        
        # Get configuration
        config = self.config.load_config()
        
        # Process response
        response = await self._get_llm_response(prompt, config)
        
        # Store interaction data
        interaction_data = {
            'input': user_input,
            'context': context,
            'response': response,
            'template': template,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save to CSV for analysis
        self.data_manager.save_csv(
            [interaction_data], 
            f"llm_interactions_{pd.Timestamp.now().strftime('%Y%m')}"
        )
        
        return self.processor.process_response(response)
    
    async def _get_llm_response(self, prompt: str, config):
        # Implement actual LLM call here
        return "AI Response"
    
    def add_model(self, name: str, model_path: str) -> bool:
        try:
            self.tokenizers[name] = AutoTokenizer.from_pretrained(model_path)
            self.models[name] = AutoModelForCausalLM.from_pretrained(model_path)
            if torch.cuda.is_available():
                self.models[name] = self.models[name].to("cuda")
            return True
        except Exception as e:
            logger.error(f"Failed to add model {name}: {e}")
            return False
            
    def switch_model(self, name: str) -> bool:
        if name in self.models:
            self.active_model = name
            return True
        return False
    
    def save_training_data(self, data: List[Dict]):
        """Save new training data"""
        self.data_manager.save_json(
            {'training_examples': data},
            f"training_data_{pd.Timestamp.now().strftime('%Y%m%d')}"
        )
