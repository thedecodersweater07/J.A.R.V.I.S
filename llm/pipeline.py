from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

logger = logging.getLogger(__name__)

class LLMPipeline:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.active_model = "default"
        self.max_length = 1024
        self.model_path = Path(__file__).parent / "models"
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
            
    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not self.models or self.active_model not in self.models:
            return "System is initializing..."
            
        try:
            tokenizer = self.tokenizers[self.active_model]
            model = self.models[self.active_model]
            
            inputs = tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            outputs = model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return "Sorry, I encountered an error processing your request."
            
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
