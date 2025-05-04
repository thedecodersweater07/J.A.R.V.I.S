import torch
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class TokenPredictor:
    """Handles token prediction for the inference engine"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    async def predict_next(self, input_text: str, num_tokens: int = 1) -> str:
        """Predict the next tokens given an input text"""
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + num_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return predicted_text[len(input_text):]
            
        except Exception as e:
            logger.error(f"Token prediction failed: {e}")
            return ""
