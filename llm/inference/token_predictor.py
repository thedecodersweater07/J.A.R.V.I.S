import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TokenPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.using_fallback = False
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info("Initialized TokenPredictor with transformers")
            
        except ImportError:
            logger.warning("Transformers not installed, using fallback mode. Run: pip install transformers torch")
            self.using_fallback = True

    def predict(self, text: str, context: Dict) -> List[str]:
        if self.using_fallback:
            # Simple fallback using basic text manipulation
            words = text.split()
            return [" ".join(words[:5])]  # Simple 5-word response
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=100)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error in token prediction: {e}")
            return ["I am processing that..."]
