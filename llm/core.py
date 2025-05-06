from sqlalchemy.orm import Session
from typing import Optional, List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..db.models import Conversation

class LLMCore:
    def __init__(self, model_name: str = "gpt2", db_session: Optional[Session] = None):
        self.model_name = model_name
        self.db_session = db_session
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate_response(self, input_text: str) -> str:
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if self.db_session:
            conversation = Conversation(
                user_input=input_text,
                response=response
            )
            self.db_session.add(conversation)
            self.db_session.commit()
            
        return response
