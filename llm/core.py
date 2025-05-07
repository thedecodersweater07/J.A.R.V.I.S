from sqlalchemy.orm import Session
from typing import Optional, List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from pathlib import Path
import yaml
from ..db.models import Conversation

class LLMCore:
    def __init__(self, model_name: str = "gpt2", db_session: Optional[Session] = None):
        self.model_name = model_name
        self.db_session = db_session
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load config
        config_path = Path(__file__).parent.parent / "data" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Load training data
        self.training_data = pd.read_csv(
            Path(self.config["data_paths"]["llm"]["training"]),
            encoding=self.config["file_formats"]["csv_encoding"]
        )
        
        # Load feedback data
        self.feedback_data = pd.read_csv(
            Path(self.config["data_paths"]["llm"]["feedback"]),
            encoding=self.config["file_formats"]["csv_encoding"]
        )
        
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

    def load_training_examples(self) -> List[Dict[str, str]]:
        """Load training examples from CSV"""
        return self.training_data.to_dict('records')

    def save_feedback(self, response_id: int, rating: int, feedback_text: str):
        """Save feedback to CSV"""
        new_feedback = pd.DataFrame([{
            'response_id': response_id,
            'rating': rating,
            'feedback_text': feedback_text,
            'timestamp': pd.Timestamp.now()
        }])
        new_feedback.to_csv(
            Path(self.config["data_paths"]["llm"]["feedback"]),
            mode='a',
            header=False,
            index=False
        )
