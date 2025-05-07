import json
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from datetime import datetime

class DutchDataGenerator:
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_conversation_data(self, num_samples: int = 1000) -> None:
        """Genereer Nederlandse conversatie data"""
        conversations = []
        templates = {
            "greeting": ["Hallo", "Goedemorgen", "Hoi", "Goedemiddag"],
            "question": ["Hoe gaat het?", "Wat doe je?", "Waar ben je?"],
            "response": ["Goed, dank je", "Prima, en met jou?", "Het gaat goed"]
        }

        for _ in range(num_samples):
            conv = {
                "input": np.random.choice(templates["greeting"]) + " " + 
                        np.random.choice(templates["question"]),
                "output": np.random.choice(templates["response"]),
                "type": "conversation",
                "timestamp": datetime.now().isoformat()
            }
            conversations.append(conv)

        self._save_json(conversations, "conversations.json")

    def _save_json(self, data: List[Dict], filename: str) -> None:
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generator = DutchDataGenerator("data/dutch")
    generator.generate_conversation_data()
