import torch
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from security.secure_data_handler import SecureDataHandler

class MLDataManager:
    def __init__(self):
        self.data_root = Path("data/ai_training_data")
        self.secure_handler = SecureDataHandler()
        
    def load_training_data(self, dataset_name: str) -> pd.DataFrame:
        """Load training data securely"""
        data_path = self.data_root / f"{dataset_name}.csv"
        data = pd.read_csv(data_path)
        # Strip sensitive info
        return self.secure_handler.sanitize_data(data)
        
    def get_dutch_corpus(self) -> List[str]:
        """Load Dutch language corpus"""
        with open(self.data_root / "dutch_words.txt") as f:
            return [line.strip() for line in f if line.strip()]
