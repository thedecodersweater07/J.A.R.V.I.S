import pandas as pd
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader

class MLDataIntegrator:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.datasets = {}
        self.loaders = {}
        
    def load_training_data(self) -> Dict[str, DataLoader]:
        """Load all training data sources"""
        # Load sandbox data
        sandbox_data = pd.read_csv(self.data_root / "sandbox/interactions.csv")
        
        # Load real-world data
        real_data = pd.read_csv(self.data_root / "raw/llm_training_nl.csv")
        
        # Combine and process
        combined_data = self._process_and_combine_data([sandbox_data, real_data])
        
        # Create data loaders
        self.loaders = {
            "train": DataLoader(
                JARVISDataset(combined_data["train"]),
                batch_size=32,
                shuffle=True
            ),
            "val": DataLoader(
                JARVISDataset(combined_data["val"]),
                batch_size=32
            )
        }
        
        return self.loaders

class JARVISDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            "input": item["input"],
            "output": item["output"],
            "context": item["context"]
        }
