import torch
import torch.nn as nn
from typing import Dict, Any
from ..base import BaseModel
from .encoder import BERTEncoderLayer

class BERTPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERTModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model dimensions
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.vocab_size = config.get("vocab_size", 30522)
        
        # Core components
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(512, self.hidden_size)
        self.token_type_embedding = nn.Embedding(2, self.hidden_size)
        
        # Transformer layers
        self.encoder = nn.ModuleList([
            BERTEncoderLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=config.get("intermediate_size", 3072),
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_layers)
        ])
        
        self.pooler = BERTPooler(self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
