import torch
import torch.nn as nn
from typing import Dict, Any
from ..base import BaseModel
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        norm_x = self.layer_norm1(x)
        attn_output, attn_weights = self.attention(norm_x)
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x, attn_weights

class GPTModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model dimensions
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.vocab_size = config.get("vocab_size", 50257)
        
        # Core components
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(1024, self.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.output_head = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs["input_ids"]
        
        # Get positions
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        
        # Embeddings
        token_embeds = self.token_embedding(x)
        pos_embeds = self.position_embedding(positions)
        hidden_states = token_embeds + pos_embeds
        
        # Transformer layers
        attention_weights = []
        for layer in self.layers:
            hidden_states, attn = layer(hidden_states)
            attention_weights.append(attn)
            
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "attention_weights": attention_weights
        }
