import torch
import torch.nn as nn
from typing import Tuple
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout=dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.attention(x)
        x = residual + self.dropout(attn_out)
        
        # Feed forward
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        
        return x, attn_weights
