import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BERTEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = BERTAttention(hidden_size, num_heads, dropout)
        
        # Feed-forward network
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention
        residual = x
        x = self.norm1(x)
        attention_output, attention_weights = self.attention(x, mask)
        x = residual + self.dropout(attention_output)
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.intermediate(x)
        
        return x, attention_weights

class BERTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Linear projections and reshape for attention
        q = self._reshape_for_attention(self.q_linear(x))
        k = self._reshape_for_attention(self.k_linear(x))
        v = self._reshape_for_attention(self.v_linear(x))
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float)
        )
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_size
        )
        
        output = self.out_proj(context)
        return output, attention_weights
        
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
