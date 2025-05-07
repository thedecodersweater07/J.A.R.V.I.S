import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = hidden_size * 4
            
        self.fc1 = nn.Linear(hidden_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First linear layer with activation
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        
        # Second linear layer
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x
