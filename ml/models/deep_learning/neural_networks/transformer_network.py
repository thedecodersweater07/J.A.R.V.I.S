import torch
import torch.nn as nn
from typing import List, Optional

class TransformerNetwork(nn.Module):
    def __init__(self, 
                 vocab_size: int = 50257,  # GPT-2 vocabulary size
                 d_model: int = 768,
                 nhead: int = 12,
                 num_layers: int = 6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate(self, tokens: List[str]) -> str:
        try:
            with torch.no_grad():
                # Convert tokens to tensor
                x = torch.tensor([tokens]).long()
                
                # Generate through transformer
                output = self.transformer(x, x)
                
                # Get final output
                logits = self.fc_out(output)
                return torch.argmax(logits, dim=-1)
                
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
