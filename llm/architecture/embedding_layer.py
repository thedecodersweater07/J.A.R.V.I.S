"""
Embedding layer implementation for the Jarvis AI system.
This module provides word embeddings and positional encodings
for the transformer architecture.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the transformer model.
    Uses sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Dimensionality of the model
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Forward pass of the positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with scaling.
    """
    
    def __init__(self, vocab_size, d_model):
        """
        Initialize the token embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimensionality of the model
        """
        super(TokenEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        """
        Forward pass of the token embedding.
        
        Args:
            x: Input tensor of token indices
            
        Returns:
            Embedded tensor scaled by sqrt(d_model)
        """
        # Scale embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class InputEmbedding(nn.Module):
    """
    Complete input embedding layer combining token embedding and positional encoding.
    """
    
    def __init__(self, vocab_size, d_model, max_seq_length=5000, dropout=0.1):
        """
        Initialize the input embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimensionality of the model
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(InputEmbedding, self).__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
    def forward(self, x):
        """
        Forward pass of the input embedding.
        
        Args:
            x: Input tensor of token indices
            
        Returns:
            Fully embedded and positionally encoded tensor
        """
        # Apply token embedding and then positional encoding
        return self.positional_encoding(self.token_embedding(x))