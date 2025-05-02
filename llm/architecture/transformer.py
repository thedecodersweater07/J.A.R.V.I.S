"""
Transformer model implementation for the Jarvis AI system.
This module provides the core transformer architecture that powers
the language learning capabilities of the system.
"""

import torch
import torch.nn as nn
from .attention_mechanism import MultiHeadAttention
from .encoder_decoder import Encoder, Decoder


class Transformer(nn.Module):
    """
    Complete transformer architecture with encoder and decoder stacks.
    """
    
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512, 
                 num_heads=8, 
                 num_layers=6, 
                 d_ff=2048, 
                 max_seq_length=5000, 
                 dropout=0.1):
        """
        Initialize the transformer model.
        
        Args:
            src_vocab_size: Size of the source vocabulary
            tgt_vocab_size: Size of the target vocabulary
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            num_layers: Number of encoder and decoder layers
            d_ff: Dimensionality of the feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass of the transformer.
        
        Args:
            src: Source sequence
            tgt: Target sequence
            src_mask: Source mask for attention
            tgt_mask: Target mask for attention
            src_padding_mask: Source padding mask
            tgt_padding_mask: Target padding mask
            
        Returns:
            Output logits
        """
        # Pass through encoder and decoder
        enc_output = self.encoder(src, src_mask, src_padding_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_padding_mask, tgt_padding_mask)
        
        # Final linear layer
        output = self.final_layer(dec_output)
        
        return output
    
    def encode(self, src, src_mask):
        """
        Encode the source sequence.
        
        Args:
            src: Source sequence
            src_mask: Source mask
            
        Returns:
            Encoder output
        """
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, memory, tgt_mask):
        """
        Decode with the given memory.
        
        Args:
            tgt: Target sequence
            memory: Encoder output
            tgt_mask: Target mask
            
        Returns:
            Decoder output
        """
        return self.decoder(tgt, memory, tgt_mask)