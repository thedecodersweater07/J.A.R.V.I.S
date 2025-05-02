"""
Encoder and Decoder components for the Jarvis AI system.
This module implements the encoder and decoder stacks used in the
transformer architecture, including their layers and sublayers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_mechanism import SelfAttention, CrossAttention
from .embedding_layer import PositionalEncoding


class FeedForward(nn.Module):
    """
    Feed Forward Neural Network used in transformer layers.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            d_model: Dimensionality of the model
            d_ff: Dimensionality of the feed-forward network
            dropout: Dropout rate
        """
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Feed-forward output
        """
        residual = x
        
        # Apply feed-forward network
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(x + residual)
        
        return output


class EncoderLayer(nn.Module):
    """
    Single layer of the encoder stack.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize the encoder layer.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward network
            dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attention = SelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the encoder layer.
        
        Args:
            x: Input tensor
            mask: Optional mask tensor
            
        Returns:
            Encoder layer output
        """
        # Apply self-attention
        attn_output = self.self_attention(x, mask)
        
        # Apply feed-forward network
        output = self.feed_forward(attn_output)
        
        return output


class DecoderLayer(nn.Module):
    """
    Single layer of the decoder stack.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize the decoder layer.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward network
            dropout: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attention = SelfAttention(d_model, num_heads, dropout)
        self.cross_attention = CrossAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        Forward pass of the decoder layer.
        
        Args:
            x: Input tensor
            enc_output: Output from encoder
            tgt_mask: Target mask for self-attention
            src_mask: Source mask for cross-attention
            
        Returns:
            Decoder layer output
        """
        # Apply masked self-attention
        self_attn_output = self.self_attention(x, tgt_mask)
        
        # Apply cross-attention
        cross_attn_output = self.cross_attention(self_attn_output, enc_output, src_mask)
        
        # Apply feed-forward network
        output = self.feed_forward(cross_attn_output)
        
        return output


class Encoder(nn.Module):
    """
    Encoder stack of the transformer.
    """
    
    def __init__(self, src_vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_length=5000, dropout=0.1):
        """
        Initialize the encoder.
        
        Args:
            src_vocab_size: Size of the source vocabulary
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Dimensionality of the feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(Encoder, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None, padding_mask=None):
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor
            mask: Optional mask tensor
            padding_mask: Optional padding mask
            
        Returns:
            Encoder output
        """
        # Apply embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Combine masks if both are provided
        if mask is not None and padding_mask is not None:
            mask = mask & padding_mask
        elif padding_mask is not None:
            mask = padding_mask
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        output = self.layer_norm(x)
        
        return output


class Decoder(nn.Module):
    """
    Decoder stack of the transformer.
    """
    
    def __init__(self, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_length=5000, dropout=0.1):
        """
        Initialize the decoder.
        
        Args:
            tgt_vocab_size: Size of the target vocabulary
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Dimensionality of the feed-forward network
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(Decoder, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None, tgt_padding_mask=None):
        """
        Forward pass of the decoder.
        
        Args:
            x: Input tensor
            enc_output: Output from encoder
            tgt_mask: Target mask for self-attention
            src_mask: Source mask for cross-attention
            tgt_padding_mask: Target padding mask
            
        Returns:
            Decoder output
        """
        # Apply embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Combine masks if both are provided
        if tgt_mask is not None and tgt_padding_mask is not None:
            tgt_mask = tgt_mask & tgt_padding_mask
        elif tgt_padding_mask is not None:
            tgt_mask = tgt_padding_mask
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        
        # Apply final layer normalization
        output = self.layer_norm(x)
        
        return output