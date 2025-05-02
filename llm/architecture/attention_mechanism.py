"""
Attention mechanism implementation for the Jarvis AI system.
This module provides various attention mechanisms used in the transformer
architecture, including self-attention and multi-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    """
    
    def __init__(self, dropout=0.1):
        """
        Initialize the attention mechanism.
        
        Args:
            dropout: Dropout rate
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the attention mechanism.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask tensor
            
        Returns:
            Attention output and attention weights
        """
        # Get dimensions
        d_k = query.size(-1)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Calculate final output
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize the multi-head attention mechanism.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the multi-head attention mechanism.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask tensor
            
        Returns:
            Multi-head attention output
        """
        batch_size = query.size(0)
        residual = query
        
        # Linear projections and reshape
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Adjust mask for multi-head attention
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # Apply attention
        attn_output, _ = self.attention(q, k, v, mask)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(output + residual)
        
        return output


class SelfAttention(nn.Module):
    """
    Self-Attention layer.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize the self-attention layer.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SelfAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the self-attention layer.
        
        Args:
            x: Input tensor
            mask: Optional mask tensor
            
        Returns:
            Self-attention output
        """
        return self.multi_head_attention(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-Attention layer for encoder-decoder attention.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize the cross-attention layer.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, x, enc_output, mask=None):
        """
        Forward pass of the cross-attention layer.
        
        Args:
            x: Input tensor from decoder
            enc_output: Output from encoder
            mask: Optional mask tensor
            
        Returns:
            Cross-attention output
        """
        return self.multi_head_attention(x, enc_output, enc_output, mask)