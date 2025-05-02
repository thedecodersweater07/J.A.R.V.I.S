"""
Architecture module for the LLM component of the Jarvis AI system.
This module provides core transformer architecture components.
"""

from .transformer import Transformer
from .attention_mechanism import (
    ScaledDotProductAttention, 
    MultiHeadAttention, 
    SelfAttention, 
    CrossAttention
)
from .encoder_decoder import (
    FeedForward,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder
)
from .embedding_layer import (
    PositionalEncoding,
    TokenEmbedding,
    InputEmbedding
)

__all__ = [
    'Transformer',
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'SelfAttention',
    'CrossAttention',
    'FeedForward',
    'EncoderLayer',
    'DecoderLayer', 
    'Encoder',
    'Decoder',
    'PositionalEncoding',
    'TokenEmbedding',
    'InputEmbedding'
]