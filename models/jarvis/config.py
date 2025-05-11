from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class JarvisConfig:
    """Base configuration for JARVIS models"""
    name: str
    version: str = "1.0.0"
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_cache: bool = True

    # Add missing attributes
    num_classes: int = 2  # Default for binary classification
    embedding_size: int = 768  # Same as hidden_size by default
    max_seq_length: int = 512
    type_vocab_size: int = 2
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Task-specific configurations
    classification_classes: Dict[str, int] = field(default_factory=lambda: {
        "intent": 10,
        "sentiment": 3,
        "topic": 5
    })

JARVIS_CONFIGS = {
    "jarvis-small": JarvisConfig(
        name="jarvis-small",
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536
    ),
    "jarvis-base": JarvisConfig(
        name="jarvis-base",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_classes=10  # Set reasonable default
    ),
    "jarvis-large": JarvisConfig(
        name="jarvis-large",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096
    ),
    "jarvis-xl": JarvisConfig(
        name="jarvis-xl",
        hidden_size=1536,
        num_hidden_layers=32,
        num_attention_heads=24,
        intermediate_size=6144
    )
}
