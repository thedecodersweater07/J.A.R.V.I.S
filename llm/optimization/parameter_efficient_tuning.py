"""
Parameter-efficient fine-tuning module implementing various PEFT methods.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PEFTConfig:
    method: str = "lora"  # One of: lora, prefix, adapter, prompt
    rank: int = 8  # For LoRA
    alpha: float = 16  # For LoRA scaling
    dropout: float = 0.1
    num_prefix_tokens: int = 16  # For prefix tuning
    adapter_size: int = 64  # For adapters
    num_prompt_tokens: int = 100  # For prompt tuning
    init_scale: float = 0.01

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""
    
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        self.A = nn.Parameter(torch.randn(in_features, rank) * self.scaling)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A) @ self.B

class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient tuning."""
    
    def __init__(self, input_size: int, adapter_size: int, dropout: float = 0.1):
        super().__init__()
        self.down_project = nn.Linear(input_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        x = self.dropout(x)
        x = x + residual
        return self.layer_norm(x)

class ParameterEfficientTuner:
    """Implements parameter-efficient fine-tuning methods."""
    
    def __init__(self, model: nn.Module, config: PEFTConfig):
        self.model = model
        self.config = config
        self.trainable_params = []
        
    def prepare_model(self):
        """Prepare model for parameter-efficient tuning."""
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        if self.config.method == "lora":
            self._add_lora_layers()
        elif self.config.method == "adapter":
            self._add_adapter_layers()
        elif self.config.method == "prefix":
            self._add_prefix_tuning()
        elif self.config.method == "prompt":
            self._add_prompt_tuning()
            
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.2f}%")

    def _add_lora_layers(self):
        """Add LoRA layers to linear transformations."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                lora = LoRALayer(
                    module.in_features,
                    module.out_features,
                    self.config.rank,
                    self.config.alpha
                )
                self.trainable_params.extend([lora.A, lora.B])
                
                # Wrap original module
                original_forward = module.forward
                def wrapped_forward(x):
                    return original_forward(x) + lora(x)
                module.forward = wrapped_forward

    def _add_adapter_layers(self):
        """Add adapter layers after attention and FFN blocks."""
        for name, module in self.model.named_modules():
            if "output" in name.lower():
                adapter = AdapterLayer(
                    module.dense.out_features,
                    self.config.adapter_size,
                    self.config.dropout
                )
                self.trainable_params.extend(adapter.parameters())
                
                # Insert adapter after the module
                original_forward = module.forward
                def wrapped_forward(x):
                    return adapter(original_forward(x))
                module.forward = wrapped_forward

    def _add_prefix_tuning(self):
        """Add trainable prefix tokens to inputs."""
        prefix_tokens = nn.Parameter(
            torch.randn(1, self.config.num_prefix_tokens, self.model.config.hidden_size)
            * self.config.init_scale
        )
        self.trainable_params.append(prefix_tokens)
        
        # Modify the model's forward pass to prepend prefix tokens
        original_forward = self.model.forward
        def wrapped_forward(input_ids, **kwargs):
            batch_size = input_ids.shape[0]
            prefixes = prefix_tokens.expand(batch_size, -1, -1)
            return original_forward(input_ids, past_key_values=prefixes, **kwargs)
        self.model.forward = wrapped_forward

    def _add_prompt_tuning(self):
        """Add trainable prompt tokens."""
        prompt_embeddings = nn.Parameter(
            torch.randn(1, self.config.num_prompt_tokens, self.model.config.hidden_size)
            * self.config.init_scale
        )
        self.trainable_params.append(prompt_embeddings)
        
        # Modify embeddings lookup to include prompt tokens
        original_forward = self.model.forward
        def wrapped_forward(input_ids, **kwargs):
            batch_size = input_ids.shape[0]
            prompts = prompt_embeddings.expand(batch_size, -1, -1)
            return original_forward(input_ids, prompt_embeddings=prompts, **kwargs)
        self.model.forward = wrapped_forward

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of trainable parameters."""
        return self.trainable_params
