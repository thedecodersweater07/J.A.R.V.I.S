"""
Model pruning module for removing redundant weights and connections.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class PruningConfig:
    def __init__(
        self,
        method: str = "l1_unstructured",
        sparsity: float = 0.3,
        schedule: str = "gradual",
        pruning_steps: int = 10,
        target_sparsity: float = 0.7,
        **kwargs
    ):
        self.method = method
        self.sparsity = sparsity
        self.schedule = schedule
        self.pruning_steps = pruning_steps
        self.target_sparsity = target_sparsity
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class ModelPruner:
    """Handles model pruning for reducing model size while maintaining performance."""
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        self.pruning_stats = {}
        self.original_state_dict = None
        
    def save_original_state(self):
        """Save original model state before pruning."""
        self.original_state_dict = {
            name: param.clone().detach() 
            for name, param in self.model.named_parameters()
        }

    def apply_pruning(self, layer_specs: Optional[List[Dict]] = None):
        """Apply pruning to specified layers or globally."""
        if layer_specs is None:
            # Global pruning
            parameters_to_prune = [
                (module, "weight")
                for name, module in self.model.named_modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))
            ]
            
            if self.config.method == "l1_unstructured":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.config.sparsity
                )
        else:
            # Layer-specific pruning
            for spec in layer_specs:
                module = dict(self.model.named_modules())[spec["layer_name"]]
                self._prune_layer(module, spec)
                
        self._update_pruning_stats()
        logger.info(f"Pruning applied with {self.config.method} method")

    def _prune_layer(self, layer: nn.Module, spec: Dict):
        """Apply pruning to a specific layer."""
        sparsity = spec.get("sparsity", self.config.sparsity)
        
        if self.config.method == "l1_unstructured":
            prune.l1_unstructured(layer, "weight", amount=sparsity)
        elif self.config.method == "structured":
            prune.ln_structured(layer, "weight", amount=sparsity, n=2, dim=0)

    def gradual_pruning(self, pruning_schedule):
        """Implement gradual pruning according to schedule."""
        initial_sparsity = self.config.sparsity
        final_sparsity = self.config.target_sparsity
        
        for step, sparsity in enumerate(pruning_schedule):
            self.config.sparsity = sparsity
            self.apply_pruning()
            logger.info(f"Gradual pruning step {step+1}: sparsity = {sparsity:.3f}")

    def _update_pruning_stats(self):
        """Update pruning statistics."""
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if "weight" in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        self.pruning_stats.update({
            "sparsity": zero_params / total_params,
            "remaining_params": total_params - zero_params,
            "compression_ratio": total_params / (total_params - zero_params)
        })

    def remove_pruning(self):
        """Remove pruning and make sparsity permanent."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, "weight")
        logger.info("Pruning removed and sparsity made permanent")

    def restore_original(self):
        """Restore model to pre-pruning state."""
        if self.original_state_dict is not None:
            for name, param in self.model.named_parameters():
                param.data = self.original_state_dict[name].clone()
            logger.info("Model restored to original state")
