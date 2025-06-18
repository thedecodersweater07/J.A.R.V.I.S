"""
Utility functions for JARVIS models.

This module provides helper functions for model operations such as saving, loading,
and managing model checkpoints.
"""

import os
import json
import logging
import sys
from typing import Dict, Any, Optional, Union, TypeVar, Type, Tuple
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Try to import PyTorch, create dummy classes if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Running in limited functionality mode.")
    TORCH_AVAILABLE = False
    
    # Create dummy classes for type checking
    class DummyModule:
        def __init__(self, *args, **kwargs):
            self.state_dict = lambda: {}
            self.load_state_dict = lambda *args, **kwargs: None
            self.parameters = lambda recurse=True: []
            self.to = lambda *args, **kwargs: self
            self.train = lambda mode=True: self
            self.eval = lambda: self
            
    class DummyOptimizer:
        def __init__(self, *args, **kwargs):
            self.state_dict = lambda: {}
            self.load_state_dict = lambda *args, **kwargs: None
            
    # Create dummy modules
    torch = type('torch', (), {
        'Tensor': type('DummyTensor', (), {'to': lambda self, *args, **kwargs: self}),
        'device': type('device', (), {'__call__': lambda *args, **kwargs: 'cpu'}),
        'cuda': type('cuda', (), {'is_available': lambda: False}),
        'nn': type('nn', (), {'Module': DummyModule}),
        'optim': type('optim', (), {'Optimizer': DummyOptimizer})
    })()
    nn = torch.nn
    optim = torch.optim

# Create type variables for better type hints
if TORCH_AVAILABLE:
    ModelType = TypeVar('ModelType', bound=nn.Module)
    OptimizerType = TypeVar('OptimizerType', bound=optim.Optimizer)
else:
    ModelType = TypeVar('ModelType', bound=DummyModule)
    OptimizerType = TypeVar('OptimizerType', bound=DummyOptimizer)

def save_model(
    model: torch.nn.Module,
    save_dir: Union[str, Path],
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """Save model checkpoint to disk.
    
    Args:
        model: Model to save
        save_dir: Directory to save the model
        model_name: Name of the model
        config: Model configuration
        tokenizer: Optional tokenizer to save
        optimizer: Optional optimizer state to save
        scheduler: Optional learning rate scheduler to save
        epoch: Current epoch
        global_step: Current global step
        metrics: Optional metrics to save with the model
        
    Returns:
        Path to the saved checkpoint
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config or {},
        'epoch': epoch,
        'global_step': global_step,
        'metrics': metrics or {}
    }
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Determine checkpoint filename
    checkpoint_name = f"{model_name}"
    if epoch is not None:
        checkpoint_name += f"_epoch_{epoch}"
    if global_step is not None:
        checkpoint_name += f"_step_{global_step}"
    checkpoint_name += ".pt"
    
    checkpoint_path = save_dir / checkpoint_name
    
    # Save model checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")
    
    # Save config as JSON
    if config:
        config_path = save_dir / f"{model_name}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    # Save tokenizer if provided
    if tokenizer is not None and hasattr(tokenizer, 'save_pretrained'):
        tokenizer_path = save_dir / f"{model_name}_tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
    
    return str(checkpoint_path)

def load_model(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint from disk.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to the checkpoint file
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load the model on
        
    Returns:
        Dictionary containing loaded checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback for older checkpoints
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded model checkpoint from {checkpoint_path}")
    
    # Return additional checkpoint information
    return {
        'epoch': checkpoint.get('epoch'),
        'global_step': checkpoint.get('global_step'),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_parameters(model: torch.nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze model parameters.
    
    Args:
        model: Model to modify
        freeze: Whether to freeze (True) or unfreeze (False) parameters
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    logger.info(f"Model parameters {'frozen' if freeze else 'unfrozen'}")

def get_device(device: Union[str, torch.device] = None) -> torch.device:
    """Get the appropriate device for model training/inference.
    
    Args:
        device: Preferred device (None for auto-detection)
        
    Returns:
        torch.device: Device to use
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device
