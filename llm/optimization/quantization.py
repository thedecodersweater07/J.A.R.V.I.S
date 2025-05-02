"""
Quantization module for optimizing model size and inference speed.
"""

import torch
import torch.nn as nn
import torch.quantization
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class QuantizationConfig:
    def __init__(
        self,
        dtype: str = "qint8",
        scheme: str = "symmetric",
        granularity: str = "per_tensor",
        calibration_method: str = "histogram",
        num_calibration_batches: int = 100,
        **kwargs
    ):
        self.dtype = dtype
        self.scheme = scheme
        self.granularity = granularity
        self.calibration_method = calibration_method
        self.num_calibration_batches = num_calibration_batches
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class ModelQuantizer:
    """Handles model quantization for reducing model size and improving inference speed."""
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        self.model = model
        self.config = config
        self.original_state_dict = None
        self.quantized_model = None
        
    def prepare_for_quantization(self):
        """Prepare model for quantization by adding observers."""
        self.original_state_dict = self.model.state_dict()
        self.model.eval()
        
        if self.config.dtype == "qint8":
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        elif self.config.dtype == "float16":
            self.model.qconfig = torch.quantization.float16_dynamic_qconfig
            
        torch.quantization.prepare(self.model, inplace=True)
        logger.info("Model prepared for quantization")

    def calibrate(self, calibration_loader):
        """Calibrate the model using calibration data."""
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= self.config.num_calibration_batches:
                    break
                if isinstance(batch, Dict):
                    self.model(**batch)
                else:
                    self.model(batch)
        logger.info("Calibration completed")

    def quantize(self) -> nn.Module:
        """Convert the model to quantized format."""
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        logger.info(f"Model quantized to {self.config.dtype}")
        return self.quantized_model

    def evaluate_model_size(self) -> Dict[str, float]:
        """Compare original and quantized model sizes."""
        original_size = sum(p.numel() * p.element_size() 
                          for p in self.model.parameters()) / (1024 * 1024)
        quantized_size = sum(p.numel() * p.element_size() 
                           for p in self.quantized_model.parameters()) / (1024 * 1024)
        
        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size
        }

    def restore_original(self):
        """Restore model to pre-quantization state."""
        if self.original_state_dict is not None:
            self.model.load_state_dict(self.original_state_dict)
            logger.info("Model restored to original state")
