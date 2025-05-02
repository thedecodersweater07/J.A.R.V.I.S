"""
Knowledge distillation module for transferring knowledge from large to small models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class DistillationConfig:
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        hard_label_weight: float = 0.5,
        hidden_layer_mapping: Optional[Dict] = None,
        use_attention_transfer: bool = False,
        **kwargs
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.hard_label_weight = hard_label_weight
        self.hidden_layer_mapping = hidden_layer_mapping
        self.use_attention_transfer = use_attention_transfer
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class KnowledgeDistiller:
    """Handles knowledge distillation from teacher to student model."""
    
    def __init__(
        self, 
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        
        self.teacher.eval()  # Teacher model should always be in eval mode
        
    def compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the distillation loss between teacher and student outputs."""
        # Compute soft targets with temperature
        soft_targets = F.softmax(teacher_outputs / self.config.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_outputs / self.config.temperature, dim=-1)
        
        # Compute distillation loss
        distillation_loss = F.kl_div(
            soft_predictions,
            soft_targets.detach(),
            reduction='batchmean'
        ) * (self.config.temperature ** 2)
        
        # Compute hard loss if labels are provided
        if labels is not None:
            hard_loss = F.cross_entropy(student_outputs, labels)
            total_loss = (
                self.config.hard_label_weight * hard_loss +
                (1 - self.config.hard_label_weight) * distillation_loss
            )
        else:
            total_loss = distillation_loss
            hard_loss = torch.tensor(0.0)
        
        metrics = {
            "distillation_loss": distillation_loss.item(),
            "hard_loss": hard_loss.item() if labels is not None else 0.0,
            "total_loss": total_loss.item()
        }
        
        return total_loss, metrics

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform a single training step of distillation."""
        self.student.train()
        optimizer.zero_grad()
        
        # Forward pass through both models
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)
            
        student_outputs = self.student(**batch)
        
        # Compute losses
        total_loss, metrics = self.compute_distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            batch.get("labels")
        )
        
        # Add attention transfer loss if configured
        if self.config.use_attention_transfer:
            attention_loss = self.compute_attention_transfer_loss(
                student_outputs.attentions,
                teacher_outputs.attentions
            )
            total_loss += self.config.alpha * attention_loss
            metrics["attention_loss"] = attention_loss.item()
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return metrics

    def compute_attention_transfer_loss(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention transfer loss between teacher and student."""
        if student_attention is None or teacher_attention is None:
            return torch.tensor(0.0)
            
        attention_loss = 0.0
        for student_att, teacher_att in zip(student_attention, teacher_attention):
            student_att = student_att.mean(dim=1)  # Average over heads
            teacher_att = teacher_att.mean(dim=1)  # Average over heads
            attention_loss += F.mse_loss(student_att, teacher_att)
            
        return attention_loss / len(student_attention)
