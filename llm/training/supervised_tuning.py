#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Supervised tuning module for language models.
"""

import os
import torch
import torch.nn as nn
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SupervisedConfig:
    """Configuration class for supervised tuning parameters."""
    
    def __init__(
        self,
        learning_rate=3e-5,
        batch_size=16,
        max_seq_length=512,
        epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=42,
        **kwargs
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.seed = seed
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class SupervisedTrainer:
    """Main trainer class for supervised tuning."""
    
    def __init__(self, config, model, tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = self._get_optimizer()
        logger.info(f"Using device: {self.device}")

    def _get_optimizer(self):
        """Initialize the optimizer."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def _get_scheduler(self, num_training_steps: int):
        """Initialize the learning rate scheduler."""
        from transformers import get_scheduler
        
        return get_scheduler(
            "linear",
            self.optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps,
        )

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop for supervised tuning."""
        logger.info("Starting supervised training...")
        self.model.train()
        
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.config.epochs
        scheduler = self._get_scheduler(num_training_steps)
        
        # Training metrics
        train_losses = []
        eval_metrics = {}
        best_eval_metric = float('inf')
        global_step = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            self.model.train()
            
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{self.config.epochs}") as pbar:
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    
                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    pbar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
                    
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                        
                        self.optimizer.step()
                        scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                    
                    pbar.update(1)
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            train_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f}")
            
            if eval_dataloader is not None and self.config.evaluation_strategy == "epoch":
                metrics = self.evaluate(eval_dataloader)
                eval_metrics[f"epoch_{epoch+1}"] = metrics
                
                if metrics["eval_loss"] < best_eval_metric:
                    best_eval_metric = metrics["eval_loss"]
                    self._save_checkpoint("best")
            
            if self.config.save_strategy == "epoch":
                self._save_checkpoint(f"epoch_{epoch+1}")
        
        self._save_checkpoint("final")
        return {
            "train_losses": train_losses,
            "eval_metrics": eval_metrics,
            "best_eval_metric": best_eval_metric
        }

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the evaluation set."""
        logger.info("Running evaluation...")
        self.model.eval()
        
        eval_loss = 0
        eval_predictions = []
        eval_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                eval_loss += outputs.loss.item()
                
                if hasattr(outputs, "logits"):
                    eval_predictions.extend(
                        outputs.logits.argmax(dim=-1).cpu().numpy()
                    )
                if "labels" in batch:
                    eval_labels.extend(batch["labels"].cpu().numpy())
        
        eval_loss = eval_loss / len(eval_dataloader)
        metrics = {"eval_loss": eval_loss}
        
        if eval_predictions and eval_labels:
            accuracy = np.mean(np.array(eval_predictions) == np.array(eval_labels))
            metrics["accuracy"] = accuracy
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics

    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        output_dir = f"checkpoints/supervised_{checkpoint_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.bin"))
        
        torch.save(self.config.__dict__, os.path.join(output_dir, "supervised_config.bin"))
