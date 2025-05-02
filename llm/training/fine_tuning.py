#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning module for language models.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FineTuningConfig:
    """Configuration class for fine-tuning parameters."""
    
    def __init__(
        self,
        task_type="classification",
        num_labels=2,
        batch_size=16,
        learning_rate=2e-5,
        max_seq_length=128,
        epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=42,
        **kwargs
    ):
        self.task_type = task_type
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.seed = seed
        
        # Add any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class FineTuner:
    """Main trainer class for fine-tuning language models."""
    
    def __init__(self, config, model, tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        logger.info(f"Using device: {self.device}")
    
    def _get_optimizer(self):
        """Initialize the optimizer based on config settings."""
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
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.config.learning_rate
        )
    
    def _get_scheduler(self, num_training_steps):
        """Initialize the learning rate scheduler."""
        from torch.optim.lr_scheduler import LambdaLR
        
        # Linear warmup and decay
        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, 
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - warmup_steps))
            )
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop for fine-tuning."""
        logger.info("Starting fine-tuning...")
        self.model.train()
        
        # Setup learning rate scheduler
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.config.epochs
        scheduler = self._get_scheduler(num_training_steps)
        
        # Track metrics
        train_losses = []
        eval_metrics = {}
        best_eval_metric = float('inf')
        global_step = 0
        
        # Training loop
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            self.model.train()
            
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{self.config.epochs}") as pbar:
                for step, batch in enumerate(train_dataloader):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    
                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    pbar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
                    
                    # Gradient accumulation
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        
                        # Optimizer and scheduler step
                        self.optimizer.step()
                        scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                    
                    pbar.update(1)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            train_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f}")
            
            # Evaluate at the end of the epoch if requested
            if eval_dataloader is not None and self.config.evaluation_strategy == "epoch":
                metrics = self.evaluate(eval_dataloader)
                eval_metrics[f"epoch_{epoch+1}"] = metrics
                
                # Save best model
                if metrics["eval_loss"] < best_eval_metric:
                    best_eval_metric = metrics["eval_loss"]
                    self._save_checkpoint("best")
            
            # Save at the end of the epoch if requested
            if self.config.save_strategy == "epoch":
                self._save_checkpoint(f"epoch_{epoch+1}")
        
        # Save final model
        self._save_checkpoint("final")
        
        return {
            "train_losses": train_losses,
            "eval_metrics": eval_metrics,
            "best_eval_metric": best_eval_metric
        }
    
    def evaluate(self, eval_dataloader):
        """Evaluate the model on the evaluation set."""
        logger.info("Running evaluation...")
        self.model.eval()
        
        eval_loss = 0
        preds = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
                
                # Collect predictions and labels
                if self.config.task_type == "classification":
                    if hasattr(outputs, "logits"):
                        preds.append(outputs.logits.cpu().numpy())
                    if "labels" in batch:
                        labels.append(batch["labels"].cpu().numpy())
        
        # Calculate average loss
        eval_loss /= len(eval_dataloader)
        
        # Calculate additional metrics
        metrics = {"eval_loss": eval_loss}
        
        if self.config.task_type == "classification" and labels:
            # Convert lists to numpy arrays
            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            # Get predictions
            preds = np.argmax(preds, axis=1)
            
            # Calculate accuracy
            accuracy = (preds == labels).mean()
            metrics["accuracy"] = accuracy
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def _save_checkpoint(self, checkpoint_name):
        """Save model checkpoint."""
        output_dir = f"checkpoints/finetune_{checkpoint_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        # Save model and tokenizer
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
            if self.tokenizer and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save training config
        with open(os.path.join(output_dir, "fine_tuning_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)


def main():
    """Main entry point for fine-tuning script."""
    parser = argparse.ArgumentParser(description="Fine-tuning script for language models")
    
    # Model and data arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels
    )
    
    # Create dummy dataset for demonstration
    class TextClassificationDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 30000, (args.max_seq_length,)),
                "attention_mask": torch.ones(args.max_seq_length),
                "labels": torch.randint(0, args.num_labels, (1,)).item()
            }
    
    train_dataset = TextClassificationDataset()
    eval_dataset = TextClassificationDataset(size=20)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    
    # Initialize trainer and train
    config = FineTuningConfig(
        task_type=args.task_type,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
    )
    
    trainer = FineTuner(config, model, tokenizer)
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()