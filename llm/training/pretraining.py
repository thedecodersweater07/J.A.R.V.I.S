#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pretraining module for language models.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PretrainingConfig:
    """Configuration class for pretraining parameters."""
    
    def __init__(
        self,
        model_size="base",
        batch_size=32,
        learning_rate=5e-5,
        max_seq_length=512,
        epochs=10,
        warmup_steps=10000,
        optimizer="adam",
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_steps=10000,
        seed=42,
        **kwargs
    ):
        self.model_size = model_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.seed = seed
        
        # Add any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class PreTrainer:
    """Main trainer class for pretraining language models."""
    
    def __init__(self, config, model, tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
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
        
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(optimizer_grouped_parameters, lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def train(self, train_dataloader):
        """Main training loop for pretraining."""
        logger.info("Starting pretraining...")
        self.model.train()
        
        total_loss = 0
        global_step = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Normalize loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # Save checkpoint
                if global_step > 0 and global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step)
                    
                # Log progress
                if step % 100 == 0:
                    logger.info(
                        f"Epoch: {epoch+1}/{self.config.epochs} | "
                        f"Step: {step}/{len(train_dataloader)} | "
                        f"Loss: {loss.item():.4f}"
                    )
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
        
        # Training complete
        avg_total_loss = total_loss / self.config.epochs
        logger.info(f"Pretraining complete | Avg total loss: {avg_total_loss:.4f}")
        
        # Save final model
        self._save_checkpoint("final")
        
        return {"avg_total_loss": avg_total_loss}
    
    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        output_dir = f"checkpoints/pretrain_step_{step}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training config
        torch.save(self.config.__dict__, os.path.join(output_dir, "training_config.bin"))


def main():
    """Main entry point for pretraining script."""
    parser = argparse.ArgumentParser(description="Pretraining script for language models")
    
    # Model and data arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    
    # Training arguments
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./output")
    
    args = parser.parse_args()
    
    # Load model, prepare data, and train
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    
    # Create dummy data loader for demonstration
    from torch.utils.data import Dataset
    
    class DummyDataset(Dataset):
        def __len__(self):
            return 1000
        
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 30000, (args.max_seq_length,)),
                "attention_mask": torch.ones(args.max_seq_length),
                "labels": torch.randint(0, 30000, (args.max_seq_length,))
            }
    
    train_dataloader = DataLoader(DummyDataset(), batch_size=args.batch_size)
    
    # Initialize trainer and train
    config = PretrainingConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
    )
    
    trainer = PreTrainer(config, model, tokenizer)
    trainer.train(train_dataloader)


if __name__ == "__main__":
    main()