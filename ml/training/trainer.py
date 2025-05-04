"""Model training module"""
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model: Optional[nn.Module] = None, config: Dict[str, Any] = None):
        self.config = config or {"epochs": 10, "batch_size": 32}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_model(model)
        
    def _set_model(self, model: Optional[nn.Module]) -> None:
        """Safely set model and move to device if provided"""
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
        
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, float]:
        """Train model"""
        if self.model is None:
            raise ValueError("Model must be set before training")
            
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters())
            
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(self.config.get("epochs", 10)):
            epoch_loss = 0.0
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}") as pbar:
                for batch in pbar:
                    optimizer.zero_grad()
                    loss = self._training_step(batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}: loss={avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
            
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"Validation loss: {val_loss:.4f}")
                
        return {
            "train_loss": total_loss / self.config.get("epochs", 10)
        }
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                loss = self._training_step(batch)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute single training step"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        return outputs.loss
