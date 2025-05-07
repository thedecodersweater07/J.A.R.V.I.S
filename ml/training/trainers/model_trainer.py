import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Base model trainer class"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def _set_model(self, model: Optional[nn.Module] = None) -> None:
        """Set and configure model"""
        if isinstance(model, nn.Module):
            self.model = model.to(self.device)
        else:
            # Create default model if none provided
            self.model = self._create_default_model()
    
    def _create_default_model(self) -> nn.Module:
        """Create a default model based on config"""
        # Basic default model
        model = nn.Sequential(
            nn.Linear(self.config.get("input_size", 10), 
                      self.config.get("hidden_size", 50)),
            nn.ReLU(),
            nn.Linear(self.config.get("hidden_size", 50), 
                      self.config.get("output_size", 2))
        ).to(self.device)
        return model
            
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Train the model"""
        self.model.train()
        total_loss = 0.0
        best_val_loss = float('inf')
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.config.get("learning_rate", 0.001))
        
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
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
        return {
            "train_loss": total_loss / self.config.get("epochs", 10),
            "best_val_loss": best_val_loss if val_dataloader else None
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
