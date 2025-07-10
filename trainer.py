"""
Handles model training, validation, and checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Tuple
from tqdm.auto import tqdm
from pathlib import Path

from config import config
from models import ProteinClassifier

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Manages the training and validation process."""

    def __init__(self, model: ProteinClassifier, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_losses': [], 'val_losses': [], 'val_accuracies': []}
        self.best_accuracy = 0.0

    def _train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            logits = self.model(batch['input_ids'].to(config.device), batch['attention_mask'].to(config.device))
            loss = self.criterion(logits, batch['labels'].to(config.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> Tuple[float, float]:
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss, correct, total_samples = 0, 0, 0
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                logits = self.model(batch['input_ids'].to(config.device), batch['attention_mask'].to(config.device))
                labels = batch['labels'].to(config.device)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                pbar.set_postfix({'acc': f'{100*correct/total_samples:.2f}%'})
        accuracy = 100 * correct / total_samples
        return total_loss / len(self.val_loader), accuracy

    def train(self) -> Dict:
        """Runs the full training loop."""
        logger.info("--- Starting Model Training ---")
        for epoch in range(1, config.num_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_accuracy = self._validate_epoch()
            
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['val_accuracies'].append(val_accuracy)
            
            logger.info(f"Epoch {epoch}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), config.model_save_path / "best_model.pth")
                logger.info(f"🏆 New best model saved with accuracy: {self.best_accuracy:.2f}%")
        
        logger.info("--- Model Training Complete ---")
        return self.history 