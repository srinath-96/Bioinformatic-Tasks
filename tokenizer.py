"""
Protein sequence tokenization and PyTorch Dataset creation.
"""
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer
from typing import List, Dict, Tuple

from config import config

logger = logging.getLogger(__name__)

class ProteinDataset(Dataset):
    """Custom PyTorch Dataset for protein sequences."""
    def __init__(self, sequences: List[str], labels: List[int], tokenizer: EsmTokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = ' '.join(list(self.sequences[idx]))  # ESM needs spaces
        inputs = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=config.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_dataloaders(splits: tuple, tokenizer: EsmTokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for train, val, and test sets."""
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    
    train_dataset = ProteinDataset(X_train, y_train, tokenizer)
    val_dataset = ProteinDataset(X_val, y_val, tokenizer)
    test_dataset = ProteinDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    logger.info("DataLoaders created.")
    return train_loader, val_loader, test_loader 