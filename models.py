"""
Neural network model definitions for protein function prediction.
"""
import logging
import torch
import torch.nn as nn
from transformers import EsmModel

from config import config

logger = logging.getLogger(__name__)

class ProteinClassifier(nn.Module):
    """Protein function classifier using a pre-trained ESM-2 model."""
    def __init__(self, num_classes: int):
        super().__init__()
        logger.info(f"Initializing ProteinClassifier with {config.model_name}")
        
        self.esm = EsmModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.esm.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use the [CLS] token's representation for classification
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits 