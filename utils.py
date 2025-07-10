"""
Utility functions for the Protein Function Prediction project.
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

from config import config

def setup_logging():
    """Set up logging to file and console."""
    log_file = config.logs_path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_json(data: Dict[str, Any], path: Path):
    """Save dictionary data to a JSON file."""
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, Path): return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=convert)
    logging.info(f"Results saved to {path}")

def plot_training_history(history: Dict[str, List], save_path: Path):
    """Plot training and validation history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(1, len(history['train_losses']) + 1)
    
    ax1.plot(epochs, history['train_losses'], 'b-o', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-o', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    
    ax2.plot(epochs, history['val_accuracies'], 'g-o', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy'); ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()
    logging.info(f"Training history plot saved to {save_path}")

def print_system_info():
    """Print system and configuration information."""
    import torch
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("🔧 SYSTEM & CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Device: {config.device}")
    if config.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info("=" * 50) 