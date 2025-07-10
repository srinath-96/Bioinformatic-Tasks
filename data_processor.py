"""
Data loading, cleaning, and preprocessing for protein sequences.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import config, COLUMN_MAPPING, VALID_AMINO_ACIDS

logger = logging.getLogger(__name__)

class ProteinDataProcessor:
    """Handles all data processing steps."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.class_names = []

    def load_and_clean(self) -> pd.DataFrame:
        """Loads and cleans the raw data from CSV."""
        logger.info(f"Loading data from {config.data_path}...")
        try:
            df = pd.read_csv(config.data_path)
            df = df.rename(columns=COLUMN_MAPPING)
        except FileNotFoundError:
            logger.error(f"Data file not found: {config.data_path}")
            raise

        initial_count = len(df)
        df = df.dropna(subset=['Sequence'])
        df['Sequence'] = df['Sequence'].str.strip()
        df = df[df['Sequence'] != '']
        logger.info(f"Removed {initial_count - len(df)} rows with missing sequences.")
        return df

    def validate_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates sequences and filters by length."""
        # Validate amino acids
        is_valid = df['Sequence'].apply(lambda s: set(s.upper()).issubset(VALID_AMINO_ACIDS))
        if not is_valid.all():
            logger.warning(f"Found {len(df) - is_valid.sum()} invalid sequences. Removing them.")
            df = df[is_valid]

        # Filter by length
        seq_len = df['Sequence'].str.len()
        len_mask = (seq_len >= 20) & (seq_len <= 1000)
        if not len_mask.all():
            logger.info(f"Filtering {len(df) - len_mask.sum()} sequences by length (20-1000).")
            df = df[len_mask]
        
        df = df.copy()
        return df

    def encode_labels_and_split(self, df: pd.DataFrame) -> Tuple[List, List, List, List, List, List, dict]:
        """Encodes labels and splits data into train, val, and test sets."""
        df['label'] = self.label_encoder.fit_transform(df['Class'])
        self.class_names = list(self.label_encoder.classes_)
        class_mapping = {i: name for i, name in enumerate(self.class_names)}
        
        logger.info(f"Class mapping: {class_mapping}")

        sequences = df['Sequence'].tolist()
        labels = df['label'].tolist()

        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=config.test_size, random_state=config.random_state, stratify=labels
        )
        val_size_adj = config.val_size / (1 - config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=config.random_state, stratify=y_temp
        )

        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test, class_mapping
    
    def run(self) -> Tuple[pd.DataFrame, Tuple, dict]:
        """Runs the complete data processing pipeline."""
        logger.info("--- Starting Data Processing ---")
        df = self.load_and_clean()
        df = self.validate_and_filter(df)
        data_splits = self.encode_labels_and_split(df)
        logger.info("--- Data Processing Complete ---")
        return df, data_splits[:-1], data_splits[-1]
