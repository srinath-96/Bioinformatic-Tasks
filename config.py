"""
Configuration settings for the Protein Function Prediction project.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch
from typing import List

@dataclass
class Config:
    """Configuration class for the protein prediction project."""
    # --- Paths ---
    data_path: str = "proteinas_20000_enriquecido.csv"
    output_dir: str = "output/"
    
    # --- Model Parameters ---
    model_name: str = "facebook/esm2_t6_8M_UR50D"  # Smaller model for M1 Mac
    # Alternative models:
    # "facebook/esm2_t6_8M_UR50D"     # Smaller, for quick tests (CURRENT)
    # "facebook/esm2_t12_35M_UR50D"   # Medium - caused OOM on M1 Mac
    # "facebook/esm2_t30_150M_UR50D"  # Larger, for better performance
    max_length: int = 512
    dropout_rate: float = 0.3
    
    # --- Training Parameters ---
    num_epochs: int = 5
    batch_size: int = 4  # Reduced for M1 Mac
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    
    # --- Data Splitting ---
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # --- System ---
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # --- Feature columns for traditional ML ---
    feature_cols: List[str] = field(default_factory=lambda: [
        'Molecular_Weight', 'Isoelectric_Point', 'Hydrophobicity', 
        'Total_Charge', 'Polar_Proportion', 'Apolar_Proportion', 'Sequence_Length'
    ])
    
    # --- Derived Paths (initialized after creation) ---
    model_save_path: Path = field(init=False)
    results_path: Path = field(init=False)
    logs_path: Path = field(init=False)

    def __post_init__(self):
        """Create derived paths and ensure directories exist."""
        output_path = Path(self.output_dir)
        self.model_save_path = output_path / "models"
        self.results_path = output_path / "results"
        self.logs_path = output_path / "logs"
        
        for path in [self.model_save_path, self.results_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)

# --- Global Constants ---

# Column mapping from Portuguese to English
COLUMN_MAPPING = {
    "ID_Proteína": "Protein_ID",
    "Sequência": "Sequence", 
    "Massa_Molecular": "Molecular_Weight",
    "Ponto_Isoelétrico": "Isoelectric_Point",
    "Hidrofobicidade": "Hydrophobicity",
    "Carga_Total": "Total_Charge",
    "Proporção_Polar": "Polar_Proportion",
    "Proporção_Apolar": "Apolar_Proportion",
    "Comprimento_Sequência": "Sequence_Length",
    "Classe": "Class"
}

# Valid amino acids for protein sequences
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# Create a single global config instance
config = Config() 