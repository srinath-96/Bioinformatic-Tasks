"""
Evaluates the trained model on the test set.
"""
import torch
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from config import config
from models import ProteinClassifier

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation on the test set."""

    def __init__(self, model: ProteinClassifier, test_loader: DataLoader, class_names: List[str]):
        self.model = model.to(config.device)
        self.test_loader = test_loader
        self.class_names = class_names

    def _get_predictions(self) -> Tuple[List[int], List[int]]:
        """Gets predictions and true labels from the test set."""
        self.model.load_state_dict(torch.load(config.model_save_path / "best_model.pth"))
        self.model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                logits = self.model(
                    batch['input_ids'].to(config.device), 
                    batch['attention_mask'].to(config.device)
                )
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        return all_preds, all_labels

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """Plots and saves the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=self.class_names, yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        save_path = config.results_path / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")

    def evaluate(self) -> Dict:
        """Runs evaluation and returns a classification report."""
        logger.info("--- Starting Model Evaluation on Test Set ---")
        
        y_pred, y_true = self._get_predictions()
        
        report_str = classification_report(y_true, y_pred, target_names=self.class_names)
        logger.info("Classification Report:\n" + report_str)
        
        self.plot_confusion_matrix(y_true, y_pred)
        
        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        logger.info("--- Model Evaluation Complete ---")
        return report_dict
```

```python:Bioinformatic-Tasks/traditional_ml.py
"""
Runs traditional ML models for baseline comparison.
"""
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from typing import Dict

from config import config

logger = logging.getLogger(__name__)

class TraditionalMLComparison:
    """Compares traditional ML models on numerical features."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X = self.df[config.feature_cols]
        self.y = self.df['label']

    def _prepare_data(self):
        """Splits and scales the data."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=config.test_size, 
            random_state=config.random_state, 
            stratify=self.y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def run(self) -> Dict:
        """Runs the comparison and returns results."""
        logger.info("--- Starting Traditional ML Comparison ---")
        X_train, X_test, y_train, y_test = self._prepare_data()
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.random_state),
            "Random Forest": RandomForestClassifier(random_state=config.random_state),
            "Support Vector Machine": SVC(random_state=config.random_state)
        }
        
        all_reports = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            all_reports[name] = report
            
            logger.info(f"{name} -> Accuracy: {report['accuracy']:.4f}, Weighted F1: {report['weighted avg']['f1-score']:.4f}")

        logger.info("--- Traditional ML Comparison Complete ---")
        return all_reports
```

```python:Bioinformatic-Tasks/main.py
"""
Main execution script for Protein Function Prediction with Language Models.
"""
from transformers import EsmTokenizer

from config import config
from utils import setup_logging, print_system_info, save_json, plot_training_history
from data_processor import ProteinDataProcessor
from tokenizer import create_dataloaders
from models import ProteinClassifier
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from traditional_ml import TraditionalMLComparison

def main():
    """Main execution pipeline."""
    logger = setup_logging()
    print_system_info()

    # 1. Data Processing
    processor = ProteinDataProcessor()
    df, data_splits, class_mapping = processor.run()
    num_classes = len(class_mapping)

    # 2. Tokenization and DataLoaders
    tokenizer = EsmTokenizer.from_pretrained(config.model_name)
    train_loader, val_loader, test_loader = create_dataloaders(data_splits, tokenizer)
    
    # 3. Model Training
    model = ProteinClassifier(num_classes=num_classes)
    trainer = ModelTrainer(model, train_loader, val_loader)
    history = trainer.train()
    
    # 4. Model Evaluation
    evaluator = ModelEvaluator(model, test_loader, list(class_mapping.values()))
    protein_lm_report = evaluator.evaluate()
    
    # 5. Traditional ML Comparison
    ml_comparison = TraditionalMLComparison(df)
    traditional_ml_reports = ml_comparison.run()
    
    # 6. Save Results
    final_results = {
        "protein_lm_report": protein_lm_report,
        "traditional_ml_reports": traditional_ml_reports,
        "training_history": history,
        "config": config.__dict__
    }
    save_json(final_results, config.results_path / "final_summary.json")
    plot_training_history(history, config.results_path / "training_history.png")
    
    logger.info("🎉 Project completed successfully!")

if __name__ == "__main__":
    main()
```

```text:Bioinformatic-Tasks/requirements.txt
# Core dependencies
torch>=2.0
transformers>=4.30
scikit-learn>=1.2
pandas>=2.0
biopython>=1.80

# Visualization and utilities
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.60
```

Perfect! I've recreated all your code files with the M1 Mac optimizations we had made:

- **Smaller model**: `facebook/esm2_t6_8M_UR50D` (8M parameters instead of 35M)
- **Reduced batch size**: `4` instead of `16` 
- All the modular structure intact

The project is now fully restored and ready to run. You can now execute:

```bash
python main.py
```

All the files should work exactly as they did before! 