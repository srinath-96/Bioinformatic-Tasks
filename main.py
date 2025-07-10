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
