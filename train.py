"""
Training pipeline script for spam classification models.
"""
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import get_data
from model_trainer import (
    prepare_features, split_data, train_model, evaluate_model,
    save_model, save_vectorizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model_name: str = "logistic_regression"):
    """
    Main training pipeline.
    
    Args:
        model_name: 'logistic_regression', 'naive_bayes', or 'svm'
    """
    # 1. Load data
    logger.info("Step 1: Loading data...")
    messages, labels = get_data(download=True)
    
    # 2. Prepare features
    logger.info("Step 2: Preparing features (TF-IDF)...")
    X, vectorizer = prepare_features(messages)
    
    # 3. Split data
    logger.info("Step 3: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, labels)
    
    # 4. Train model
    logger.info(f"Step 4: Training {model_name}...")
    model = train_model(model_name, X_train, y_train)
    
    # 5. Evaluate model
    logger.info("Step 5: Evaluating model...")
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
    
    # 6. Save model and vectorizer
    logger.info("Step 6: Saving model and vectorizer...")
    save_model(model, model_name)
    save_vectorizer(vectorizer)
    
    # 7. Log results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
    logger.info(f"Validation F1: {metrics['val_f1']:.4f}")
    if 'test_accuracy' in metrics:
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1: {metrics['test_f1']:.4f}")
    logger.info("="*60)
    
    # Save metrics to file
    metrics_to_save = {
        k: v for k, v in metrics.items()
        if k not in ['val_confusion_matrix', 'test_confusion_matrix',
                     'val_proba', 'test_proba', 'val_classification_report',
                     'test_classification_report']
    }
    
    import numpy as np
    for k, v in metrics_to_save.items():
        if isinstance(v, np.ndarray):
            metrics_to_save[k] = v.tolist()
    
    with open(f"models/metrics_{model_name}.json", 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    return model, vectorizer, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "naive_bayes", "svm"],
        help="Model to train"
    )
    args = parser.parse_args()
    
    main(args.model)
