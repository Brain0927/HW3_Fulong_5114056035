"""
Model training and evaluation module for spam classification.
"""
import os
import pickle
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "{}.pkl")


def prepare_features(
    messages: pd.Series,
    max_features: int = 5000,
    vectorizer: TfidfVectorizer = None
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Extract TF-IDF features from messages.
    
    Args:
        messages: Text data
        max_features: Maximum number of features
        vectorizer: Existing vectorizer (if None, fit new one)
        
    Returns:
        Tuple of (feature matrix, vectorizer)
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(messages)
        logger.info(f"Fitted vectorizer with {len(vectorizer.vocabulary_)} features")
    else:
        X = vectorizer.transform(messages)
    
    return X, vectorizer


def split_data(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
    """Split data into train/val/test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    **kwargs
) -> Any:
    """
    Train a classifier model.
    
    Args:
        model_name: Name of model ('logistic_regression', 'naive_bayes', 'svm')
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional model parameters
        
    Returns:
        Trained model
    """
    if model_name == 'logistic_regression':
        model = LogisticRegression(
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        )
    elif model_name == 'naive_bayes':
        model = MultinomialNB()
    elif model_name == 'svm':
        model = LinearSVC(
            max_iter=kwargs.get('max_iter', 2000),
            random_state=kwargs.get('random_state', 42),
            dual=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"Training complete")
    
    return model


def evaluate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: pd.Series,
    X_test: np.ndarray = None,
    y_test: pd.Series = None
) -> Dict[str, Any]:
    """
    Evaluate model on validation and test sets.
    
    Returns:
        Dictionary of metrics
    """
    y_pred_val = model.predict(X_val)
    
    metrics = {
        'val_accuracy': accuracy_score(y_val, y_pred_val),
        'val_precision': precision_score(y_val, y_pred_val),
        'val_recall': recall_score(y_val, y_pred_val),
        'val_f1': f1_score(y_val, y_pred_val),
        'val_confusion_matrix': confusion_matrix(y_val, y_pred_val),
        'val_classification_report': classification_report(y_val, y_pred_val),
    }
    
    # Try to get probability scores for ROC-AUC (not available for SVM's LinearSVC)
    try:
        y_proba_val = model.predict_proba(X_val)[:, 1]
        metrics['val_roc_auc'] = roc_auc_score(y_val, y_proba_val)
        metrics['val_proba'] = y_proba_val
    except AttributeError:
        logger.warning("Model does not support predict_proba, skipping ROC-AUC")
    
    if X_test is not None and y_test is not None:
        y_pred_test = model.predict(X_test)
        metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        metrics['test_precision'] = precision_score(y_test, y_pred_test)
        metrics['test_recall'] = recall_score(y_test, y_pred_test)
        metrics['test_f1'] = f1_score(y_test, y_pred_test)
        metrics['test_confusion_matrix'] = confusion_matrix(y_test, y_pred_test)
        metrics['test_classification_report'] = classification_report(y_test, y_pred_test)
        
        try:
            y_proba_test = model.predict_proba(X_test)[:, 1]
            metrics['test_roc_auc'] = roc_auc_score(y_test, y_proba_test)
            metrics['test_proba'] = y_proba_test
        except AttributeError:
            pass
    
    logger.info(f"Val Accuracy: {metrics['val_accuracy']:.4f}")
    if 'test_accuracy' in metrics:
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    
    return metrics


def save_model(model: Any, model_name: str) -> None:
    """Save trained model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = MODEL_PATH_TEMPLATE.format(model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")


def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    """Save TF-IDF vectorizer to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectorizer saved to {VECTORIZER_PATH}")


def load_model(model_name: str) -> Any:
    """Load trained model from disk."""
    model_path = MODEL_PATH_TEMPLATE.format(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model


def load_vectorizer() -> TfidfVectorizer:
    """Load TF-IDF vectorizer from disk."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.info(f"Vectorizer loaded from {VECTORIZER_PATH}")
    return vectorizer
