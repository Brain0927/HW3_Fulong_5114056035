"""
Spam classifier prediction script — single text or batch CSV mode.

Examples:
  # Single prediction
  python scripts/predict_spam.py --text "Free cash now!"

  # Batch prediction from CSV
  python scripts/predict_spam.py \\
    --input datasets/processed/sms_spam_clean.csv \\
    --text-col text_clean \\
    --output predictions.csv

  # Custom threshold
  python scripts/predict_spam.py --text "Test" --threshold 0.4
"""
import argparse
import json
import os
import re
import sys
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# Text normalization (matching training preprocessing)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")


def normalize_text(text: str, keep_numbers: bool = False) -> str:
    """Normalize text to match training preprocessing."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = URL_RE.sub("<URL>", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = PHONE_RE.sub("<PHONE>", t)
    if not keep_numbers:
        t = re.sub(r"\d+", "<NUM>", t)
    t = re.sub(r"[^\w\s<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_artifacts(models_dir: str) -> Tuple[object, object, str, str]:
    """Load model, vectorizer, and label mapping."""
    # Try to find model file (logistic_regression.pkl or model.pkl)
    model_path = None
    for name in ["logistic_regression.pkl", "model.pkl"]:
        full_path = os.path.join(models_dir, name)
        if os.path.exists(full_path):
            model_path = full_path
            break
    
    if not model_path:
        raise FileNotFoundError(f"Model file not found in {models_dir}")
    
    vec_path = os.path.join(models_dir, "vectorizer.pkl")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer file not found in {models_dir}")
    
    vec = joblib.load(vec_path)
    model = joblib.load(model_path)
    
    # Try to load label mapping
    pos_label, neg_label = "spam", "ham"
    label_map_path = os.path.join(models_dir, "label_mapping.json")
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
                pos_label = label_map.get("positive", "spam")
                neg_label = label_map.get("negative", "ham")
        except Exception as e:
            print(f"Warning: Could not load label mapping: {e}", file=sys.stderr)
    
    return vec, model, pos_label, neg_label


def predict_single(text: str, vectorizer: object, model: object,
                   threshold: float = 0.5) -> Tuple[str, float]:
    """Predict for a single text."""
    clean_text = normalize_text(text)
    X = vectorizer.transform([clean_text])
    proba = float(model.predict_proba(X)[0, 1])
    pred = 1 if proba >= threshold else 0
    return pred, proba, clean_text


def predict_batch(df: pd.DataFrame, text_col: str, vectorizer: object,
                  model: object, threshold: float = 0.5) -> pd.DataFrame:
    """Predict for a batch of texts."""
    df_copy = df.copy()
    
    # Normalize texts
    df_copy['text_normalized'] = df_copy[text_col].astype(str).apply(normalize_text)
    
    # Vectorize
    X = vectorizer.transform(df_copy['text_normalized'])
    
    # Predict
    proba = model.predict_proba(X)[:, 1]
    predictions = (proba >= threshold).astype(int)
    
    # Add results
    df_copy['spam_probability'] = proba
    df_copy['prediction'] = predictions
    df_copy['label'] = df_copy['prediction'].map({0: 'ham', 1: 'spam'})
    
    return df_copy


def main():
    p = argparse.ArgumentParser(
        description="Predict spam/ham for single text or batch CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input modes
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text to classify")
    group.add_argument("--input", help="Input CSV path for batch prediction")
    
    p.add_argument("--text-col", default="text", help="Text column name in CSV")
    p.add_argument("--output", help="Output CSV path (batch mode only)")
    p.add_argument("--models-dir", default="models", help="Directory with model artifacts")
    p.add_argument("--threshold", type=float, default=0.5,
                  help="Decision threshold for spam classification")
    
    args = p.parse_args()
    
    # Load artifacts
    try:
        vec, model, pos_label, neg_label = load_artifacts(args.models_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not load model artifacts from {args.models_dir}: {e}", file=sys.stderr)
        return 1
    
    # Single prediction mode
    if args.text:
        pred, proba, clean = predict_single(args.text, vec, model, args.threshold)
        label = "SPAM" if pred == 1 else "HAM"
        
        print(f"\n{'='*60}")
        print(f"Original:   {args.text}")
        print(f"Normalized: {clean}")
        print(f"{'='*60}")
        print(f"Prediction: {label}")
        print(f"Probability (spam): {proba:.4f}")
        print(f"Threshold: {args.threshold}")
        print(f"{'='*60}\n")
        
        return 0
    
    # Batch prediction mode
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        
        df = pd.read_csv(args.input)
        if args.text_col not in df.columns:
            print(f"Error: Column '{args.text_col}' not found in CSV. Available: {list(df.columns)}", file=sys.stderr)
            return 1
        
        print(f"Processing {len(df)} rows...")
        df_pred = predict_batch(df, args.text_col, vec, model, args.threshold)
        
        # Display summary
        spam_count = (df_pred['prediction'] == 1).sum()
        ham_count = (df_pred['prediction'] == 0).sum()
        
        print(f"\nResults:")
        print(f"  Total: {len(df_pred)}")
        print(f"  Spam: {spam_count} ({spam_count/len(df_pred)*100:.1f}%)")
        print(f"  Ham: {ham_count} ({ham_count/len(df_pred)*100:.1f}%)")
        
        # Save output
        if args.output:
            df_pred.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")
        else:
            print(f"\nFirst 10 predictions:")
            print(df_pred[['text_normalized', 'spam_probability', 'label']].head(10))
        
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
