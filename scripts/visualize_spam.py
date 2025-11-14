"""
Visualization tool for spam classification analysis.

Generate histograms, token frequency plots, confusion matrices, ROC/PR curves.

Examples:
  # Analyze dataset distribution
  python scripts/visualize_spam.py --input datasets/sms_spam_no_header.csv --text-col col_1 --label-col col_0

  # Generate ROC curve from predictions
  python scripts/visualize_spam.py --roc-data predictions.csv --labels col_0 --probs spam_probability

  # Token frequency for ham class
  python scripts/visualize_spam.py --input datasets/sms_spam_no_header.csv --text-col col_1 --label-col col_0 --tokens-for ham --topn 30
"""
import argparse
import sys
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def load_csv(path: str, no_header: bool = True) -> pd.DataFrame:
    """Load CSV with optional header."""
    if no_header:
        return pd.read_csv(path, header=None)
    return pd.read_csv(path)


def token_topn(series: pd.Series, topn: int = 20) -> List[Tuple[str, int]]:
    """Extract top-N most frequent tokens."""
    counter = Counter()
    for s in series.astype(str):
        counter.update(s.split())
    return counter.most_common(topn)


def plot_class_distribution(df: pd.DataFrame, label_col: str, output: str = None):
    """Plot class distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[label_col].value_counts().sort_index()
    colors = ['#1f77b4', '#ff7f0e']
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=100)
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_message_lengths(df: pd.DataFrame, text_col: str, label_col: str = None, output: str = None):
    """Plot message length distribution."""
    df_copy = df.copy()
    df_copy['length'] = df_copy[text_col].astype(str).str.len()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if label_col:
        for label in df_copy[label_col].unique():
            subset = df_copy[df_copy[label_col] == label]['length']
            ax.hist(subset, bins=30, alpha=0.6, label=str(label))
        ax.legend()
    else:
        ax.hist(df_copy['length'], bins=30, color='steelblue')
    
    ax.set_xlabel('Message Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Message Length Distribution')
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=100)
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_tokens(df: pd.DataFrame, text_col: str, label_col: str, label_value: str, topn: int = 20, output: str = None):
    """Plot token frequency for a specific class."""
    subset = df[df[label_col] == label_value][text_col]
    top = token_topn(subset, topn)
    
    if not top:
        print(f"No tokens found for {label_col}={label_value}")
        return
    
    toks, freqs = zip(*top)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=list(freqs), y=list(toks), ax=ax, palette="viridis")
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Token')
    ax.set_title(f'Top {topn} Tokens: {label_col}={label_value}')
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=100)
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None, output: str = None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels or [0, 1],
                yticklabels=labels or [0, 1])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=100)
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, output: str = None):
    """Plot ROC and Precision-Recall curves side-by-side."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    ax1.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # Precision-Recall curve
    ax2.plot(rec, prec, label='PR Curve', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=100)
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray, output: str = None):
    """Plot threshold sweep (accuracy, precision, recall, F1)."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    thresholds = np.linspace(0.1, 0.9, 50)
    metrics = {'threshold': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        metrics['threshold'].append(t)
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['threshold'], metrics['accuracy'], label='Accuracy', linewidth=2)
    ax.plot(metrics['threshold'], metrics['precision'], label='Precision', linewidth=2)
    ax.plot(metrics['threshold'], metrics['recall'], label='Recall', linewidth=2)
    ax.plot(metrics['threshold'], metrics['f1'], label='F1 Score', linewidth=2)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Decision Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=100)
        print(f"Saved: {output}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(
        description="Visualization tool for spam classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data input
    p.add_argument("--input", help="Input CSV path")
    p.add_argument("--text-col", help="Text column name (e.g., col_1 or text)")
    p.add_argument("--label-col", help="Label column name (e.g., col_0 or label)")
    p.add_argument("--no-header", action="store_true", help="CSV has no header")
    
    # Visualizations
    viz_group = p.add_argument_group("visualizations")
    viz_group.add_argument("--dist", action="store_true", help="Plot class distribution")
    viz_group.add_argument("--lengths", action="store_true", help="Plot message lengths")
    viz_group.add_argument("--tokens", action="store_true", help="Plot token frequency")
    viz_group.add_argument("--tokens-for", help="Token class filter (e.g., spam, ham, 0, 1)")
    viz_group.add_argument("--topn", type=int, default=20, help="Top-N tokens")
    viz_group.add_argument("--confusion", action="store_true", help="Plot confusion matrix")
    viz_group.add_argument("--roc", action="store_true", help="Plot ROC/PR curves")
    viz_group.add_argument("--threshold-sweep", action="store_true", help="Plot threshold sweep")
    
    # Model output
    p.add_argument("--output-dir", default="outputs", help="Output directory for saved plots")
    p.add_argument("--no-save", action="store_true", help="Don't save plots (display instead)")
    
    # For ROC/confusion (if using predictions CSV)
    p.add_argument("--roc-data", help="CSV with predictions for ROC/confusion")
    p.add_argument("--labels", help="Labels column in predictions CSV")
    p.add_argument("--probs", help="Probability column in predictions CSV")
    
    args = p.parse_args()
    
    # Ensure output dir exists
    if not args.no_save:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load primary data
    if args.input:
        df = load_csv(args.input, no_header=args.no_header)
        
        # Plot distributions
        if args.dist:
            out = f"{args.output_dir}/class_distribution.png" if not args.no_save else None
            plot_class_distribution(df, args.label_col or (0 if args.no_header else 'label'), output=out)
        
        if args.lengths:
            out = f"{args.output_dir}/message_lengths.png" if not args.no_save else None
            plot_message_lengths(df, args.text_col or (1 if args.no_header else 'text'),
                               label_col=args.label_col or (0 if args.no_header else 'label'),
                               output=out)
        
        if args.tokens:
            if not args.tokens_for:
                print("Error: --tokens requires --tokens-for (e.g., spam, ham, 0, 1)")
                return 1
            out = f"{args.output_dir}/tokens_{args.tokens_for}.png" if not args.no_save else None
            plot_tokens(df, args.text_col or (1 if args.no_header else 'text'),
                       args.label_col or (0 if args.no_header else 'label'),
                       args.tokens_for, topn=args.topn, output=out)
    
    # Load predictions data for ROC/confusion
    if args.roc_data:
        df_pred = pd.read_csv(args.roc_data)
        y_true = df_pred[args.labels].values
        y_proba = df_pred[args.probs].values
        
        if args.confusion:
            y_pred = (y_proba >= 0.5).astype(int)
            out = f"{args.output_dir}/confusion_matrix.png" if not args.no_save else None
            plot_confusion_matrix(y_true, y_pred, labels=['Ham', 'Spam'], output=out)
        
        if args.roc:
            out = f"{args.output_dir}/roc_pr_curves.png" if not args.no_save else None
            plot_roc_pr_curves(y_true, y_proba, output=out)
        
        if args.threshold_sweep:
            out = f"{args.output_dir}/threshold_sweep.png" if not args.no_save else None
            plot_threshold_sweep(y_true, y_proba, output=out)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
