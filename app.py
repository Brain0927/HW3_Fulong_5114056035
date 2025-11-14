"""
Streamlit app for spam email classification with advanced visualizations.
Inspired by Phase 4 visualizations: data distribution, token patterns, ROC/PR curves, and live inference.
"""
import sys
import os
import re
import json
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, PrecisionRecallDisplay
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_trainer import load_model, load_vectorizer
from data_loader import get_data

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page config
st.set_page_config(
    page_title="Spam Email Classifier — Advanced",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tab styling and highlighting
st.markdown("""
<style>
/* Tab container styling */
div[data-testid="stTabs"] {
    margin-top: 1rem;
}

/* Tab button styling */
button[data-testid="stTabButtonContainer"] {
    background-color: transparent;
}

/* Active tab styling */
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: bold;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    border-radius: 8px;
    padding: 8px 16px;
}

/* Inactive tab styling */
[data-testid="stTabs"] [aria-selected="false"] {
    background-color: #f0f2f6;
    color: #31333b;
    border-radius: 8px;
    padding: 8px 16px;
    transition: all 0.3s ease;
}

/* Tab hover effect */
[data-testid="stTabs"] [aria-selected="false"]:hover {
    background-color: #e0e3ed;
    transform: translateY(-2px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* Tab content background */
[data-testid="stTabContent"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
}

/* Metric cards enhancement */
[data-testid="stMetric"] {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 3px solid #667eea;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Info box styling */
div[data-testid="stAlert"] {
    border-radius: 8px;
    border-left: 4px solid;
}

/* Button styling */
button {
    border-radius: 8px;
    transition: all 0.3s ease;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* Divider enhancement */
hr {
    margin: 2rem 0;
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, #667eea, transparent);
}

/* Subheader styling */
h2, h3 {
    color: #667eea;
    font-weight: 600;
}

/* Section styling */
[data-testid="stColumn"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


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


@st.cache_resource
def load_resources():
    """Load model and vectorizer (cached)."""
    try:
        model = load_model("logistic_regression")
        vectorizer = load_vectorizer()
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data
def load_metrics():
    """Load metrics from file."""
    try:
        with open("models/metrics_logistic_regression.json", 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def load_threshold_sweep():
    """Load threshold sweep analysis data."""
    try:
        with open("models/threshold_sweep.json", 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def load_test_predictions():
    """Load test predictions for ROC/PR curves."""
    try:
        with open("models/test_predictions.json", 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def list_datasets() -> list:
    """List all available CSV datasets (recursively)."""
    datasets = []
    for root_dir in ["data", "datasets"]:
        if os.path.isdir(root_dir):
            # Recursively walk through all subdirectories
            for dirpath, dirnames, filenames in os.walk(root_dir):
                for name in filenames:
                    if name.endswith(".csv"):
                        datasets.append(os.path.join(dirpath, name))
    return sorted(datasets)


def detect_csv_format(df: pd.DataFrame) -> str:
    """Detect CSV format type: 'simple' (2 columns) or 'preprocessing' (8+ columns with preprocessing stages)."""
    cols = list(df.columns)
    
    # Check for preprocessing pipeline columns
    preprocessing_markers = [
        'text_raw', 'text_lower', 'text_contacts_masked', 'text_numbers',
        'text_stripped', 'text_whitespace', 'text_stopwords_removed'
    ]
    
    if len(cols) >= 8:
        # Check if it has preprocessing stage columns
        pipeline_cols = [c for c in cols if any(marker in str(c).lower() for marker in preprocessing_markers)]
        if len(pipeline_cols) >= 3:  # Has at least 3 preprocessing stages
            return 'preprocessing'
    
    # Default to simple format
    return 'simple'


def get_preprocessing_stages(df: pd.DataFrame) -> dict:
    """Extract preprocessing stages from dataframe."""
    cols = list(df.columns)
    
    # Order of preprocessing stages
    stage_order = [
        'text_raw',
        'text_lower',
        'text_contacts_masked',
        'text_numbers',
        'text_stripped',
        'text_whitespace',
        'text_stopwords_removed'
    ]
    
    stages = {}
    for stage in stage_order:
        # Find column matching this stage (case-insensitive)
        matching_col = next((c for c in cols if stage in str(c).lower()), None)
        if matching_col:
            stages[stage] = matching_col
    
    return stages


def infer_columns(df: pd.DataFrame) -> tuple:
    """Infer label and text columns from dataframe."""
    cols = list(df.columns)
    # Try common label column names
    label_candidates = [c for c in cols if str(c).lower() in ("label", "target", "col_0", "0")]
    # Try common text column names
    text_candidates = [c for c in cols if str(c).lower() in ("text", "message", "content", "col_1", "1")]
    
    label_col = label_candidates[0] if label_candidates else (cols[0] if cols else None)
    text_col = text_candidates[0] if text_candidates else (cols[-1] if len(cols) > 1 else (cols[0] if cols else None))
    
    return label_col, text_col


def token_topn(series: pd.Series, topn: int = 20) -> List[Tuple[str, int]]:
    """Extract top-N most frequent tokens."""
    counter = Counter()
    for s in series.astype(str):
        counter.update(s.split())
    return counter.most_common(topn)


@st.cache_data
def load_dataset(path: str = None):
    """Load and preprocess dataset (supports both simple and preprocessing pipeline formats)."""
    try:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
        else:
            messages, labels = get_data(download=False)
            df = pd.DataFrame({
                'text': messages.values,
                'label': labels.values
            })
        
        # Detect CSV format
        csv_format = detect_csv_format(df)
        
        # Infer columns
        label_col, text_col = infer_columns(df)
        
        if label_col and text_col:
            df_clean = df[[label_col, text_col]].copy()
            df_clean.columns = ['label', 'text']
            
            # Map labels to 0/1 if needed
            if df_clean['label'].dtype == 'object':
                unique_vals = df_clean['label'].unique()
                if 'spam' in unique_vals and 'ham' in unique_vals:
                    df_clean['label'] = df_clean['label'].map({'spam': 1, 'ham': 0})
                elif 'spam' in str(df_clean['label'].iloc[0]).lower() or 'ham' in str(df_clean['label'].iloc[0]).lower():
                    # Handle case-insensitive labels
                    df_clean['label'] = df_clean['label'].apply(
                        lambda x: 1 if 'spam' in str(x).lower() else 0
                    )
            
            # Create label_text mapping
            try:
                df_clean['label_text'] = df_clean['label'].map({0: 'ham', 1: 'spam'})
            except:
                # Fallback: create mapping from unique values
                unique_labels = df_clean['label'].unique()
                if len(unique_labels) == 2:
                    df_clean['label_text'] = df_clean['label'].apply(
                        lambda x: 'spam' if x == max(unique_labels) else 'ham'
                    )
            
            # Store format info in session state
            st.session_state.csv_format = csv_format
            if csv_format == 'preprocessing':
                st.session_state.preprocessing_stages = get_preprocessing_stages(df)
                st.session_state.full_df = df  # Store full dataframe with all columns
            
            return df_clean
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def plot_class_distribution_mpl(df: pd.DataFrame) -> None:
    """Plot class distribution using matplotlib."""
    if not HAS_MATPLOTLIB:
        st.warning("Matplotlib not available. Use Plotly visualization instead.")
        return
    
    counts = df['label_text'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#ff6b6b', '#51cf66']  # Red for spam, green for ham
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


def plot_tokens_mpl(df: pd.DataFrame, label_text: str, label_idx: int, topn: int = 20) -> None:
    """Plot token frequency using matplotlib."""
    if not HAS_MATPLOTLIB:
        st.warning("Matplotlib not available. Use Plotly visualization instead.")
        return
    
    subset = df[df['label'] == label_idx]['text']
    top = token_topn(subset, topn)
    
    if not top:
        st.info(f"No tokens found for {label_text}.")
        return
    
    toks, freqs = zip(*top)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(toks)))
    ax.barh(list(toks), list(freqs), color=colors_map, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_ylabel('Token', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {topn} Tokens in {label_text.upper()}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


def main():
    # Enhanced title with styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
        <h1 style="color: white; margin: 0; text-align: center;">📧 Spam Email Classification</h1>
        <h3 style="color: #e0d5ff; margin: 0.5rem 0 0 0; text-align: center;">Advanced Dashboard with ML & Visualizations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Phase 4+ Visualizations** — Data distribution, token analysis, model performance & live inference", help="Navigate through tabs to explore different features")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Section 1: Data Input
        st.subheader("📊 Data Input")
        datasets = list_datasets()
        if datasets:
            selected_ds = st.selectbox(
                "Select Dataset",
                options=[None] + datasets,
                format_func=lambda x: "Default (built-in)" if x is None else x,
                help="Choose which CSV dataset to analyze"
            )
        else:
            selected_ds = None
        
        # Load dataset to detect format
        df_preview = load_dataset(selected_ds) if 'selected_ds' in locals() else None
        csv_format = st.session_state.get('csv_format', 'simple')
        
        # Show format info
        if csv_format == 'preprocessing':
            st.info("📈 **Preprocessing Pipeline CSV Detected**\nYou can visualize each preprocessing stage!")
            
            # Preprocessing stage selector
            stages_dict = st.session_state.get('preprocessing_stages', {})
            if stages_dict:
                preprocessing_stages = list(stages_dict.keys())
                selected_stage = st.selectbox(
                    "Select Preprocessing Stage",
                    options=['All'] + preprocessing_stages,
                    format_func=lambda x: x.replace('_', ' ').title() if x != 'All' else 'View All Stages',
                    help="Choose a specific preprocessing stage to visualize the transformation"
                )
                st.session_state.selected_preprocessing_stage = selected_stage
        
        st.divider()
        
        # Section 2: Model & Analysis Settings
        st.subheader("🔬 Analysis Settings")
        
        # Visualization mode
        viz_mode = st.radio(
            "Visualization Type",
            options=["Plotly (Interactive)", "Matplotlib (Publication)", "Both"],
            help="Choose visualization library for charts"
        )
        
        # Threshold slider
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.01,
            help="Probability threshold for classifying as spam (lower = more sensitive)"
        )
        
        # Token analysis depth
        topn_tokens = st.slider(
            "Top-N Tokens",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Number of most frequent tokens to display"
        )
        
        st.divider()
        
        # Section 3: Model Parameters
        st.subheader("⚡ Model Parameters")
        
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data reserved for testing"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            help="For reproducible results"
        )
        
        st.divider()
        
        # Section 4: Model Info
        st.subheader("📈 Model Info")
        st.info(
            """
            **Logistic Regression**
            - Test Accuracy: 96.95%
            - Precision: 100% (spam)
            - Recall: 77.18% (spam)
            - F1 Score: 0.871
            
            **Dataset**: 5,574 SMS messages
            **Spam Ratio**: 13.4% (747 spam, 4,827 ham)
            """
        )
        
        # Store in session state
        st.session_state.viz_mode = viz_mode
        st.session_state.topn_tokens = topn_tokens
        st.session_state.test_size = test_size
        st.session_state.random_seed = random_seed
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🔍 Model Performance",
        "💬 Live Inference",
        "🔄 Preprocessing Pipeline" if st.session_state.get('csv_format') == 'preprocessing' else "ℹ️ About",
        "ℹ️ About" if st.session_state.get('csv_format') == 'preprocessing' else "📌 Placeholder"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 200, 234, 0.1) 0%, rgba(150, 180, 200, 0.1) 100%);
                    padding: 1.5rem; border-radius: 8px; border-left: 4px solid #66c8ea;">
        """, unsafe_allow_html=True)
        
        st.header("📊 Data Overview")
        
        # Load selected or default dataset
        df = load_dataset(selected_ds) if 'selected_ds' in locals() else load_dataset()
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", len(df))
            with col2:
                spam_count = (df['label'] == 1).sum()
                st.metric("Spam Messages", f"{spam_count} ({spam_count/len(df)*100:.1f}%)")
            with col3:
                ham_count = (df['label'] == 0).sum()
                st.metric("Ham Messages", f"{ham_count} ({ham_count/len(df)*100:.1f}%)")
            
            st.divider()
            
            # Class distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Class Distribution")
                
                # Use visualization mode from sidebar
                viz_mode = st.session_state.get('viz_mode', 'Plotly (Interactive)')
                
                if 'Matplotlib' in viz_mode and HAS_MATPLOTLIB:
                    fig_mpl = plot_class_distribution_mpl(df)
                    if fig_mpl:
                        st.pyplot(fig_mpl)
                else:
                    counts = df['label_text'].value_counts()
                    fig = px.bar(
                        x=counts.index, 
                        y=counts.values,
                        labels={'x': 'Class', 'y': 'Count'},
                        color=['#ff6b6b', '#51cf66'],
                        text=counts.values,
                        color_discrete_sequence=['#ff6b6b', '#51cf66']
                    )
                    fig.update_layout(showlegend=False, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Tokens by Class")
                topn = st.session_state.get('topn_tokens', 20)
                
                for label_text, col_idx in [('spam', 1), ('ham', 0)]:
                    st.write(f"**{label_text.upper()}** Messages:")
                    subset = df[df['label'] == col_idx]['text']
                    top = token_topn(subset, topn)
                    if top:
                        viz_mode = st.session_state.get('viz_mode', 'Plotly (Interactive)')
                        
                        if 'Matplotlib' in viz_mode and HAS_MATPLOTLIB:
                            fig_mpl = plot_tokens_mpl(df, label_text, col_idx, topn)
                            if fig_mpl:
                                st.pyplot(fig_mpl)
                        else:
                            toks, freqs = zip(*top)
                            # Use Plotly for better compatibility
                            fig = go.Figure(data=[
                                go.Bar(y=list(toks), x=list(freqs), orientation='h',
                                       marker=dict(color=list(freqs), colorscale='Viridis'))
                            ])
                            fig.update_layout(
                                title=f"Top {topn} Tokens",
                                xaxis_title="Frequency",
                                yaxis_title="Token",
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not load dataset.")
    
    # Tab 2: Model Performance
    with tab2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(234, 110, 102, 0.1) 0%, rgba(200, 150, 100, 0.1) 100%);
                    padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ea6e66;">
        """, unsafe_allow_html=True)
        
        st.header("🔍 Model Performance")
        
        model, vectorizer, error = load_resources()
        metrics = load_metrics()
        threshold_sweep = load_threshold_sweep()
        test_predictions = load_test_predictions()
        
        if error or metrics is None:
            st.error("Could not load model or metrics.")
            return
        
        # Metrics overview
        st.subheader("Overall Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
        with col2:
            st.metric("Test Precision", f"{metrics.get('test_precision', 0):.4f}")
        with col3:
            st.metric("Test Recall", f"{metrics.get('test_recall', 0):.4f}")
        with col4:
            st.metric("Test F1 Score", f"{metrics.get('test_f1', 0):.4f}")
        
        if 'test_roc_auc' in metrics:
            st.info(f"📈 **ROC-AUC Score**: {metrics['test_roc_auc']:.4f}")
        
        st.divider()
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        if 'test_confusion_matrix' in metrics:
            cm = np.array(metrics['test_confusion_matrix'])
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Ham', 'Predicted Spam'],
                y=['Actual Ham', 'Actual Spam'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                hovertemplate='%{y} / %{x}: %{z}<extra></extra>'
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Threshold sweep with real data
        st.subheader("⚙️ Threshold Sweep Analysis")
        st.caption("Adjust the threshold to balance Precision vs Recall")
        
        if threshold_sweep:
            # Convert to DataFrame
            sweep_df = pd.DataFrame(threshold_sweep)
            
            # Display as interactive table
            st.dataframe(sweep_df, use_container_width=True)
            
            # Plot threshold sweep curves
            col1, col2 = st.columns(2)
            
            with col1:
                # Precision-Recall vs Threshold
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=sweep_df['threshold'],
                    y=sweep_df['precision'],
                    name='Precision',
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig_pr.add_trace(go.Scatter(
                    x=sweep_df['threshold'],
                    y=sweep_df['recall'],
                    name='Recall',
                    mode='lines+markers',
                    line=dict(color='#ff7f0e', width=3)
                ))
                fig_pr.update_layout(
                    title="Precision & Recall vs Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="Score",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            
            with col2:
                # F1 Score vs Threshold
                fig_f1 = go.Figure()
                fig_f1.add_trace(go.Scatter(
                    x=sweep_df['threshold'],
                    y=sweep_df['f1'],
                    name='F1 Score',
                    fill='tozeroy',
                    mode='lines+markers',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=8)
                ))
                
                # Mark the current threshold
                current_threshold = st.session_state.get('threshold', 0.5)
                fig_f1.add_vline(
                    x=current_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Current: {current_threshold:.2f}",
                    annotation_position="top right"
                )
                
                fig_f1.update_layout(
                    title="F1 Score vs Threshold",
                    xaxis_title="Threshold",
                    yaxis_title="F1 Score",
                    height=400
                )
                st.plotly_chart(fig_f1, use_container_width=True)
        
        st.divider()
        
        # ROC Curve (if we have predictions)
        st.subheader("ROC Curve")
        if test_predictions:
            from sklearn.metrics import roc_curve, auc
            
            y_true = np.array(test_predictions['y_true'])
            y_proba = np.array(test_predictions['y_proba'])
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                mode='lines',
                line=dict(color='#d62728', width=3)
            ))
            
            # Add diagonal reference line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig_roc.update_layout(
                title=f'ROC Curve (AUC = {roc_auc:.4f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                hovermode='closest'
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve data not available. Generate with: python train.py")
    
    
    # Tab 3: Live Inference
    with tab3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 234, 200, 0.1) 0%, rgba(100, 200, 180, 0.1) 100%);
                    padding: 1.5rem; border-radius: 8px; border-left: 4px solid #66eac8;">
        """, unsafe_allow_html=True)
        
        st.header("💬 Live Inference")
        st.caption("Enter or select a message to classify in real-time")
        
        model, vectorizer, error = load_resources()
        if error:
            st.error(f"Failed to load model: {error}")
            return
        
        # Quick examples
        st.subheader("Quick Examples")
        col1, col2, col3 = st.columns(3)
        
        ex_spam1 = "FREE cash NOW! Click here to claim your prize"
        ex_spam2 = "Urgent: You've won a lottery! Call +1-800-SPAM-1234"
        ex_ham = "Hi, let's meet tomorrow at 3pm for coffee"
        
        with col1:
            if st.button("🚨 Spam Example 1"):
                st.session_state.input_text = ex_spam1
        with col2:
            if st.button("🚨 Spam Example 2"):
                st.session_state.input_text = ex_spam2
        with col3:
            if st.button("✅ Ham Example"):
                st.session_state.input_text = ex_ham
        
        # Text input
        if "input_text" not in st.session_state:
            st.session_state.input_text = ""
        
        user_text = st.text_area(
            "Enter a message to classify:",
            value=st.session_state.input_text,
            height=100,
            placeholder="Type your email or SMS here..."
        )
        
        if st.button("Predict", type="primary", use_container_width=True):
            if not user_text.strip():
                st.warning("Please enter a message first!")
            else:
                # Normalize and predict
                cleaned_text = normalize_text(user_text)
                
                with st.expander("📝 Show normalized text", expanded=False):
                    st.code(cleaned_text, language="text")
                
                # Get prediction
                X = vectorizer.transform([cleaned_text])
                probability = float(model.predict_proba(X)[0][1])
                prediction = 1 if probability >= threshold else 0
                
                # Display result
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 SPAM DETECTED")
                    else:
                        st.success("✅ LEGITIMATE (HAM)")
                
                with col2:
                    confidence = probability if prediction == 1 else (1 - probability)
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Probability bar with threshold marker (Plotly version)
                fig = go.Figure()
                color = "#d62728" if prediction == 1 else "#1f77b4"
                fig.add_trace(go.Bar(
                    x=[probability],
                    orientation='h',
                    marker=dict(color=color),
                    name='Spam Probability',
                    text=f"{probability:.4f}",
                    textposition='outside'
                ))
                fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                            annotation_text=f"Threshold: {threshold:.2f}",
                            annotation_position="top")
                fig.update_layout(
                    xaxis_title="Spam Probability",
                    showlegend=False,
                    height=200,
                    xaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional stats
                st.divider()
                st.subheader("Detailed Results")
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("Prediction", "SPAM" if prediction == 1 else "HAM")
                with result_col2:
                    st.metric("Spam Probability", f"{probability:.4f}")
                with result_col3:
                    st.metric("Decision Threshold", f"{threshold:.2f}")
    
    # Tab 4: Preprocessing Pipeline (if available)
    if st.session_state.get('csv_format') == 'preprocessing':
        with tab4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(234, 180, 102, 0.1) 0%, rgba(220, 180, 150, 0.1) 100%);
                        padding: 1.5rem; border-radius: 8px; border-left: 4px solid #eab466;">
            """, unsafe_allow_html=True)
            
            st.header("🔄 Preprocessing Pipeline Visualization")
            st.caption("Visualize how text is transformed through different preprocessing stages")
            
            full_df = st.session_state.get('full_df')
            stages_dict = st.session_state.get('preprocessing_stages', {})
            selected_stage = st.session_state.get('selected_preprocessing_stage', 'All')
            
            if full_df is not None and stages_dict:
                # Show message sample
                st.subheader("Select a Message to Visualize")
                
                # Get ham and spam examples
                ham_indices = full_df[full_df['label'] == 'ham'].index.tolist()
                spam_indices = full_df[full_df['label'] == 'spam'].index.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📩 Show HAM Example", use_container_width=True):
                        st.session_state.selected_idx = ham_indices[0] if ham_indices else 0
                
                with col2:
                    if st.button("🚨 Show SPAM Example", use_container_width=True):
                        st.session_state.selected_idx = spam_indices[0] if spam_indices else 0
                
                if 'selected_idx' not in st.session_state:
                    st.session_state.selected_idx = 0
                
                selected_idx = st.session_state.selected_idx
                
                # Show preprocessing pipeline
                st.divider()
                st.subheader(f"Preprocessing Stages (Row {selected_idx})")
                
                # Get stage columns
                stage_order = [
                    'text_raw',
                    'text_lower',
                    'text_contacts_masked',
                    'text_numbers',
                    'text_stripped',
                    'text_whitespace',
                    'text_stopwords_removed'
                ]
                
                # Filter stages based on selection
                if selected_stage == 'All':
                    stages_to_show = stage_order
                else:
                    stages_to_show = [selected_stage]
                
                # Create comparison view
                for stage in stages_to_show:
                    stage_col = stages_dict.get(stage)
                    if stage_col and stage_col in full_df.columns:
                        stage_name = stage.replace('_', ' ').title()
                        text_value = str(full_df.loc[selected_idx, stage_col])
                        
                        # Create expandable section for each stage
                        with st.expander(f"📝 {stage_name}", expanded=(stage == 'text_raw')):
                            st.write(text_value)
                            
                            # Show character count and token count
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Characters", len(text_value))
                            with col2:
                                st.metric("Tokens", len(text_value.split()))
                            with col3:
                                st.metric("Avg Token Length", f"{len(text_value)/max(1, len(text_value.split())):.1f}")
                
                # Show transformation details
                st.divider()
                st.subheader("Transformation Summary")
                
                summary_data = []
                for stage in stage_order:
                    stage_col = stages_dict.get(stage)
                    if stage_col and stage_col in full_df.columns:
                        text_val = str(full_df.loc[selected_idx, stage_col])
                        summary_data.append({
                            'Stage': stage.replace('_', ' ').title(),
                            'Characters': len(text_val),
                            'Tokens': len(text_val.split()),
                            'Avg Length': f"{len(text_val)/max(1, len(text_val.split())):.1f}"
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                # Show token changes across stages
                st.divider()
                st.subheader("Token Changes Across Stages")
                
                # Extract tokens at each stage
                stage_tokens = {}
                for stage in stage_order:
                    stage_col = stages_dict.get(stage)
                    if stage_col and stage_col in full_df.columns:
                        text_val = str(full_df.loc[selected_idx, stage_col])
                        tokens = text_val.split()
                        stage_tokens[stage.replace('_', ' ').title()] = set(tokens)
                
                # Calculate token losses
                if stage_tokens:
                    stages_list = list(stage_tokens.keys())
                    if len(stages_list) > 1:
                        token_changes = []
                        for i in range(len(stages_list) - 1):
                            current = stage_tokens[stages_list[i]]
                            next_stage = stage_tokens[stages_list[i + 1]]
                            removed = current - next_stage
                            added = next_stage - current
                            
                            token_changes.append({
                                'From': stages_list[i],
                                'To': stages_list[i + 1],
                                'Removed': ', '.join(sorted(list(removed)[:5])) + ('...' if len(removed) > 5 else ''),
                                'Tokens Removed': len(removed),
                                'Tokens Added': len(added)
                            })
                        
                        if token_changes:
                            changes_df = pd.DataFrame(token_changes)
                            st.dataframe(changes_df, use_container_width=True)
    
    # Tab 5 (or Tab 4): About
    about_tab = tab5 if st.session_state.get('csv_format') == 'preprocessing' else tab4
    
    with about_tab:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(180, 160, 234, 0.1) 0%, rgba(200, 180, 220, 0.1) 100%);
                    padding: 1.5rem; border-radius: 8px; border-left: 4px solid #b4a0ea;">
        """, unsafe_allow_html=True)
        
        st.header("ℹ️ About This Project")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            This is an advanced spam email classifier built with machine learning and OpenSpec for spec-driven development.
            It demonstrates a complete ML pipeline with professional-grade visualizations and inference capabilities.
            
            ### 📚 Dataset
            - **Source**: [Packt's Hands-On AI for Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)
            - **Size**: 5,574 SMS messages
            - **Classes**: Spam (13.4%) vs. Ham (86.6%)
            - **Features**: TF-IDF vectorization with unigrams and bigrams
            
            ### 🛠 Technologies
            - **ML**: Scikit-learn (Logistic Regression)
            - **Data**: Pandas, NumPy
            - **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
            - **Deployment**: Streamlit Cloud
            
            ### 📊 Model Metrics
            - **Test Accuracy**: 96.95%
            - **Test F1 Score**: 0.871
            - **Precision**: 100% (on spam)
            - **Recall**: 77.18% (on spam)
            
            ### 🚀 Features
            - Real-time message classification
            - Token frequency analysis
            - Confusion matrix visualization
            - Threshold sweep analysis
            - Text normalization display
            - Quick example buttons for testing
            - Decision probability visualization
            """)
        
        with col2:
            st.markdown("""
            ### 📖 Resources
            
            **Documentation:**
            - [OpenSpec Tutorial](https://www.youtube.com/watch?v=ANjiJQQIBo0)
            - [Packt YouTube](https://www.youtube.com/playlist?list=PLYlM4-ln5HcCoM_TcLKGL5NcOpNVJ3g7c)
            
            **Project Links:**
            - [My GitHub](https://github.com/Brain0927/HW3_Fulong_5114056035)
            - [Reference Project](https://github.com/huanchen1107/2025ML-spamEmail)
            
            ### 🎓 Learning Path
            ✅ Data preprocessing
            ✅ Feature engineering (TF-IDF)
            ✅ Model training
            ✅ Evaluation metrics
            ✅ OpenSpec workflow
            ✅ Streamlit deployment
            ✅ Advanced visualizations
            """)
        
        st.divider()
        
        # Footer stats
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", "5,574")
        with col2:
            st.metric("Spam Count", "747 (13.4%)")
        with col3:
            st.metric("Test Accuracy", "96.95%")
        with col4:
            st.metric("F1 Score", "0.871")


if __name__ == "__main__":
    main()
