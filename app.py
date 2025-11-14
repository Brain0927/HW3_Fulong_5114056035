"""
Streamlit app for spam email classification.
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_trainer import load_model, load_vectorizer
from data_loader import get_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def main():
    st.set_page_config(page_title="Spam Email Classifier", layout="wide")
    st.title("📧 Spam Email Classification")
    st.markdown("A machine learning-powered spam email classifier using Logistic Regression")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Classifier", "Model Performance", "About"])
    
    if page == "Classifier":
        show_classifier()
    elif page == "Model Performance":
        show_performance()
    else:
        show_about()


def show_classifier():
    st.header("Spam Classifier")
    
    # Load model
    model, vectorizer, error = load_resources()
    if error:
        st.error(f"Failed to load model: {error}")
        st.info("Please run `python train.py` first to train the model.")
        return
    
    # Input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_message = st.text_area(
            "Enter a message to classify:",
            height=100,
            placeholder="Type your email or SMS here..."
        )
    
    with col2:
        st.markdown("### Example Messages")
        st.markdown("**Spam example:**\n*'FREE cash NOW! Click here to claim'*")
        st.markdown("**Ham example:**\n*'Hi, let's meet tomorrow at 3pm'*")
    
    if st.button("Classify", type="primary"):
        if not user_message.strip():
            st.warning("Please enter a message first!")
        else:
            # Preprocess and predict
            X = vectorizer.transform([user_message])
            prediction = model.predict(X)[0]
            
            # Get probability if available
            try:
                probability = model.predict_proba(X)[0][1]
            except:
                probability = None
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 SPAM DETECTED")
                    label_text = "SPAM"
                else:
                    st.success("✅ LEGITIMATE")
                    label_text = "HAM"
            
            with col2:
                if probability is not None:
                    confidence = probability if prediction == 1 else (1 - probability)
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Draw gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Spam Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)


def show_performance():
    st.header("Model Performance")
    
    # Load metrics
    metrics = load_metrics()
    if metrics is None:
        st.error("Metrics file not found. Please run `python train.py` first.")
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Validation Accuracy", f"{metrics.get('val_accuracy', 0):.4f}")
    with col2:
        st.metric("Validation Precision", f"{metrics.get('val_precision', 0):.4f}")
    with col3:
        st.metric("Validation Recall", f"{metrics.get('val_recall', 0):.4f}")
    with col4:
        st.metric("Validation F1 Score", f"{metrics.get('val_f1', 0):.4f}")
    
    st.divider()
    
    # Test metrics
    if 'test_accuracy' in metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
        with col2:
            st.metric("Test Precision", f"{metrics.get('test_precision', 0):.4f}")
        with col3:
            st.metric("Test Recall", f"{metrics.get('test_recall', 0):.4f}")
        with col4:
            st.metric("Test F1 Score", f"{metrics.get('test_f1', 0):.4f}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        if 'test_accuracy' in metrics:
            acc_data = {
                'Set': ['Validation', 'Test'],
                'Accuracy': [metrics['val_accuracy'], metrics['test_accuracy']]
            }
        else:
            acc_data = {
                'Set': ['Validation'],
                'Accuracy': [metrics['val_accuracy']]
            }
        
        fig = px.bar(acc_data, x='Set', y='Accuracy', title='Accuracy Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F1 comparison
        if 'test_f1' in metrics:
            f1_data = {
                'Set': ['Validation', 'Test'],
                'F1': [metrics['val_f1'], metrics['test_f1']]
            }
        else:
            f1_data = {
                'Set': ['Validation'],
                'F1': [metrics['val_f1']]
            }
        
        fig = px.bar(f1_data, x='Set', y='F1', title='F1 Score Comparison')
        st.plotly_chart(fig, use_container_width=True)


def show_about():
    st.header("About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This spam email classifier is built using machine learning and OpenSpec for spec-driven development.
    
    ### 📚 Dataset
    - **Source**: [Packt's Hands-On Artificial Intelligence for Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)
    - **Size**: ~5,000 SMS messages
    - **Classes**: Spam vs. Ham (legitimate)
    
    ### 🛠 Technologies
    - **Python 3.11+**
    - **Scikit-learn**: Model training and evaluation
    - **Pandas**: Data manipulation
    - **Streamlit**: Interactive UI
    - **Plotly**: Visualization
    
    ### 🚀 Development Workflow
    This project uses **OpenSpec** for specification-driven development:
    1. Define specifications in `openspec/specs/`
    2. Create change proposals in `openspec/changes/`
    3. Implement and validate against specs
    4. Deploy and archive completed changes
    
    ### 📖 Resources
    - [OpenSpec Tutorial](https://www.youtube.com/watch?v=ANjiJQQIBo0)
    - [Packt YouTube Playlist](https://www.youtube.com/playlist?list=PLYlM4-ln5HcCoM_TcLKGL5NcOpNVJ3g7c)
    - [Project Repository](https://github.com/huanchen1107/2025ML-spamEmail)
    
    ### 📝 Features
    - Real-time spam classification
    - Model performance metrics and visualizations
    - Training pipeline with multiple model options
    - Reproducible results with fixed random seeds
    """)


if __name__ == "__main__":
    main()
