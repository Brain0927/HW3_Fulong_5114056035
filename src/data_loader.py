"""
Data loading and preprocessing module for spam email classification.
"""
import os
import pandas as pd
import urllib.request
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset URL from Packt GitHub
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/"
    "master/Chapter03/datasets/sms_spam_no_header.csv"
)

DATA_DIR = "data"
DATASET_PATH = os.path.join(DATA_DIR, "sms_spam_no_header.csv")


def download_dataset(url: str = DATASET_URL, output_path: str = DATASET_PATH) -> None:
    """Download dataset from GitHub if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(output_path):
        logger.info(f"Dataset already exists at {output_path}")
        return
    
    logger.info(f"Downloading dataset from {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Dataset downloaded to {output_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def load_raw_data(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """Load raw dataset."""
    if not os.path.exists(filepath):
        logger.warning(f"File {filepath} not found. Downloading...")
        download_dataset()
    
    # Dataset format: "label","message" (CSV with quotes)
    df = pd.read_csv(filepath, header=None, encoding='latin-1', sep=',', quotechar='"')
    
    # Handle case where there are only 2 columns
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ['label', 'message']
    else:
        raise ValueError(f"Expected at least 2 columns, got {df.shape[1]}")
    
    # Remove quotes from labels if present
    df['label'] = df['label'].str.strip('"').str.lower()
    
    logger.info(f"Loaded {len(df)} records from {filepath}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Preprocess messages and labels.
    
    Args:
        df: DataFrame with 'message' and 'label' columns
        
    Returns:
        Tuple of (messages, labels) with processed data
    """
    # Normalize labels
    df = df.copy()
    df['label_encoded'] = (df['label'] == 'spam').astype(int)
    
    # Basic text preprocessing
    df['message'] = df['message'].str.lower().str.strip()
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    logger.info(f"Spam ratio: {df['label_encoded'].mean():.2%}")
    
    return df['message'], df['label_encoded']


def get_data(download: bool = True) -> Tuple[pd.Series, pd.Series]:
    """Main entry point to get preprocessed data."""
    if download:
        download_dataset()
    
    raw_df = load_raw_data()
    messages, labels = preprocess_data(raw_df)
    
    return messages, labels
