# 🧾 Email Spam Classification — HW3 Project

A complete machine learning pipeline for spam email classification using OpenSpec for specification-driven development.

## 📋 Project Overview

This project implements an end-to-end spam email classification system:
- **Data Pipeline**: Load, preprocess, and vectorize email text data
- **Model Training**: Train Logistic Regression / Naive Bayes / SVM models
- **Evaluation**: Comprehensive metrics and visualizations
- **Interactive UI**: Streamlit-based web application for real-time classification

**Stack**: Python 3.11+, scikit-learn, pandas, Streamlit, Plotly

## 📚 References

- **Dataset**: [Packt's Hands-On AI for Cybersecurity — Chapter 3](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv)
- **Tutorial**: [YouTube Playlist](https://www.youtube.com/watch?v=FeCCYFK0TJ8&list=PLYlM4-ln5HcCoM_TcLKGL5NcOpNVJ3g7c)
- **OpenSpec Workflow**: [Tutorial Video](https://www.youtube.com/watch?v=ANjiJQQIBo0)
- **GitHub Repo**: [2025ML-spamEmail](https://github.com/huanchen1107/2025ML-spamEmail)
- **Demo Site**: [Streamlit App](https://2025spamemail.streamlit.app/)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Brain0927/HW3_Fulong_5114056035.git
cd HW3_Fulong_5114056035
```

### 2. Set Up Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model
```bash
# Train baseline model (Logistic Regression)
python train.py

# Or specify model type
python train.py --model logistic_regression
python train.py --model naive_bayes
python train.py --model svm
```

This will:
- Download the dataset automatically
- Preprocess text data (lowercase, tokenization)
- Extract TF-IDF features
- Split into train/val/test sets
- Train the model
- Evaluate and save metrics
- Save model artifacts to `models/`

### 4. Run Streamlit App
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### 5. Test Classification
- Go to the **Classifier** tab
- Enter a message
- Click "Classify"
- View spam probability and confidence

## 📁 Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train.py                           # Training pipeline script
├── app.py                             # Streamlit application
├── src/
│   ├── data_loader.py                 # Data loading & preprocessing
│   ├── model_trainer.py               # Model training & evaluation
│   └── __init__.py
├── scripts/
│   ├── predict_spam.py                # CLI: Single/batch prediction
│   ├── visualize_spam.py              # CLI: Visualization toolkit
│   ├── preprocess_emails.py           # CLI: Preprocessing pipeline
│   └── train_spam_classifier.py       # CLI: Training with parameters
├── data/
│   ├── sms_spam_no_header.csv         # Dataset (auto-downloaded)
│   ├── sms_spam_clean.csv             # Cleaned 2-column format (label, message)
│   └── sms_spam_preprocessing.csv     # 9-column preprocessing pipeline (NEW)
├── models/
│   ├── logistic_regression.pkl        # Trained model (joblib)
│   ├── vectorizer.pkl                 # TF-IDF vectorizer (joblib)
│   ├── label_mapping.json             # Label mappings
│   └── metrics_logistic_regression.json  # Performance metrics
├── docs/
│   └── PREPROCESSING.md               # Preprocessing documentation
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # EDA notebook
│   └── 02_model_comparison.ipynb      # Model comparison
└── openspec/                          # OpenSpec workflow files
    ├── project.md                     # Project context
    ├── AGENTS.md                      # OpenSpec instructions
    ├── specs/
    │   └── ml/
    │       └── spec.md                # ML capability specification
    └── changes/
        ├── add-spam-classification/   # Change proposal
        │   ├── proposal.md
        │   ├── tasks.md
        │   └── specs/
        │       └── ml/
        │           └── spec.md
        └── archive/
```

## 🧪 Model Evaluation

### Baseline Performance (Logistic Regression)

After training, check `models/metrics_logistic_regression.json` for:
- **Validation Accuracy**: ~98%
- **Validation F1 Score**: ~0.95
- **Validation Precision/Recall**: High performance on both spam and ham

### Available Models

1. **Logistic Regression** (default, fastest)
   - Interpretable coefficients
   - Fast training & inference
   - Best for baseline

2. **Naive Bayes**
   - Probabilistic approach
   - Handles sparse features well
   - Faster inference than LR

3. **SVM (LinearSVC)**
   - Maximum margin classifier
   - Good generalization
   - Slower training

## 📊 Features

### Data Preprocessing
- **Tokenization**: Lowercase normalization
- **Vectorization**: TF-IDF (max 5,000 features)
- **N-grams**: Unigrams and bigrams (1-2)
- **Stop words**: English stop words removed

### Model Pipeline
- **Train/Val/Test Split**: 70% / 10% / 20%
- **Stratified Split**: Maintains class balance
- **Feature Engineering**: TF-IDF vectorization with n-grams

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC (for models supporting `predict_proba`)
- Classification Report

### Streamlit App Features
**Multi-Format CSV Support:**
- ✅ Simple 2-column CSV format (label, text)
- ✅ **NEW**: 9-column preprocessing pipeline CSV
  - Visualize 7 preprocessing stages in real-time
  - Stage-by-stage text transformation display
  - Token changes tracking across each stage
  - Character & token count metrics
  - Interactive stage selector in sidebar

**Main Dashboard Tabs:**
1. **📊 Data Overview**
   - Class distribution visualization (Plotly/Matplotlib)
   - Top-N token frequency analysis
   - Dataset statistics (total messages, spam ratio)

2. **🔍 Model Performance**
   - Metrics overview (accuracy, precision, recall, F1)
   - Confusion matrix heatmap
   - Threshold sweep analysis
   - ROC-AUC score display

3. **💬 Live Inference**
   - Real-time message classification
   - Spam probability gauge with threshold indicator
   - Text normalization display
   - Quick example buttons (spam/ham samples)
   - Detailed prediction results

4. **🔄 Preprocessing Pipeline** (NEW - for 9-column CSV)
   - Stage-by-stage transformation visualization
   - Text progression through preprocessing pipeline
   - Token changes at each stage (removals/additions)
   - Character and token statistics per stage
   - Ham/spam example quick selectors

5. **ℹ️ About**
   - Project overview and technologies
   - Dataset information
   - Model metrics summary
   - Learning resources and references

**Sidebar Configuration:**
- Dataset selector (auto-discovers CSV files)
- Preprocessing stage selector (when applicable)
- Visualization mode: Plotly / Matplotlib / Both
- Decision threshold slider
- Top-N tokens selector
- Model parameters (test size, random seed)
- Model info panel
- ROC-AUC (for models supporting `predict_proba`)
- Classification Report

## 🎯 Multi-Format CSV Support

The Streamlit app automatically detects and handles different CSV formats:

### Simple Format (2 columns)
**File**: `data/sms_spam_clean.csv`
```csv
label,message
ham,Go until jurong point crazy available...
spam,Free entry to WIN cash...
```

**Usage in App:**
- Select from dropdown: `data/sms_spam_clean.csv`
- Shows class distribution and token analysis
- Standard model evaluation tabs

### Preprocessing Pipeline Format (8 columns)
**File**: `data/sms_spam_preprocessing.csv` (NEW)
```csv
label,text_raw,text_lower,text_contacts_masked,text_numbers,text_stripped,text_whitespace,text_stopwords_removed
ham,Go until jurong...,go until jurong...,go until jurong...,go until jurong...,go until jurong...,go until jurong...,go until jurong...
```

**Pipeline Stages:**
1. `text_raw` — Original message
2. `text_lower` — Lowercase conversion
3. `text_contacts_masked` — Email/phone masking
4. `text_numbers` — Number replacement (<NUM>)
5. `text_stripped` — Punctuation removal
6. `text_whitespace` — Whitespace normalization
7. `text_stopwords_removed` — Stop word filtering

**Usage in App:**
1. Open Streamlit app: `streamlit run app.py`
2. Select dataset: `data/sms_spam_preprocessing.csv`
3. App detects preprocessing pipeline format
4. New tab appears: **"🔄 Preprocessing Pipeline"**
5. Use sidebar selector to choose stage
6. Visualize text transformation:
   - Stage-by-stage comparison
   - Token count metrics
   - Changes tracking (removed/added tokens)
   - Ham/spam examples

**Example Transformation:**
```
text_raw:               "Free entry in 2 a wkly comp"
↓ text_lower:          "free entry in 2 a wkly comp"
↓ text_numbers:        "free entry in <NUM> a wkly comp"
↓ text_stripped:       "free entry in <NUM> a wkly comp"
↓ text_whitespace:     "free entry in <NUM> a wkly comp"
↓ text_stopwords_removed: "free entry <NUM> wkly comp"
```

## 🔧 Advanced Usage

### CLI Tools

#### 1. Single Message Prediction
```bash
python scripts/predict_spam.py --text "Free cash now! Click here"
```

Output:
```
============================================================
Original:   Free cash now! Click here
Normalized: free cash now click here
============================================================
Prediction: SPAM
Probability (spam): 0.7821
Threshold: 0.5
============================================================
```

#### 2. Batch Prediction from CSV
```bash
# Predict on new dataset
python scripts/predict_spam.py \
  --input data/sms_spam_no_header.csv \
  --text-col "Message Column" \
  --output predictions.csv \
  --threshold 0.5
```

Output saves CSV with columns:
- `text_normalized`: Preprocessed message
- `spam_probability`: Raw prediction probability
- `label`: Final prediction (spam/ham)

#### 3. Visualization Tools
```bash
# Class distribution
python scripts/visualize_spam.py \
  --input data/sms_spam_no_header.csv \
  --label-col "ham" \
  --text-col "message text" \
  --dist --output-dir outputs

# Token frequency analysis
python scripts/visualize_spam.py \
  --input data/sms_spam_no_header.csv \
  --label-col "ham" \
  --text-col "message text" \
  --tokens --tokens-for spam --topn 30

# ROC/Precision-Recall curves
python scripts/visualize_spam.py \
  --roc-data predictions.csv \
  --labels true_label \
  --probs spam_probability \
  --roc --threshold-sweep
```

### Python API

#### Training with Custom Parameters
```python
from src.model_trainer import train_model, evaluate_model, prepare_features, split_data
from src.data_loader import get_data

# Load and preprocess
messages, labels = get_data()
X, vectorizer = prepare_features(messages, max_features=3000)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, labels)

# Train with custom max_iter
model = train_model('svm', X_train, y_train, max_iter=3000)

# Evaluate
metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
print(metrics['test_f1'])
```

#### Batch Classification
```python
from src.model_trainer import load_model, load_vectorizer

model = load_model('logistic_regression')
vectorizer = load_vectorizer()

messages = ["FREE cash NOW!", "Meeting tomorrow at 3pm"]
X = vectorizer.transform(messages)
predictions = model.predict(X)
probabilities = model.predict_proba(X)

for msg, pred, prob in zip(messages, predictions, probabilities):
    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[1] if pred == 1 else prob[0]
    print(f"{label}: {msg} ({confidence:.2%})")
```

## 📖 Documentation

### Detailed Documentation

- **[PREPROCESSING.md](docs/PREPROCESSING.md)** — Complete preprocessing pipeline with step-by-step examples
  - Text normalization (lowercase, punctuation removal)
  - Contact masking (URLs, emails, phones → tokens)
  - Number handling and whitespace normalization
  - TF-IDF feature extraction
  - Before/after examples

### OpenSpec Workflow

This project follows the OpenSpec specification-driven development process:

### Files
- `openspec/project.md` — Project context and conventions
- `openspec/specs/ml/spec.md` — ML capability requirements
- `openspec/changes/add-spam-classification/` — Implementation proposal and tasks

### Workflow
1. **Read specifications**: `openspec/specs/ml/spec.md`
2. **Check proposal**: `openspec/changes/add-spam-classification/proposal.md`
3. **Review tasks**: `openspec/changes/add-spam-classification/tasks.md`
4. **Implement**: Follow tasks in order
5. **Validate**: `openspec validate add-spam-classification --strict`

### Commands
```bash
# List active changes and specs
openspec list
openspec list --specs

# Show details
openspec show add-spam-classification
openspec show ml --type spec

# Validate
openspec validate add-spam-classification --strict
```

## 📦 Deployment (Streamlit Cloud)

### Steps
1. Push code to GitHub repo `2025ML-spamEmail`
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect GitHub repo
4. Deploy with `app.py` as main file
5. Set secrets if needed (none required for this project)

### Environment Variables
None required for the default configuration. Dataset is downloaded automatically.

## 🐛 Troubleshooting

### Model file not found
```
Error: Model file not found: models/logistic_regression.pkl
```
**Solution**: Run `python train.py` first to train and save the model.

### Dataset download fails
```
Error: Failed to download dataset
```
**Solution**: Download manually from [Packt GitHub](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity) and save to `data/sms_spam_no_header.csv`.

### Streamlit cache issues
```bash
streamlit cache clear
streamlit run app.py
```

## 📝 Assignment Checklist

- [x] GitHub repository created and public
- [x] OpenSpec workflow files (`project.md`, `proposal.md`, `tasks.md`, `spec.md`)
- [x] ML pipeline implemented (preprocessing, training, evaluation)
- [x] Streamlit demo deployed
- [x] requirements.txt and README completed
- [x] Model saved and metrics tracked

## 🎓 Learning Objectives

By completing this project, you'll learn:
- ✅ End-to-end ML pipeline development
- ✅ Text preprocessing and TF-IDF vectorization
- ✅ Model training and evaluation best practices
- ✅ OpenSpec specification-driven development
- ✅ Building interactive ML applications with Streamlit
- ✅ Model deployment and monitoring

## 🚀 Phase 4+ Enhancements

This project includes professional-grade enhancements beyond basic requirements:

### CLI Toolchain
- **predict_spam.py** — Single message or batch CSV predictions
- **visualize_spam.py** — Comprehensive visualization suite
- **label_mapping.json** — Explicit class mapping for production use

### Architecture Improvements
- **Joblib serialization** — Replaced pickle for better model artifact handling
- **Column inference** — Automatic label/text column detection in Streamlit
- **Dataset selector** — Dynamic CSV loading in Streamlit app
- **Production-ready** — CLI scripts suitable for deployment pipelines

### Documentation
- **PREPROCESSING.md** — Detailed pipeline documentation with examples
- **Enhanced README** — Complete usage guide with CLI examples

## 📄 License

This project uses the [Packt repository](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity) dataset.

## 🙋 Support

For questions or issues:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [OpenSpec workflow](#-openspec-workflow)
3. Consult the tutorial videos in [References](#-references)

---

**Created**: 2025-11-13 | **Assignment**: HW3 — Email Spam Classification (10% grade weight)
