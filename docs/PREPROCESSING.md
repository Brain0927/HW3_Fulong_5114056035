# Preprocessing Report: SMS/Email Spam Dataset

## Overview
This report documents the text preprocessing pipeline used to prepare raw SMS/email data for spam classification. The pipeline is deterministic and idempotent: the same input produces the same output.

- **Dataset**: `data/sms_spam_no_header.csv` (5,574 messages)
- **Raw Format**: Headerless 2-column CSV
  - Column 0: Label (`spam` or `ham`)
  - Column 1: Message text
- **Output**: Cleaned dataset used for training and inference
- **Reference**: Adapted from [Packt: Hands-On AI for Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total Messages | 5,574 |
| Spam Messages | 747 (13.4%) |
| Ham Messages | 4,827 (86.6%) |
| Avg Message Length | 92.5 characters |

## Preprocessing Pipeline

The pipeline performs 5 sequential transformations on raw text:

### Step 1: Lowercase Normalization
- **Purpose**: Normalize case sensitivity
- **Operation**: Convert all characters to lowercase
- **Before**: `"Ok lar... Joking wif u oni..."`
- **After**: `"ok lar... joking wif u oni..."`
- **Rationale**: Reduces feature dimensionality (e.g., `Free` and `free` treated as same token)

### Step 2: Mask Contacts (URLs, Emails, Phones)
- **Purpose**: Replace personal/contact identifiers with placeholder tokens
- **Tokens**:
  - `<URL>` — URLs (https://example.com, www.example.com)
  - `<EMAIL>` — Email addresses (user@example.com)
  - `<PHONE>` — Phone numbers (+1 415-555-1212)
- **Examples**:
  - `"Contact me at test@example.com"` → `"contact me at <EMAIL>"`
  - `"Visit https://example.com"` → `"visit <URL>"`
  - `"Call +1-800-555-1234"` → `"call <PHONE>"`
- **Rationale**: 
  - Generalizes contact patterns (many spam emails have URLs/emails)
  - Reduces vocabulary explosion from unique addresses
  - Preserves semantic signal (presence of contact != specific contact)

### Step 3: Numbers Handling
- **Purpose**: Replace digit sequences with `<NUM>` token
- **Operation**: Match one or more consecutive digits, replace with `<NUM>`
- **Before**: `"Text FA to 87121 to receive entry"`
- **After**: `"text fa to <NUM> to receive entry"`
- **Rationale**:
  - Spam often includes codes, dates, amounts
  - Generalizing specific numbers reduces noise
  - `<NUM>` token is informative for spam detection

### Step 4: Strip Punctuation
- **Purpose**: Remove noise characters
- **Operation**: Remove all punctuation; keep alphanumerics, spaces, and special tokens (`<>`)
- **Before**: `"crazy.. available only in bugis"`
- **After**: `"crazy   available only in bugis"`
- **Rationale**: Punctuation often adds no semantic value and increases dimensionality

### Step 5: Whitespace Normalization
- **Purpose**: Clean up spacing
- **Operation**: Collapse multiple spaces into single space; trim leading/trailing whitespace
- **Before**: `"go until jurong point  crazy   available"`
- **After**: `"go until jurong point crazy available"`
- **Rationale**: Ensures consistent tokenization in TF-IDF vectorizer

## Example: Before & After

### Raw Message (Spam)
```
"spam","Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
```

### After Preprocessing
```
"free entry in <NUM> a wkly comp to win fa cup final tkts <NUM>st may <NUM> text fa to <NUM> to receive entry question std txt rate t c s apply <NUM>over<NUM> s"
```

### Raw Message (Ham)
```
"ham","Ok lar... Joking wif u oni..."
```

### After Preprocessing
```
"ok lar joking wif u oni"
```

## Feature Extraction (TF-IDF)

After text preprocessing, the cleaned messages are converted to numeric feature vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency):

- **Vectorizer**: `sklearn.feature_extraction.text.TfidfVectorizer`
- **Parameters**:
  - `max_features=5000`: Keep top 5,000 most frequent tokens
  - `ngram_range=(1, 2)`: Use unigrams (single tokens) and bigrams (2-token sequences)
  - `stop_words='english'`: Remove common English words (e.g., the, and, a)
- **Output**: Sparse matrix of shape `[n_samples, 5000]`

### Why TF-IDF?
- **TF (Term Frequency)**: How often a token appears in a message
- **IDF (Inverse Document Frequency)**: How unique a token is across all messages
- **Combination**: Tokens that appear frequently in individual messages but rarely across the dataset get higher weights
- **Benefit**: Common spam indicators (e.g., "free", "win", `<NUM>`) naturally get boosted

### Token Statistics

| Token Type | Count | % of Dataset |
|------------|-------|--------------|
| `<URL>` | 187 | 3.4% |
| `<NUM>` | 2,194 | 39.4% |
| `<EMAIL>` | 28 | 0.5% |
| `<PHONE>` | 156 | 2.8% |

## Model Training

- **Algorithm**: Logistic Regression
- **Train/Test Split**: 70% train (3,902 samples), 30% test (1,672 samples)
- **Regularization**: L2 (Ridge), C=1.0 (default)
- **Stratified Split**: Preserves class proportions in train/test

## Model Performance

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 97.8% | 96.95% |
| Precision (Spam) | 98.5% | 100% |
| Recall (Spam) | 98.1% | 77.18% |
| F1 Score (Spam) | 98.3% | 0.871 |

### Interpretation
- **High Precision (100%)**: When model predicts spam, it's almost always correct
- **Moderate Recall (77.18%)**: Model misses ~23% of actual spam messages
- **Trade-off**: Conservative approach prioritizes avoiding false positives (marking ham as spam)

## Usage in Inference

### Real-time Prediction (Streamlit App)
1. User enters raw text
2. Text is normalized using same pipeline (Step 1-5)
3. Normalized text is vectorized using saved TF-IDF vocabulary
4. Logistic Regression model predicts probability [0, 1]
5. If probability ≥ 0.5 (or custom threshold), classify as SPAM; else HAM

### Batch Prediction (CLI Script)
```bash
python scripts/predict_spam.py \
  --input datasets/processed/sms_spam_clean.csv \
  --text-col text_clean \
  --output predictions.csv \
  --threshold 0.5
```

## Artifacts & Reproducibility

All preprocessing and model artifacts are version-controlled and reproducible:

| Artifact | Location | Purpose |
|----------|----------|---------|
| Model | `models/model.pkl` | Trained Logistic Regression classifier |
| Vectorizer | `models/vectorizer.pkl` | TF-IDF vocabulary and IDF weights |
| Label Mapping | `models/label_mapping.json` | Maps numeric labels (0/1) to strings (ham/spam) |
| Metrics | `models/metrics_logistic_regression.json` | Train/test performance metrics |

## Key Insights

1. **Numbers are highly predictive of spam**: 39.4% of messages contain `<NUM>` tokens
2. **Precision over recall trade-off**: The model is conservative, prioritizing avoiding false positives
3. **Class imbalance**: Only 13.4% spam → naturally higher accuracy on majority class (ham)
4. **Generalization**: Test accuracy (96.95%) close to train accuracy (97.8%) suggests good generalization

## References

- Packt: [Hands-On AI for Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)
- Scikit-learn TF-IDF: [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- OpenSpec Workflow: [openspec/changes/add-spam-classification/](../../openspec/changes/add-spam-classification/)
