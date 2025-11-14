# Machine Learning — Spam Classification Capability

## ADDED Requirements

### Requirement: Spam Email Classification
The system SHALL provide a machine learning-powered spam email classification capability that can identify whether a given email or SMS message is spam or legitimate (ham). The system SHALL support training with publicly available datasets, model evaluation, and real-time inference via API or UI.

#### Scenario: Train baseline model
- **WHEN** developers run the training pipeline with the baseline dataset (`sms_spam_no_header.csv`) and default hyperparameters
- **THEN** the pipeline SHALL produce a trained model artifact (e.g., `.pkl` file) and a metrics report containing accuracy, precision, recall, F1 score, and confusion matrix

#### Scenario: Inference API returns label and score
- **WHEN** a client or UI sends a message text to the inference endpoint or function
- **THEN** the system SHALL return a classification result including `label` (spam/ham) and `confidence_score` (probability between 0.0 and 1.0)

#### Scenario: Model reproducibility
- **WHEN** the training pipeline is executed with the same dataset version and random seed
- **THEN** the model training results SHALL be reproducible within reasonable variance and documented in training logs and metrics

#### Scenario: Evaluation metrics accuracy
- **WHEN** the trained model is evaluated on the test set
- **THEN** the system SHALL compute and report accuracy, precision, recall, F1 score, and confusion matrix correctly

### Requirement: Data Preprocessing Pipeline
The system SHALL preprocess raw email/SMS text data for feature extraction, including lowercasing, tokenization, and removal of stop words.

#### Scenario: Text normalization
- **WHEN** raw text is loaded from the dataset
- **THEN** the preprocessing pipeline SHALL normalize text to lowercase and remove leading/trailing whitespace

#### Scenario: Feature vectorization
- **WHEN** preprocessed text is transformed to features
- **THEN** the system SHALL extract TF-IDF features with configurable max features (default 5,000) and support unigrams and bigrams

### Requirement: Multiple Model Support
The system SHALL support training and evaluation of multiple classifier models (Logistic Regression, Naive Bayes, SVM) to compare performance.

#### Scenario: Train Logistic Regression
- **WHEN** user specifies `--model logistic_regression` in the training script
- **THEN** the system SHALL train a Logistic Regression classifier and save it with appropriate hyperparameters

#### Scenario: Train Naive Bayes
- **WHEN** user specifies `--model naive_bayes`
- **THEN** the system SHALL train a Multinomial Naive Bayes classifier suitable for text classification

#### Scenario: Train SVM
- **WHEN** user specifies `--model svm`
- **THEN** the system SHALL train a LinearSVC (Support Vector Machine) classifier with appropriate settings

### Requirement: Interactive Visualization Dashboard
The system SHALL provide a web-based UI for real-time spam classification and model performance visualization using Streamlit.

#### Scenario: Classify message via UI
- **WHEN** a user enters a message in the Streamlit app and clicks "Classify"
- **THEN** the app SHALL display the classification result (SPAM/HAM) and confidence score with visual indicators

#### Scenario: View model performance
- **WHEN** a user navigates to the "Model Performance" tab
- **THEN** the app SHALL display accuracy, precision, recall, F1 metrics, and comparison charts for validation and test sets

#### Scenario: Provide example messages
- **WHEN** the user is on the Classifier tab
- **THEN** the app SHALL display example spam and ham messages to help users understand the classifier's purpose
