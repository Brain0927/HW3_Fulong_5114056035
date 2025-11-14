## ADDED Requirements

### Requirement: Spam Email Classification
The system SHALL provide a spam-email classification capability that can be trained offline and served via an inference API. 系統 SHALL 可離線訓練並透過推論 API 提供垃圾郵件分類能力。

#### Scenario: Train baseline model
- **WHEN** developers run the training pipeline with the baseline dataset (`sms_spam_no_header.csv`) and default hyperparameters
- **THEN** the pipeline SHALL produce a trained model artifact and a report containing precision, recall, and F1 score on the validation set

#### Scenario: Inference API returns label and score
- **WHEN** a client POSTs a message text to the inference endpoint
- **THEN** the API SHALL return a JSON response including `label` (spam/ham) and `score` (probability), and the `score` SHALL be a float between 0.0 and 1.0

#### Scenario: Baseline model reproducibility
- **WHEN** the training pipeline is executed with the same dataset version and seed
- **THEN** the baseline model training results SHALL be reproducible within reasonable variance and documented in the training report
