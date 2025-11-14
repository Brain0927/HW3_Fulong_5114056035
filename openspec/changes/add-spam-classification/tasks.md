## Phase 1 — Baseline
- [ ] 1.1 下載並檢視資料：從指定來源取得 `sms_spam_no_header.csv` 並檢查資料品質
- [ ] 1.2 建立資料處理腳本：清理、分割（train/val/test）、文字向量化（例如 TF-IDF）
- [ ] 1.3 建立 baseline 模型訓練腳本：Logistic Regression（或改為 SVM，如需）
- [ ] 1.4 評估：計算 precision/recall/F1、混淆矩陣，並記錄 baseline 指標
- [ ] 1.5 建立簡易推論 API（HTTP），可接受輸入文字並回傳垃圾郵件機率/標籤
- [ ] 1.6 撰寫單元測試與整合測試（資料 pipeline 與 API）
- [ ] 1.7 文件：說明如何重現訓練過程與如何呼叫推論 API
- [ ] 1.8 驗證：`openspec validate add-spam-classification --strict`

## Phase 2..N (placeholders)
- [ ] 2.1 (empty)
- [ ] 2.2 (empty)
