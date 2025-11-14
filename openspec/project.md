# Project Context

## Purpose
建立一個端到端的垃圾郵件分類系統，使用機器學習模型（Logistic Regression / Naïve Bayes / SVM）從文字資料中識別垃圾郵件。
目標：
- 複現與擴展《Hands-On Artificial Intelligence for Cybersecurity》第 3 章的內容
- 提供完整的 data preprocessing → model training → evaluation → visualization 工作流程
- 部署互動式 Streamlit 前端供使用者測試分類模型
- 演示 OpenSpec 規範驅動開發的完整工作流程

## Tech Stack
- **語言**: Python 3.11+
- **ML 框架**: scikit-learn（模型訓練）、pandas（資料處理）
- **向量化**: TF-IDF（特徵提取）
- **可視化**: matplotlib、seaborn、plotly
- **前端**: Streamlit（互動式 UI）
- **部署**: Streamlit Cloud （demo site）、GitHub（程式碼倉庫）
- **資料**: sms_spam_no_header.csv（GitHub Packt 倉庫）

## Project Conventions

### Code Style
- **格式化**: Black（Python）
- **Lint**: pylint、flake8（Python code quality）
- **命名慣例**: snake_case（Python 變數/函式/檔案）、PascalCase（類名）
- **型別提示**: 建議使用 type hints（Python 3.9+）

### Architecture Patterns
- **結構**: 模組化單體（monolithic with modules）
  - `src/` — 核心程式碼（preprocessing、model、evaluation）
  - `notebooks/` — 探索、教學、結果報告（Jupyter notebooks）
  - `app/` — Streamlit 應用
  - `data/` — 資料檔案與中間結果
  - `models/` — 訓練後的模型檔案（.pkl）
- **分層**: Data Pipeline → Model Training → Evaluation → Visualization/API

### Testing Strategy
- **單元測試**: pytest（資料 pipeline、模型功能）
- **覆蓋率目標**: 至少 80%（critical paths）
- **測試資料**: 使用公開資料集，無需真實用戶資料

### Git Workflow
- **分支策略**: GitHub Flow
  - `main` 為可發佈/已部署狀態
  - 功能分支命名：`feature/phase-<N>-<feature-name>` 或 `fix/<issue>`
- **Commit 訊息**: Conventional Commits
  - 例：`feat: add logistic regression baseline`、`docs: update README with setup steps`
- **PR review**: 至少一位 reviewer，CI 通過後合併

## Domain Context
- **垃圾郵件類別**: 本專案主要處理 SMS/Email 垃圾郵件與合法訊息的二元分類（spam/ham）。
- **資料來源**: Packt 倉庫提供的 sms_spam_no_header.csv（約 5K 筆訊息）
- **隱私與合規**: 使用的資料已公開且去識別，無需額外隱私合規處理。
- **目標使用者**: 教學/示範目的，非生產系統。

## Important Constraints
- **資料大小**: 訓練資料限於 ~5K 筆，不適合大規模深度學習。
- **計算資源**: 假設在 CPU 環境訓練（無 GPU 需求）。
- **部署環境**: Streamlit Cloud 免費方案（資源有限）。
- **模型複雜度**: 保持 baseline 模型簡單（Logistic Regression），便於解釋與教學。

## External Dependencies
- **資料來源**: https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity
- **Streamlit Cloud**: 部署演示應用
- **GitHub**: 程式碼倉庫與版本控制
- **教學資源**: YouTube 播放清單（Packt 官方）與 OpenSpec 教程

```
