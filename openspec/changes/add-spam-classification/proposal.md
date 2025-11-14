# Change: Add Spam Email Classification (add-spam-classification)

## Why
偵測並分類垃圾郵件可提升使用者體驗並降低詐騙/濫發風險。本變更旨在引入機器學習驅動的垃圾郵件分類能力，提供可訓練、可驗證與可部署的 baseline 模型。

## What Changes
- 建立一個機器學習 (ML) pipeline，用以訓練與評估垃圾郵件分類模型（Phase 1: baseline）。
- 新增可供推論的 API 端點以回傳分類結果。
- 引入資料提取與處理腳本，包含從公開資料庫取得訓練資料的步驟。
- 為未來 PhaseN 保留流程與目錄結構（placeholder）。

## Dataset
- Phase 1 baseline 使用的資料來源（公開 CSV）：
  https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv

## Plan / Phases
- `phase1-baseline` (已定義)：建立基本的垃圾郵件分類 baseline，使用 Logistic Regression 作為初始模型，並實作訓練、評估與簡單推論 API。
  - 注意：你在計畫中提到 SVM 作為 baseline，但描述最初表明目標為 Logistic Regression。此提案以 Logistic Regression 為 baseline；如需改為 SVM，請回覆我會更新提案與 tasks。
- `phase2` .. `phaseN`：保留空白節點，之後再填入（例如：特徵工程強化、深度學習模型、線上學習、資料標注工具等）。

## Impact
- 受影響的規格: `specs/ml/spec.md`（新增 capability）
- 受影響的程式: `ml/`（新增資料與訓練腳本）、`src/api/ml/*`（推論 API）、CI 配置以加入訓練/測試工作流程
- 需要注意的合規與隱私事項：資料來源的授權與敏感性，若使用真實用戶郵件需先處理隱私/脫識別
