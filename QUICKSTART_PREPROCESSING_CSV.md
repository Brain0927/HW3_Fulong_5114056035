# 🔄 9列預處理管道CSV 快速使用指南

## 概述

新的9列預處理管道CSV讓你可以在Streamlit應用中**可視化文本轉換的每一步**。

## 文件結構

### 簡單格式 (2列)
- 文件：`data/sms_spam_clean.csv`
- 列：`label`, `message`
- 用途：快速分類和分析

### 預處理管道格式 (8列) ⭐ NEW
- 文件：`data/sms_spam_preprocessing.csv`
- 列：
  1. `label` — 類別 (ham/spam)
  2. `text_raw` — 原始文本
  3. `text_lower` — 小寫轉換
  4. `text_contacts_masked` — 電郵/電話遮罩
  5. `text_numbers` — 數字替換 (<NUM>)
  6. `text_stripped` — 標點移除
  7. `text_whitespace` — 空白正規化
  8. `text_stopwords_removed` — 停用詞移除

## 使用方法

### 步驟1：啟動Streamlit應用
```bash
streamlit run app.py
```

### 步驟2：選擇預處理CSV
在側邊欄的「📊 Data Input」部分：
1. 點擊「Select Dataset」下拉菜單
2. 選擇 `data/sms_spam_preprocessing.csv`

### 步驟3：應用自動偵測格式
當選擇9列CSV後，應用將：
- 自動偵測為「預處理管道」格式
- 顯示訊息：「📈 **Preprocessing Pipeline CSV Detected**」
- 在側邊欄添加「Select Preprocessing Stage」選擇器
- 在主導航中添加新標籤：「🔄 Preprocessing Pipeline」

### 步驟4：探索前處理管道

在「🔄 Preprocessing Pipeline」標籤中：

#### 選擇示例
- 點擊「📩 Show HAM Example」或「🚨 Show SPAM Example」
- 自動加載該類別的第一條消息

#### 查看轉換
- 點擊每個階段 (text_raw, text_lower, ...)
- 查看文本在該階段的狀態

#### 分析指標
每個階段顯示：
- **字符數** — 文本長度
- **令牌數** — 詞數
- **平均令牌長度** — 平均詞長

#### 追蹤變更
「Token Changes Across Stages」表格顯示：
- 移除的令牌 (例：標點符號、停用詞)
- 每個階段移除/添加多少令牌

## 示例

### 文本轉換過程

原始消息：
```
"Free entry in 2 a wkly comp to win cash NOW!"
```

**階段1 - text_raw:**
```
Free entry in 2 a wkly comp to win cash NOW!
```
- 字符: 47, 令牌: 9

**階段2 - text_lower:**
```
free entry in 2 a wkly comp to win cash now!
```
- 字符: 47, 令牌: 9

**階段3 - text_numbers:**
```
free entry in <NUM> a wkly comp to win cash now!
```
- 字符: 50, 令牌: 9

**階段4 - text_stripped:**
```
free entry in <NUM> a wkly comp to win cash now
```
- 字符: 49, 令牌: 9

**階段5 - text_whitespace:**
```
free entry in <NUM> a wkly comp to win cash now
```
- 字符: 49, 令牌: 9

**階段6 - text_stopwords_removed:**
```
free entry <NUM> wkly comp win cash
```
- 字符: 34, 令牌: 6 (移除3個停用詞)

## 功能特點

✅ **自動格式偵測** — 無需手動配置
✅ **雙格式支援** — 簡單CSV和複雜管道CSV
✅ **可視化轉換** — 查看每一步的效果
✅ **令牌追蹤** — 理解每個階段的變更
✅ **統計分析** — 字符和令牌計數
✅ **快速示例** — 一鍵查看ham/spam例子

## 側邊欄控制

當加載預處理CSV後，側邊欄顯示：

```
📊 Data Input
  └─ Select Dataset: data/sms_spam_preprocessing.csv
  └─ 📈 Preprocessing Pipeline CSV Detected
  └─ Select Preprocessing Stage: [All / text_raw / text_lower / ...]
```

選項：
- **All** — 顯示所有7個預處理階段
- **Specific Stage** — 只顯示該階段的詳細信息

## 應用程式標籤

### 📊 Data Overview
- 類別分佈 (使用預處理後的文本)
- 頻繁令牌 (每個類別)

### 🔍 Model Performance
- 模型指標 (accuracy, precision, recall)
- 混淆矩陣
- 閾值掃描

### 💬 Live Inference
- 輸入消息進行分類
- 查看歸一化文本

### 🔄 Preprocessing Pipeline ⭐ NEW
- 7個預處理階段的完整可視化
- 每個階段的文本狀態
- 令牌變更追蹤
- 統計指標

### ℹ️ About
- 項目信息
- 數據集統計
- 技術棧

## 常見問題

**Q: 我可以上傳自己的9列CSV嗎？**
A: 是的！只要你的CSV有以下任意3個或以上的列即可：
- text_raw, text_lower, text_contacts_masked, text_numbers
- text_stripped, text_whitespace, text_stopwords_removed

**Q: 如果我的CSV有不同的列名怎麼辦？**
A: 應用會嘗試偵測包含這些關鍵詞的列。列名應該包含前處理階段的標記 (例：text_lower, text_numbers, 等)。

**Q: 為什麼有些行在預處理後令牌更少？**
A: 這是因為停用詞 (the, a, an, in, of, ...) 被移除了。這是正常的文本預處理行為。

**Q: 我可以改變預處理階段嗎？**
A: 在側邊欄的「Select Preprocessing Stage」中選擇特定階段，或選擇「All」查看所有階段。

## 技術細節

### 自動偵測邏輯
```python
def detect_csv_format(df):
    # 如果有8+列 AND 包含3+個預處理標記
    # → 視為「preprocessing」格式
    # 否則 → 視為「simple」格式
```

### 提取的階段
應用會自動提取這7個階段 (如果存在)：
1. text_raw
2. text_lower
3. text_contacts_masked
4. text_numbers
5. text_stripped
6. text_whitespace
7. text_stopwords_removed

## 下一步

- 試試用自己的預處理CSV
- 比較不同類別的令牌變更
- 分析停用詞對文本的影響
- 實驗不同的預處理方法

---

📧 問題？查看 [README.md](README.md) 或訪問 [GitHub](https://github.com/Brain0927/HW3_Fulong_5114056035)
