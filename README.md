# Stock Selection Modeling

運用機器學習進行台股選股的專題，目標為挑選出「打敗大盤」的股票組合。專案分為三個子任務：

| 任務 | 方法 | 資料 |
| --- | --- | --- |
| **Task1** | 自製連續型 ID3 決策樹分類器，以財務特徵預測是否打敗大盤 | 台股前 200 大 |
| **Task2** | GA（基因演算法）優化特徵子集與 SVR 超參數，SVR 回歸預測未來報酬後選股 | 台股前 200 大 |
| **Task3** | FinMind / TWSE / Goodinfo 爬蟲建資料集，再套用 GA + SVR 選股 | 台股前 300 大（2012–2024） |

所有驗證皆採時間切分（temporal validation，擴張視窗），訓練資料僅使用早於測試年份的資料，避免資料洩漏（data leakage）。

## 目錄結構

```
Task1/   ID3 決策樹模型與結果圖表
Task2/   GA + SVR 模型（top200）
Task3/   資料爬蟲 + GA + SVR 模型（top300）
  config.py      全域設定
  crawlers/      FinMind / TWSE / Goodinfo 爬蟲
  processors/    財務比率計算與資料處理
  ga/            GA + SVR 模型
```

## 環境需求

- Python 3.13
- 套件：`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`openpyxl` 等（Task3 另見 `Task3/requirements.txt`）

```bash
python -m venv venv
source venv/bin/activate
pip install -r Task3/requirements.txt
```

## FinMind Token 設定（Task3 需要）

Task3 的爬蟲需要 [FinMind](https://finmindtrade.com/) 的 API token。請自行註冊後以環境變數提供，**請勿將 token 寫進程式碼**：

```bash
export FINMIND_TOKEN="你的token"
```

## 執行

```bash
# Task1
python Task1/src/task1.py

# Task2
python Task2/gasvr.py

# Task3（需先設定 FINMIND_TOKEN）
python Task3/main.py
```

## 授權

本專案採用 [MIT License](LICENSE) 授權。

> 免責聲明：本專案僅供學術研究與教學用途，所有回測結果不構成任何投資建議。
