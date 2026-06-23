# config.py — 全域設定（FinMind 版本）

# ─────────────────────────────────────────────
# 註冊：https://finmindtrade.com/
# ─────────────────────────────────────────────
FINMIND_TOKEN = "REMOVED-FINMIND-TOKEN"
# 專題結束就刪

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

# ─────────────────────────────────────────────
# 爬取設定
# ─────────────────────────────────────────────
START_YEAR = 2012   # 資產負債表/現金流量表從 2012 起最完整
END_YEAR   = 2024
TOP_N      = 300    # 每年取成交金額前N名

# API 速率控制（FinMind 免費版 600 次/小時 = 每 6 秒 1 次）
API_SLEEP = 6.0
RATE_LIMIT_WAIT = 3700  # 觸發額度上限時等待秒數

# TWSE API 速率（抓市值排行用）
TWSE_SLEEP = 4.0

# ─────────────────────────────────────────────
# 五個資料集名稱
# ─────────────────────────────────────────────
DS_INCOME   = "TaiwanStockFinancialStatements"
DS_BALANCE  = "TaiwanStockBalanceSheet"
DS_CASHFLOW = "TaiwanStockCashFlowsStatement"
DS_PER      = "TaiwanStockPER"
DS_PRICE    = "TaiwanStockPrice"

# ─────────────────────────────────────────────
# 輸出欄位（對齊原始 xlsx）
# ─────────────────────────────────────────────
ORIGINAL_COLUMNS = [
    "證券代碼", "簡稱", "年月",
    "市值(百萬元)", "收盤價(元)_年",
    "Unknown masked parameter",
    "股價淨值比", "股價營收比",
    "M淨值報酬率─稅後", "資產報酬率ROA",
    "營業利益率OPM", "利潤邊際NPM",
    "負債/淨值比", "M流動比率", "M速動比率",
    "M存貨週轉率 (次)", "M應收帳款週轉次",
    "M營業利益成長率", "M稅後淨利成長率",
    "Return", "ReturnMean_year_Label",
]

EXTRA_COLUMNS = [
    "毛利率", "EPS", "每股淨值BPS", "現金比率", "利息保障倍數",
    "總資產成長率", "負債比率", "殖利率", "本益比PE", "營收成長率",
    "固定資產週轉率", "總資產週轉率", "權益乘數", "每股現金流量",
    "現金流量比率", "營業現金流量成長率", "應付帳款週轉率", "稅後淨利",
]

ALL_COLUMNS = ORIGINAL_COLUMNS + EXTRA_COLUMNS