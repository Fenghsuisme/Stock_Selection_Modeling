# test_finmind_fields.py
# 確認資產負債表的確切欄位名稱（程式裡有些是猜的，要驗證）
# 執行：python test_finmind_fields.py

import requests, pandas as pd
from config import FINMIND_TOKEN, FINMIND_URL, DS_BALANCE, DS_INCOME, DS_CASHFLOW

def check(dataset, keywords):
    params = {"dataset": dataset, "data_id": "2330",
              "start_date": "2020-01-01", "end_date": "2020-12-31",
              "token": FINMIND_TOKEN}
    r = requests.get(FINMIND_URL, params=params, timeout=20).json()
    df = pd.DataFrame(r.get("data", []))
    if df.empty:
        print(f"  {dataset}: 無資料")
        return
    types = sorted(df["type"].unique())
    print(f"\n=== {dataset}（共{len(types)}個欄位）===")
    for kw in keywords:
        matches = [t for t in types if kw.lower() in t.lower()]
        print(f"  含'{kw}': {matches}")

# 確認資產負債表關鍵欄位
check(DS_BALANCE, ["TotalAssets", "TotalLiabilit", "Equity",
                   "Inventor", "PropertyPlant", "CurrentAssets",
                   "CurrentLiabilit", "CashAndCash", "AccountsReceivable",
                   "AccountsPayable", "CapitalStock"])

# 確認損益表
check(DS_INCOME, ["Revenue", "GrossProfit", "OperatingIncome",
                  "NetIncome", "CostOfGoods", "EPS", "PreTax", "BeforeIncomeTax"])

# 確認現金流量表
check(DS_CASHFLOW, ["Operating", "Depreciation"])