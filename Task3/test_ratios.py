# test_ratios.py — 用台積電驗證34個指標都算得出來
# 執行：python test_ratios.py
# 只花 4 次 API

import sys
sys.path.append(".")
from crawlers.finmind_crawler import get_income, get_balance, get_cashflow, get_per
from processors.ratio_calculator import calc_ratios_for_stock

print("抓取台積電資料中（約30秒）...")
income   = get_income("2330", 2012, 2024)
balance  = get_balance("2330", 2012, 2024)
cashflow = get_cashflow("2330", 2012, 2024)
per      = get_per("2330", 2012, 2024)

print("\n=== 2023 年台積電 34 指標 ===")
ratios = calc_ratios_for_stock(income, balance, cashflow, per, 2023)

ok, empty = 0, 0
for k, v in ratios.items():
    if k.startswith("_"):
        continue
    status = "✅" if v is not None else "❌ 空"
    if v is not None:
        ok += 1
    else:
        empty += 1
    print(f"  {status}  {k}: {v}")

print(f"\n有值 {ok} 個，空值 {empty} 個")