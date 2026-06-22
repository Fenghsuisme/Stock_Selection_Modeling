# test_twse.py — 先測試 TWSE API 能不能用，不需要 Selenium
# 執行：python test_twse.py

import sys
sys.path.append(".")
from crawlers.twse_crawler import get_top_n_stocks, get_monthly_close

print("=== 測試 1：抓 2020 年市值前10名 ===")
df = get_top_n_stocks(2020, n=10)
print(df[["證券代碼", "簡稱", "市值(百萬元)"]].to_string(index=False))

print("\n=== 測試 2：台積電 2020 年12月收盤價 ===")
price = get_monthly_close("2330", 2020, 12)
print(f"台積電 2020/12 收盤價：{price}")

print("\n=== 測試 3：聯發科 2021 年12月收盤價 ===")
price2 = get_monthly_close("2454", 2021, 12)
print(f"聯發科 2021/12 收盤價：{price2}")