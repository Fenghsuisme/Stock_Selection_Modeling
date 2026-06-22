# test_goodinfo.py — 測試 Goodinfo Selenium 爬蟲
# 執行：python test_goodinfo.py

import sys
sys.path.append(".")
from crawlers.goodinfo_crawler import _make_driver, fetch_one_stock

print("=== 測試：建立 Chrome Driver ===")
driver = _make_driver()
print("Driver 建立成功！")

print("\n=== 測試：爬台積電 (2330) 財務比率 ===")
df = fetch_one_stock(driver, "2330")
driver.quit()

if df.empty:
    print("❌ 爬取失敗，資料為空")
else:
    print(f"✅ 成功！共 {len(df)} 年資料，{len(df.columns)} 個指標")
    print("\n年份清單:", df.index.tolist())
    print("\n指標清單:", df.columns.tolist())
    print("\n最近3年資料:")
    print(df.tail(3).to_string())