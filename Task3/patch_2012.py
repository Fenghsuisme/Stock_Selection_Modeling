# patch_2012_growth.py
# 補抓2011損益表，算出2012的營業利益/稅後淨利成長率，
# 填回 crawled_top300.xlsx，移除全空欄、重排、輸出新檔。
#
# 用法：python patch_2012_growth.py
# 需要：config.py 裡有 FINMIND_TOKEN

import time
import requests
import pandas as pd
import numpy as np
from config import FINMIND_TOKEN, FINMIND_URL, DS_INCOME, API_SLEEP

INPUT_FILE  = "crawled_top300.xlsx"          # 現有檔案（放同目錄）
OUTPUT_FILE = "crawled_top300_final.xlsx"    # 輸出

# ── 工具函式 ──
def _api_get(dataset, data_id, start, end):
    params = {"dataset": dataset, "data_id": data_id,
              "start_date": start, "end_date": end, "token": FINMIND_TOKEN}
    for _ in range(3):
        try:
            data = requests.get(FINMIND_URL, params=params, timeout=30).json()
        except Exception as e:
            print(f"  連線錯誤 {data_id}: {e}，重試")
            time.sleep(10); continue
        if data.get("status") == 402 or "limit" in str(data.get("msg","")).lower():
            print(f"  ⚠️ 額度用盡，等待...")
            time.sleep(3700); continue
        time.sleep(API_SLEEP)
        return pd.DataFrame(data.get("data", [])) if data.get("status")==200 else pd.DataFrame()
    return pd.DataFrame()


def get_2011_income(stock_id):
    """抓2011全年損益表(四季加總)，回傳 {OperatingIncome, NetIncome}"""
    df = _api_get(DS_INCOME, stock_id, "2011-01-01", "2011-12-31")
    if df.empty:
        return {}
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # 四季加總
    agg = df.groupby("type")["value"].sum()
    net = None
    for k in ["NetIncome", "IncomeAfterTaxes", "TotalConsolidatedProfitForThePeriod"]:
        if k in agg.index and pd.notna(agg[k]):
            net = agg[k]; break
    return {
        "OperatingIncome": agg.get("OperatingIncome"),
        "NetIncome": net,
    }


def growth(cur, prev):
    if cur is None or prev is None or prev == 0 or pd.isna(cur) or pd.isna(prev):
        return None
    return round((cur - prev) / abs(prev) * 100, 4)


# ════════════════════════════════════════════
def main():
    df = pd.read_excel(INPUT_FILE)
    print(f"讀入 {INPUT_FILE}：{len(df)} 筆")

    # 需要算 2011 → 2012 成長率的股票（2012在榜）
    mask2012 = df["年月"] == 201212
    stocks = df[mask2012]["證券代碼"].astype(str).unique()
    print(f"需補抓 2011 損益表：{len(stocks)} 支\n")

    # 先取得每支股票 2012 的營業利益/稅後淨利（從現有資料的稅後淨利欄 + 需要重抓營業利益）
    # 但現有檔案沒有營業利益的絕對值，只有成長率，所以 2012 的絕對值也要重抓
    # → 乾脆同時抓 2011 和 2012，直接算成長率
    income_2011 = {}
    income_2012 = {}

    for i, sid in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {sid}")
        # 2011
        d11 = get_2011_income(sid)
        # 2012
        df12 = _api_get(DS_INCOME, sid, "2012-01-01", "2012-12-31")
        if not df12.empty:
            df12["value"] = pd.to_numeric(df12["value"], errors="coerce")
            agg12 = df12.groupby("type")["value"].sum()
            net12 = None
            for k in ["NetIncome","IncomeAfterTaxes","TotalConsolidatedProfitForThePeriod"]:
                if k in agg12.index and pd.notna(agg12[k]):
                    net12 = agg12[k]; break
            income_2012[sid] = {"OperatingIncome": agg12.get("OperatingIncome"),
                                "NetIncome": net12}
        income_2011[sid] = d11

    # 算成長率並填回
    filled_op, filled_net = 0, 0
    for sid in stocks:
        op11  = income_2011.get(sid, {}).get("OperatingIncome")
        net11 = income_2011.get(sid, {}).get("NetIncome")
        op12  = income_2012.get(sid, {}).get("OperatingIncome")
        net12 = income_2012.get(sid, {}).get("NetIncome")

        g_op  = growth(op12, op11)
        g_net = growth(net12, net11)

        row = (df["年月"] == 201212) & (df["證券代碼"].astype(str) == sid)
        if g_op is not None:
            df.loc[row, "M營業利益成長率"] = g_op
            filled_op += 1
        if g_net is not None:
            df.loc[row, "M稅後淨利成長率"] = g_net
            filled_net += 1

    print(f"\n補回 營業利益成長率 {filled_op} 筆、稅後淨利成長率 {filled_net} 筆")

    # ── 移除全空欄 ──
    drop_cols = []
    for c in df.columns:
        if df[c].notna().sum() == 0:
            drop_cols.append(c)
    if drop_cols:
        print(f"移除全空欄：{drop_cols}")
        df = df.drop(columns=drop_cols)

    # ── 重排 ──
    prof_order = [
        '證券代碼','簡稱','年月','市值(百萬元)','收盤價(元)_年',
        '股價淨值比','股價營收比','M淨值報酬率─稅後','資產報酬率ROA',
        '營業利益率OPM','利潤邊際NPM','負債/淨值比','M流動比率','M速動比率',
        'M存貨週轉率 (次)','M應收帳款週轉次','M營業利益成長率','M稅後淨利成長率',
    ]
    extra_order = [
        '毛利率','EPS','每股淨值BPS','現金比率','總資產成長率','負債比率',
        '殖利率','本益比PE','營收成長率','固定資產週轉率','總資產週轉率',
        '權益乘數','每股現金流量','現金流量比率','營業現金流量成長率',
        '應付帳款週轉率','稅後淨利',
    ]
    label_order = ['Return','ReturnMean_year_Label']
    final_order = [c for c in (prof_order+extra_order+label_order) if c in df.columns]

    df = df[final_order]
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n已輸出 {OUTPUT_FILE}：{len(df)} 筆，{len(df.columns)} 欄")

    # 驗證 2012 成長率缺失
    s = pd.to_numeric(df[df['年月']==201212]['M營業利益成長率'], errors='coerce')
    print(f"2012營業利益成長率缺失：{s.isna().sum()} / {len(s)}")


if __name__ == "__main__":
    main()