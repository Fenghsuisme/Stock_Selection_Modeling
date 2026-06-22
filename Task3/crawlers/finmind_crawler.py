# crawlers/finmind_crawler.py
# 用 FinMind API 抓取五個資料集，含速率控制與額度偵測

import time
import requests
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FINMIND_URL, FINMIND_TOKEN, API_SLEEP, RATE_LIMIT_WAIT,
    DS_INCOME, DS_BALANCE, DS_CASHFLOW, DS_PER, DS_PRICE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [FinMind] %(message)s")
log = logging.getLogger(__name__)


def _api_get(dataset: str, data_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    呼叫 FinMind API，含速率控制與額度耗盡自動等待。
    回傳 long-format DataFrame（date, stock_id, type, value）。
    """
    params = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": FINMIND_TOKEN,
    }

    for attempt in range(3):
        try:
            resp = requests.get(FINMIND_URL, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            log.warning(f"  {dataset} {data_id} 連線錯誤：{e}，重試...")
            time.sleep(10)
            continue

        status = data.get("status")
        msg = data.get("msg", "")

        # 額度用盡
        if status == 402 or "limit" in str(msg).lower() or "request" in str(msg).lower():
            log.warning(f"  ⚠️ API 額度用盡，等待 {RATE_LIMIT_WAIT} 秒...")
            time.sleep(RATE_LIMIT_WAIT)
            continue

        time.sleep(API_SLEEP)

        if status == 200:
            return pd.DataFrame(data.get("data", []))
        else:
            # 其他錯誤（如該股票無資料）
            return pd.DataFrame()

    return pd.DataFrame()


def get_price_year_end(stock_id: str, year: int) -> dict:
    """
    抓某股票某年12月的股價資料，取最後交易日。
    回傳 {close, trading_money}
    """
    df = _api_get(DS_PRICE, stock_id, f"{year}-12-01", f"{year}-12-31")
    if df.empty:
        return {}
    df = df.sort_values("date")
    last = df.iloc[-1]
    return {
        "close": last.get("close"),
        "trading_money": last.get("Trading_money"),
    }


def get_income(stock_id: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    損益表，回傳 wide-format。
    損益表是「單季」流量資料，需把全年四季加總。
    """
    df = _api_get(DS_INCOME, stock_id, f"{start_year}-01-01", f"{end_year}-12-31")
    return _pivot_annual_sum(df)


def get_balance(stock_id: str, start_year: int, end_year: int) -> pd.DataFrame:
    """資產負債表"""
    df = _api_get(DS_BALANCE, stock_id, f"{start_year}-01-01", f"{end_year}-12-31")
    return _pivot_annual(df)


def get_cashflow(stock_id: str, start_year: int, end_year: int) -> pd.DataFrame:
    """現金流量表（單季流量，需全年加總）"""
    df = _api_get(DS_CASHFLOW, stock_id, f"{start_year}-01-01", f"{end_year}-12-31")
    return _pivot_annual_sum(df)


def get_per(stock_id: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    PER 資料集（本益比/股價淨值比/殖利率），每日資料。
    回傳 wide-format：index=年份，columns=[PER, PBR, dividend_yield]，取每年最後一筆。
    """
    df = _api_get(DS_PER, stock_id, f"{start_year}-01-01", f"{end_year}-12-31")
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df["年份"] = df["date"].dt.year
    # 每年取最後一個交易日
    df = df.sort_values("date").groupby("年份").last()
    keep = [c for c in ["PER", "PBR", "dividend_yield"] if c in df.columns]
    return df[keep]


def _pivot_annual_sum(df: pd.DataFrame) -> pd.DataFrame:
    """
    把單季流量資料（損益表、現金流量表）加總成全年。
    EPS 例外：用全年加總（四季 EPS 相加 ≈ 全年 EPS）。
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["年份"] = df["date"].dt.year

    # 每年每個 type 把所有季加總
    wide = df.pivot_table(index="年份", columns="type", values="value", aggfunc="sum")
    return wide


def _pivot_annual(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 long-format（date, type, value）轉成 wide-format（index=年份, columns=type）。
    財報用年度資料：每年取12-31（年報）那筆，若無則取該年最後一筆。
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["年份"] = df["date"].dt.year

    # 優先取年報（12月底），否則取該年最新一筆
    def pick_annual(group):
        annual = group[group["date"].dt.month == 12]
        if not annual.empty:
            return annual[annual["date"] == annual["date"].max()]
        return group[group["date"] == group["date"].max()]

    df = df.groupby("年份", group_keys=False).apply(pick_annual)

    # pivot：年份 × type
    wide = df.pivot_table(index="年份", columns="type", values="value", aggfunc="first")
    return wide


if __name__ == "__main__":
    # 測試台積電
    print("=== 損益表 ===")
    inc = get_income("2330", 2012, 2024)
    print(inc[["Revenue", "GrossProfit", "OperatingIncome", "NetIncome"]].tail(3))

    print("\n=== PER ===")
    per = get_per("2330", 2012, 2024)
    print(per.tail(3))