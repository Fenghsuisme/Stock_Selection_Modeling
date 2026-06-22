# crawlers/twse_crawler.py

import time
import requests
import pandas as pd
import logging
import sys
import os
from datetime import date, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import TWSE_SLEEP, TOP_N

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TWSE] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _find_last_trading_day(year: int, month: int) -> str:
    """
    從月底往前找，最多找10天，找到 TWSE 有資料的最後交易日。
    回傳 YYYYMMDD 字串。
    """
    # 算出該月最後一天
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    for i in range(10):
        d = last_day - timedelta(days=i)
        date_str = d.strftime("%Y%m%d")
        url = (
            f"https://www.twse.com.tw/exchangeReport/MI_INDEX"
            f"?response=json&date={date_str}&type=ALLBUT0999"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            data = resp.json()
            tables = data.get("tables", [])
            for t in tables:
                if "證券代號" in t.get("fields", []):
                    log.info(f"  找到交易日：{date_str}")
                    time.sleep(TWSE_SLEEP)
                    return date_str
        except Exception:
            pass
        time.sleep(1)

    log.warning(f"  找不到 {year}/{month} 的交易日")
    return None


def get_market_all(year: int, month: int = 12) -> pd.DataFrame:
    """
    抓某年某月全市場收盤行情（自動找最後交易日）。
    用「成交金額」作為規模排序依據。
    """
    log.info(f"抓全市場行情 {year}/{month:02d} ...")

    date_str = _find_last_trading_day(year, month)
    if date_str is None:
        return pd.DataFrame()

    url = (
        f"https://www.twse.com.tw/exchangeReport/MI_INDEX"
        f"?response=json&date={date_str}&type=ALLBUT0999"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"MI_INDEX 失敗 ({year}/{month}): {e}")
        return pd.DataFrame()

    time.sleep(TWSE_SLEEP)

    target_table = None
    for table in data.get("tables", []):
        if "證券代號" in table.get("fields", []):
            target_table = table
            break

    if target_table is None:
        log.warning(f"找不到個股 table ({year}/{month})")
        return pd.DataFrame()

    fields = target_table["fields"]
    rows   = target_table["data"]
    df = pd.DataFrame(rows, columns=fields)

    def clean_num(s):
        try:
            v = str(s).replace(",", "").replace("+", "").strip()
            if v in ("--", "-", "", "N/A"):
                return None
            return float(v)
        except Exception:
            return None

    df = df.rename(columns={"證券代號": "證券代碼", "證券名稱": "簡稱"})
    df["收盤價"]       = df["收盤價"].apply(clean_num)
    df["成交金額"]     = df["成交金額"].apply(clean_num)
    df["市值(百萬元)"] = df["成交金額"] / 1_000_000
    df = df[["證券代碼", "簡稱", "收盤價", "市值(百萬元)"]].copy()
    df = df[df["證券代碼"].str.match(r"^\d{4}$", na=False)]
    df = df.dropna(subset=["市值(百萬元)", "收盤價"])
    df = df[df["收盤價"] > 0]
    df = df.sort_values("市值(百萬元)", ascending=False).reset_index(drop=True)

    log.info(f"  → {len(df)} 支，最大：{df.iloc[0]['簡稱'] if len(df) else 'N/A'}")
    return df


def get_top_n_stocks(year: int, n: int = TOP_N) -> pd.DataFrame:
    df = get_market_all(year, month=12)
    if df.empty:
        log.warning(f"{year} 年資料為空")
        return df
    top = df.head(n).copy()
    top["年份"] = year
    log.info(f"{year} 年前{n}名：{top['證券代碼'].tolist()[:5]} ...")
    return top


def get_monthly_close(stock_id: str, year: int, month: int = 12):
    """抓單支股票某月最後交易日收盤價"""
    date_str = f"{year}{month:02d}01"
    url = (
        f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        f"?response=json&date={date_str}&stockNo={stock_id}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"STOCK_DAY 失敗 {stock_id} ({year}/{month}): {e}")
        return None

    time.sleep(TWSE_SLEEP)

    if data.get("stat") != "OK" or not data.get("data"):
        return None

    last_row = data["data"][-1]
    try:
        return float(str(last_row[6]).replace(",", ""))
    except Exception:
        return None


def get_close_prices_batch(stock_ids: list, year: int, month: int = 12) -> dict:
    result = {}
    total = len(stock_ids)
    for i, sid in enumerate(stock_ids, 1):
        log.info(f"  收盤價 [{i}/{total}] {sid} {year}/{month:02d}")
        price = get_monthly_close(sid, year, month)
        if price is not None:
            result[sid] = price
    return result


def get_valuation_data(year: int, month: int = 12) -> pd.DataFrame:
    """
    從 TWSE BWIBBU_d 抓殖利率、本益比、股價淨值比（自動找最後交易日）。
    """
    log.info(f"抓估值資料 {year}/{month:02d} ...")

    # 從月底往前找有資料的交易日
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    date_str = None
    for i in range(10):
        d = last_day - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/exchangeReport/BWIBBU_d?response=json&date={ds}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            data = resp.json()
            if data.get("stat") == "OK" and data.get("data"):
                date_str = ds
                log.info(f"  估值找到交易日：{date_str}")
                time.sleep(TWSE_SLEEP)
                break
        except Exception:
            pass
        time.sleep(1)

    if date_str is None:
        log.warning(f"  找不到 {year}/{month} 估值交易日")
        return pd.DataFrame()

    url = f"https://www.twse.com.tw/exchangeReport/BWIBBU_d?response=json&date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
    except Exception as e:
        log.error(f"BWIBBU_d 失敗: {e}")
        return pd.DataFrame()

    time.sleep(TWSE_SLEEP)

    fields = data["fields"]
    df = pd.DataFrame(data["data"], columns=fields)

    def clean_num(s):
        try:
            v = str(s).replace(",", "").strip()
            if v in ("--", "-", "", "N/A"):
                return None
            return float(v)
        except Exception:
            return None

    col_map = {}
    for c in df.columns:
        if "代號" in c:       col_map[c] = "證券代碼"
        elif "殖利率" in c:   col_map[c] = "殖利率"
        elif "本益比" in c:   col_map[c] = "本益比PE"
        elif "股價淨值比" in c: col_map[c] = "股價淨值比"
    df = df.rename(columns=col_map)

    keep = [c for c in ["證券代碼", "殖利率", "本益比PE", "股價淨值比"] if c in df.columns]
    df = df[keep].copy()
    for c in ["殖利率", "本益比PE", "股價淨值比"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_num)

    df = df[df["證券代碼"].str.match(r"^\d{4}$", na=False)]
    log.info(f"  → 估值資料 {len(df)} 筆")
    return df


if __name__ == "__main__":
    df = get_top_n_stocks(2020, n=5)
    print(df)