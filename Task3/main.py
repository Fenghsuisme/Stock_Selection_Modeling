# main.py — Task 3 爬蟲主程式（FinMind 版本）

import os
import json
import logging
import pandas as pd
from tqdm import tqdm

from config import (
    START_YEAR, END_YEAR, TOP_N, FINMIND_TOKEN,
)
from crawlers.twse_crawler import get_top_n_stocks
from crawlers.finmind_crawler import (
    get_income, get_balance, get_cashflow, get_per, get_price_year_end,
)
from processors.ratio_calculator import calc_ratios_for_stock
from processors.processor import build_final, save_to_excel

CACHE_DIR    = "output/cache"
MARKET_CACHE = os.path.join(CACHE_DIR, "market_{year}.csv")
RECORDS_CACHE = os.path.join(CACHE_DIR, "finmind_records.json")
FINAL_OUTPUT = "output/crawled_top300.xlsx"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger().addHandler(
    logging.FileHandler("logs/crawler.log", encoding="utf-8")
)
log = logging.getLogger(__name__)


def step1_get_market() -> dict:
    """每年成交金額前300（沿用 TWSE 爬蟲）"""
    log.info("=" * 50)
    log.info("Step 1: 抓每年成交金額前300清單")
    log.info("=" * 50)
    market_by_year = {}
    for year in range(START_YEAR, END_YEAR + 1):
        cache_path = MARKET_CACHE.format(year=year)
        if os.path.exists(cache_path):
            log.info(f"  {year} 讀快取")
            df = pd.read_csv(cache_path, dtype={"證券代碼": str})
        else:
            df = get_top_n_stocks(year, n=TOP_N)
            if not df.empty:
                df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        market_by_year[year] = df
    return market_by_year


def step2_crawl_finmind(market_by_year: dict) -> list:
    """
    用 FinMind 爬每支股票的財務資料並算指標。
    斷點續爬：已完成的股票存在 records cache。
    """
    log.info("=" * 50)
    log.info("Step 2: FinMind 爬取財務資料 + 計算34指標")
    log.info("=" * 50)

    # 收集不重複股票
    all_stocks = {}
    for df in market_by_year.values():
        if df.empty or "證券代碼" not in df.columns:
            continue
        for _, row in df.iterrows():
            sid = str(row["證券代碼"])
            all_stocks[sid] = str(row.get("簡稱", ""))

    # 每支股票出現在哪些年份（只算那些年的指標）
    stock_years = {}
    for year, df in market_by_year.items():
        if df.empty or "證券代碼" not in df.columns:
            continue
        for sid in df["證券代碼"].astype(str):
            stock_years.setdefault(sid, set()).add(year)

    # 讀斷點
    if os.path.exists(RECORDS_CACHE):
        with open(RECORDS_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        records = cache["records"]
        done = set(cache["done"])
        log.info(f"  讀斷點：已完成 {len(done)} 支")
    else:
        records = []
        done = set()

    todo = [(s, n) for s, n in all_stocks.items() if s not in done]
    log.info(f"  待爬 {len(todo)} / {len(all_stocks)} 支")

    # 市值/收盤價對照（從 market cache 取）
    market_lookup = {}  # {(sid, year): {市值, 收盤價}}
    for year, df in market_by_year.items():
        if df.empty or "證券代碼" not in df.columns:
            continue
        for _, row in df.iterrows():
            sid = str(row["證券代碼"])
            market_lookup[(sid, year)] = {
                "市值(百萬元)": row.get("市值(百萬元)"),
                "收盤價(元)_年": row.get("收盤價"),
            }

    for idx, (sid, name) in enumerate(todo, 1):
        log.info(f"[{len(done)+1}/{len(all_stocks)}] {sid} {name}")
        years = sorted(stock_years.get(sid, []))

        try:
            income   = get_income(sid, START_YEAR, END_YEAR)
            balance  = get_balance(sid, START_YEAR, END_YEAR)
            cashflow = get_cashflow(sid, START_YEAR, END_YEAR)
            per      = get_per(sid, START_YEAR, END_YEAR)
        except Exception as e:
            log.warning(f"  {sid} 抓取失敗：{e}")
            done.add(sid)
            continue

        # 抓收盤價（當年 + 隔年，算 Return）
        prices = {}
        for y in range(min(years), max(years) + 2):
            p = get_price_year_end(sid, y)
            if p.get("close"):
                prices[y] = p["close"]

        # 逐年算指標
        for year in years:
            ratios = calc_ratios_for_stock(income, balance, cashflow, per, year)

            # 市值、收盤價
            mk = market_lookup.get((sid, year), {})
            ratios["市值(百萬元)"] = mk.get("市值(百萬元)")
            ratios["收盤價(元)_年"] = prices.get(year) or mk.get("收盤價(元)_年")

            # Return = (P_{y+1} - P_y) / P_y * 100
            p_now  = prices.get(year)
            p_next = prices.get(year + 1)
            if p_now and p_next and p_now > 0:
                ratios["Return"] = round((p_next - p_now) / p_now * 100, 4)
            else:
                ratios["Return"] = None

            ratios["證券代碼"] = sid
            ratios["簡稱"]     = name
            ratios["年份"]     = year
            records.append(ratios)

        done.add(sid)

        # 每5支存一次斷點
        if idx % 5 == 0:
            with open(RECORDS_CACHE, "w", encoding="utf-8") as f:
                json.dump({"records": records, "done": list(done)},
                          f, ensure_ascii=False)
            log.info(f"  已存斷點（{len(done)} 支）")

    # 最後存一次
    with open(RECORDS_CACHE, "w", encoding="utf-8") as f:
        json.dump({"records": records, "done": list(done)},
                  f, ensure_ascii=False)

    return records


def step3_output(records: list):
    log.info("=" * 50)
    log.info("Step 3: 整合輸出")
    log.info("=" * 50)
    final_df = build_final(records)
    save_to_excel(final_df, FINAL_OUTPUT)
    log.info(f"完成！{len(final_df)} 筆")
    print(final_df.head())


if __name__ == "__main__":
    if FINMIND_TOKEN == "請貼上你的token":
        log.error("請先到 config.py 填入你的 FinMind token！")
        sys.exit(1)

    log.info(f"Task 3 (FinMind) 開始：{START_YEAR}~{END_YEAR}，每年前{TOP_N}支")
    market_by_year = step1_get_market()
    records = step2_crawl_finmind(market_by_year)
    step3_output(records)
    log.info("全部完成！")