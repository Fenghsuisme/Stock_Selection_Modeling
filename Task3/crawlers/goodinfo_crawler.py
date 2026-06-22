# crawlers/goodinfo_crawler.py

import time
import re
import random
import logging
import sys
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import GOODINFO_SLEEP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Goodinfo] %(message)s")
log = logging.getLogger(__name__)

CHROMEDRIVER_PATH = (
    "/Users/bro/.wdm/drivers/chromedriver/mac64/"
    "149.0.7827.155/chromedriver-mac-arm64/chromedriver"
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/93.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.119 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36",
]

GOODINFO_URLS = [
    "https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=XX_M_YEAR&STOCK_ID={stock_id}&QRY_TIME=2025",
    "https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=XX_M_YEAR&STOCK_ID={stock_id}&QRY_TIME=2013",
]

COLUMN_MAP = [
    ("營業毛利率",          "毛利率"),
    ("營業利益率",          "營業利益率OPM"),
    ("稅後淨利率",          "利潤邊際NPM"),
    ("股東權益報酬率",      "M淨值報酬率─稅後"),
    ("資產報酬率",          "資產報酬率ROA"),
    ("股價淨值比",          "股價淨值比"),
    ("股價營收比",          "股價營收比"),
    ("負債對淨值比率",      "負債/淨值比"),
    ("流動比",              "M流動比率"),
    ("速動比",              "M速動比率"),
    ("存貨週轉率",          "M存貨週轉率 (次)"),
    ("應收帳款週轉率",      "M應收帳款週轉次"),
    ("營業利益年成長率",    "M營業利益成長率"),
    ("稅後淨利年成長率",    "M稅後淨利成長率"),
    ("每股稅後盈餘",        "EPS"),
    ("每股淨值",            "每股淨值BPS"),
    ("現金比",              "現金比率"),
    ("利息保障倍數",        "利息保障倍數"),
    ("資產總額年成長率",    "總資產成長率"),
    ("負債總額 (%)",        "負債比率"),
    ("殖利率",              "殖利率"),
    ("本益比",              "本益比PE"),
    ("營收年成長率",        "營收成長率"),
    ("固定資產週轉率",      "固定資產週轉率"),
    ("總資產週轉率",        "總資產週轉率"),
    ("淨值週轉率",          "權益乘數"),
    ("每股營業現金流量",    "每股現金流量"),
    ("現金流量比",          "現金流量比率"),
    ("應付帳款週轉率",      "應付帳款週轉率"),
    ("稅後淨利年成長率(母公司)", "稅後淨利成長率(母)"),
]

# 連續失敗幾次後暫停
MAX_CONSECUTIVE_FAIL = 3
PAUSE_ON_BLOCK = 300  # 秒（5分鐘）


def _make_driver(user_agent: str = None) -> webdriver.Chrome:
    if user_agent is None:
        user_agent = random.choice(USER_AGENTS)
    options = Options()
    # 不用 headless，避免被 Goodinfo 偵測
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"user-agent={user_agent}")
    # 隱藏 webdriver 特徵，避免被偵測
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    # 覆蓋 navigator.webdriver
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def _clean_num(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("%", "").replace("+", "")
    if s in ("N/A", "--", "-", "", "N.A.", "―", "－"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _map_column(text: str):
    for keyword, col_name in COLUMN_MAP:
        if keyword in text:
            return col_name
    return None


BLOCK_KEYWORDS = ["瀏覽量異常", "暫時關閉服務", "請稍後再重新使用"]
BATCH_SIZE = 30       # 每爬幾支休息一次
BATCH_SLEEP = 120     # 休息幾秒（2分鐘）


def _is_blocked(page_source: str) -> bool:
    """偵測 Goodinfo 封鎖頁面"""
    return any(kw in page_source for kw in BLOCK_KEYWORDS)


def _parse_table(driver, url: str) -> dict:
    try:
        driver.get(url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "tblDetail"))
        )
    except Exception as e:
        # 先檢查是不是被封了
        if _is_blocked(driver.page_source):
            log.warning(f"  ⚠️ 偵測到封鎖頁面！")
            return None
        log.warning(f"  載入失敗 {url}: {e}")
        return None  # None 表示載入失敗（可能被封），空dict表示無資料

    # 隨機等待，模擬人類行為
    time.sleep(random.uniform(GOODINFO_SLEEP, GOODINFO_SLEEP + 3))

    page = driver.page_source

    # 主動偵測封鎖
    if _is_blocked(page):
        log.warning(f"  ⚠️ 偵測到封鎖頁面！")
        return None

    soup = BeautifulSoup(page, "lxml")
    tbl = soup.find("table", id="tblDetail")
    if tbl is None:
        return {}

    rows = tbl.find_all("tr")
    if len(rows) < 2:
        return {}

    header_cells = [td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])]
    years = []
    for cell in header_cells[1:]:
        m = re.search(r"(\d{4})", cell)
        years.append(int(m.group(1)) if m else None)

    results = {y: {} for y in years if y is not None}

    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if not cells:
            continue
        col_name = _map_column(cells[0])
        if col_name is None:
            continue
        for i, year in enumerate(years):
            if year is None or i + 1 >= len(cells):
                continue
            val = _clean_num(cells[i + 1])
            if val is not None and col_name not in results[year]:
                results[year][col_name] = val

    return results


def fetch_one_stock(driver: webdriver.Chrome, stock_id: str):
    """
    爬一支股票兩段年份。
    回傳 (DataFrame, is_blocked)
    is_blocked=True 代表被封鎖，False 代表正常（有資料或無資料都算正常）
    """
    all_results = {}
    is_blocked = False

    for url_template in GOODINFO_URLS:
        url = url_template.format(stock_id=stock_id)
        partial = _parse_table(driver, url)

        if partial is None:
            # 先確認是封鎖還是 timeout
            if _is_blocked(driver.page_source):
                log.warning(f"  {stock_id} 確認被封鎖！")
                is_blocked = True
            else:
                # timeout 或下市，不算被封
                log.warning(f"  {stock_id} 載入失敗（可能下市或無資料），跳過")
            break

        for year, cols in partial.items():
            if year not in all_results:
                all_results[year] = {}
            all_results[year].update(cols)

        time.sleep(random.uniform(2, 4))

    if is_blocked:
        return pd.DataFrame(), True

    if not all_results:
        log.warning(f"  {stock_id} 無資料")
        return pd.DataFrame(), False

    df = pd.DataFrame.from_dict(all_results, orient="index")
    df.index.name = "年份"
    df = df.sort_index()
    log.info(f"  {stock_id} 完成：{len(df)} 年，{len(df.columns)} 指標")
    return df, False


def fetch_stocks_batch(stock_list: list, year_range: range) -> pd.DataFrame:
    driver = _make_driver()
    all_records = []
    try:
        for i, (stock_id, name) in enumerate(stock_list, 1):
            log.info(f"[{i}/{len(stock_list)}] {stock_id} {name}")
            df, success = fetch_one_stock(driver, stock_id)
            if not df.empty:
                for year in year_range:
                    if year not in df.index:
                        continue
                    row = df.loc[year].to_dict()
                    row["證券代碼"] = stock_id
                    row["簡稱"] = name
                    row["年份"] = year
                    all_records.append(row)
            time.sleep(random.uniform(GOODINFO_SLEEP, GOODINFO_SLEEP + 2))
    finally:
        driver.quit()
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def fetch_stocks_batch_resume(
    stock_list: list,
    year_range: range,
    cache_path: str,
) -> pd.DataFrame:
    """
    支援斷點續爬 + User-Agent 輪換 + 被封自動等待。
    """
    # 讀已有快取
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, dtype={"證券代碼": str})
        done_stocks = set(existing["證券代碼"].tolist())
        log.info(f"  讀取快取：已完成 {len(done_stocks)} 支")
    else:
        existing = pd.DataFrame()
        done_stocks = set()

    todo = [(sid, name) for sid, name in stock_list if sid not in done_stocks]
    log.info(f"  剩餘待爬：{len(todo)} 支")

    if not todo:
        return existing

    all_records = list(existing.to_dict("records")) if not existing.empty else []
    consecutive_fail = 0
    driver = _make_driver()

    try:
        total = len(stock_list)
        done_count = len(done_stocks)

        for stock_id, name in todo:
            done_count += 1
            log.info(f"[{done_count}/{total}] {stock_id} {name}")

            df, is_blocked = fetch_one_stock(driver, stock_id)

            if is_blocked:
                # 確認被封鎖
                consecutive_fail += 1
                log.warning(f"  確認被封鎖 {consecutive_fail} 次")

                if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                    log.warning(f"  暫停 {PAUSE_ON_BLOCK} 秒並換 User-Agent ...")
                    driver.quit()
                    time.sleep(PAUSE_ON_BLOCK)
                    driver = _make_driver()
                    consecutive_fail = 0
                    log.info("  重新建立 Driver，繼續爬取")
            else:
                consecutive_fail = 0  # 正常（有資料或下市）都重設

                if not df.empty:
                    for year in year_range:
                        if year not in df.index:
                            continue
                        row = df.loc[year].to_dict()
                        row["證券代碼"] = stock_id
                        row["簡稱"] = name
                        row["年份"] = year
                        all_records.append(row)

                # 每支爬完立即存快取
                if all_records:
                    pd.DataFrame(all_records).to_csv(
                        cache_path, index=False, encoding="utf-8-sig"
                    )

            # 隨機間隔，模擬人類行為
            time.sleep(random.uniform(GOODINFO_SLEEP, GOODINFO_SLEEP + 3))

            # 每爬 BATCH_SIZE 支就休息一次
            crawled_count = done_count - len(done_stocks)
            if crawled_count > 0 and crawled_count % BATCH_SIZE == 0:
                log.info(f"  已爬 {crawled_count} 支，休息 {BATCH_SLEEP} 秒避免被封...")
                time.sleep(BATCH_SLEEP)
                # 換新 driver 和 User-Agent
                driver.quit()
                driver = _make_driver()
                log.info("  換新 Driver 繼續")

    finally:
        driver.quit()

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


if __name__ == "__main__":
    driver = _make_driver()
    df, _ = fetch_one_stock(driver, "2330")
    driver.quit()
    print(df)