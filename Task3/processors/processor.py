# processors/processor.py — 整合資料、算 Return/Label、格式對齊

import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import ALL_COLUMNS

log = logging.getLogger(__name__)


def calculate_label(df: pd.DataFrame) -> pd.DataFrame:
    """每年高於平均 Return → 1，否則 → -1"""
    df = df.copy()
    yearly_mean = df.groupby("年份")["Return"].transform("mean")
    df["ReturnMean_year_Label"] = np.where(df["Return"] >= yearly_mean, 1, -1)
    return df


def build_final(records: list) -> pd.DataFrame:
    """
    records: 每筆是一個 dict，含證券代碼、簡稱、年份、市值、收盤價、Return、34指標
    回傳對齊原始 xlsx 的 DataFrame。
    """
    df = pd.DataFrame(records)

    if df.empty:
        log.warning("無任何資料")
        return df

    # 股價營收比 = 市值 / 營收
    if "_revenue" in df.columns:
        df["股價營收比"] = df.apply(
            lambda x: (x["市值(百萬元)"] * 1_000_000 / x["_revenue"])
            if x.get("_revenue") and x["_revenue"] != 0 else None,
            axis=1,
        )
        df = df.drop(columns=["_revenue"])

    # 算 Label（需要 Return 完整）
    df = df[df["Return"].notna()].copy()
    df = calculate_label(df)

    # 年月
    df["年月"] = (df["年份"].astype(str) + "12").astype(int)

    # Unknown masked parameter
    df["Unknown masked parameter"] = np.nan

    # 確保所有欄位存在
    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[ALL_COLUMNS].copy()
    df = df.sort_values(["年月", "市值(百萬元)"], ascending=[True, False])
    df = df.reset_index(drop=True)

    log.info(f"最終資料：{len(df)} 筆，{len(df.columns)} 欄")
    return df


def save_to_excel(df: pd.DataFrame, path: str) -> None:
    df.to_excel(path, index=False, engine="openpyxl")
    log.info(f"已儲存：{path}")