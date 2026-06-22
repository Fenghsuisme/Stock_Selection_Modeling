# processors/ratio_calculator.py
# 從損益表、資產負債表、現金流量表的原始數字，計算 34 個財務指標

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


def _safe_div(a, b):
    """安全除法，分母為0或None回傳None"""
    try:
        if a is None or b is None or b == 0:
            return None
        return a / b
    except Exception:
        return None


def _g(df, year, col):
    """從 wide-format df 取某年某欄位的值，不存在回傳 None"""
    try:
        if year in df.index and col in df.columns:
            v = df.loc[year, col]
            return float(v) if pd.notna(v) else None
    except Exception:
        pass
    return None


def calc_ratios_for_stock(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    per: pd.DataFrame,
    year: int,
) -> dict:
    """
    計算單一股票單一年度的 34 個指標。
    回傳 {指標名稱: 值}
    """
    r = {}

    # ── 損益表項目 ──
    revenue   = _g(income, year, "Revenue")
    gross     = _g(income, year, "GrossProfit")
    op_income = _g(income, year, "OperatingIncome")
    net       = _g(income, year, "NetIncome") \
                or _g(income, year, "IncomeAfterTaxes") \
                or _g(income, year, "TotalConsolidatedProfitForThePeriod")
    cogs      = _g(income, year, "CostOfGoodsSold")
    eps       = _g(income, year, "EPS")
    pretax    = _g(income, year, "IncomeBeforeIncomeTax") or _g(income, year, "PreTaxIncome")

    # ── 資產負債表項目 ──
    cur_assets = _g(balance, year, "CurrentAssets")
    cur_liab   = _g(balance, year, "CurrentLiabilities")
    cash       = _g(balance, year, "CashAndCashEquivalents")
    inventory  = _g(balance, year, "Inventories") or _g(balance, year, "Inventory")
    ar         = _g(balance, year, "AccountsReceivableNet")
    ap         = _g(balance, year, "AccountsPayable")
    total_assets = _g(balance, year, "TotalAssets")
    total_liab_equity = _g(balance, year, "TotalLiabilitiesEquity")  # 負債+權益總額
    equity     = _g(balance, year, "Equity") \
                 or _g(balance, year, "EquityAttributableToOwnersOfParent")
    # 總負債 = (負債+權益) − 權益
    total_liab = (total_liab_equity - equity) \
                 if (total_liab_equity is not None and equity is not None) else None
    fixed_assets = _g(balance, year, "PropertyPlantAndEquipment")
    capital    = _g(balance, year, "CapitalStock")  # 股本（面額10元）

    # 股數 = 股本 / 10（面額10元）
    shares = _safe_div(capital, 10) if capital else None

    # ── 現金流量表項目 ──
    op_cashflow = _g(cashflow, year, "NetCashInflowFromOperatingActivities") \
                  or _g(cashflow, year, "CashFlowsFromOperatingActivities")

    # ── 前一年（算成長率用）──
    revenue_prev   = _g(income, year - 1, "Revenue")
    op_income_prev = _g(income, year - 1, "OperatingIncome")
    net_prev       = _g(income, year - 1, "NetIncome") \
                     or _g(income, year - 1, "IncomeAfterTaxes") \
                     or _g(income, year - 1, "TotalConsolidatedProfitForThePeriod")
    total_assets_prev = _g(balance, year - 1, "TotalAssets")
    op_cashflow_prev  = _g(cashflow, year - 1, "NetCashInflowFromOperatingActivities") \
                        or _g(cashflow, year - 1, "CashFlowsFromOperatingActivities")

    # ── PER 資料集 ──
    pbr = _g(per, year, "PBR")
    pe  = _g(per, year, "PER")
    div_yield = _g(per, year, "dividend_yield")

    # ════════════════════════════════════════════
    # 原始16個指標（市值、收盤價在 processor 補）
    # ════════════════════════════════════════════
    r["股價淨值比"]        = pbr
    r["M淨值報酬率─稅後"]  = _pct(_safe_div(net, equity))           # ROE
    r["資產報酬率ROA"]     = _pct(_safe_div(net, total_assets))     # ROA
    r["營業利益率OPM"]     = _pct(_safe_div(op_income, revenue))
    r["利潤邊際NPM"]       = _pct(_safe_div(net, revenue))
    r["負債/淨值比"]       = _pct(_safe_div(total_liab, equity))
    r["M流動比率"]         = _pct(_safe_div(cur_assets, cur_liab))
    r["M速動比率"]         = _pct(_safe_div(
        (cur_assets - inventory) if (cur_assets and inventory) else cur_assets, cur_liab))
    r["M存貨週轉率 (次)"]  = _safe_div(cogs, inventory)
    r["M應收帳款週轉次"]   = _safe_div(revenue, ar)
    r["M營業利益成長率"]   = _growth(op_income, op_income_prev)
    r["M稅後淨利成長率"]   = _growth(net, net_prev)
    # 股價營收比 = 市值/營收，市值在 processor 補，這裡先放 revenue 給後面算
    r["_revenue"]          = revenue

    # ════════════════════════════════════════════
    # 額外18個指標
    # ════════════════════════════════════════════
    r["毛利率"]            = _pct(_safe_div(gross, revenue))
    r["EPS"]               = eps
    r["每股淨值BPS"]       = _safe_div(equity, shares)
    r["現金比率"]          = _pct(_safe_div(cash, cur_liab))
    r["利息保障倍數"]      = None  # FinMind 無利息費用明細，留空
    r["總資產成長率"]      = _growth(total_assets, total_assets_prev)
    r["負債比率"]          = _pct(_safe_div(total_liab, total_assets))
    r["殖利率"]            = div_yield
    r["本益比PE"]          = pe
    r["營收成長率"]        = _growth(revenue, revenue_prev)
    r["固定資產週轉率"]    = _safe_div(revenue, fixed_assets)
    r["總資產週轉率"]      = _safe_div(revenue, total_assets)
    r["權益乘數"]          = _safe_div(total_assets, equity)
    r["每股現金流量"]      = _safe_div(op_cashflow, shares)
    r["現金流量比率"]      = _pct(_safe_div(op_cashflow, cur_liab))
    r["營業現金流量成長率"] = _growth(op_cashflow, op_cashflow_prev)
    r["應付帳款週轉率"]    = _safe_div(cogs, ap)
    r["稅後淨利"]          = net

    return r


def _pct(v):
    """轉成百分比（×100）"""
    return round(v * 100, 4) if v is not None else None


def _growth(cur, prev):
    """成長率 %"""
    if cur is None or prev is None or prev == 0:
        return None
    return round((cur - prev) / abs(prev) * 100, 4)