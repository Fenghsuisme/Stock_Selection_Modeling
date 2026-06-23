# -*- coding: utf-8 -*-
"""
AIFT Final Project — Task 2: GA + SVR 選股模型
================================================
方法概念:
  - SVR (Support Vector Regression, RBF kernel) 用「訓練年份」的 16 個財務特徵
    去回歸預測每檔股票「未來一年的報酬 Return」。
  - GA (Genetic Algorithm) 同時優化:
        (1) 特徵子集     -> 16 個二元基因 (要不要用該特徵)
        (2) SVR 超參數   -> C, gamma, epsilon (實數基因, log 空間)
    GA 的適應度 = 在「訓練資料內部」做時間切分驗證所得到的選股組合報酬,
    絕不碰到測試年份, 避免資料洩漏 (data leakage)。
  - 選股 -> 組合報酬: 每個測試年份, 對 200 檔股票預測 Return, 取預測值最高的
    前 N 檔 (等權重), 該年組合報酬 = 被選股票「實際 Return」的平均。

驗證設計 (Temporal Validation, 擴張視窗 expanding window):
  測試年 t 從 1998 跑到 2008; 訓練資料 = 所有 < t 的年份。
  逐年組合報酬串接 -> 累積報酬 / 年化報酬 / 最大回撤 / Sharpe / 周轉率。

只依賴 numpy / pandas / scikit-learn, 方便 demo 時換機器執行。
"""

import numpy as np
import pandas as pd
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ============================================================
# 0. 全域設定 (可調整)
# ============================================================
RANDOM_SEED   = 42
TOP_N         = 10      # 每年選幾檔股票進組合
RF_ANNUAL     = 0.0     # 無風險利率 (年), 計算 Sharpe 用
DISCARD_YM    = 200912  # 依作業要求丟棄的紀錄
TXN_COST      = 0.006   # 單次來回交易成本 (0.6% = 台股證交稅0.3%+手續費約0.3%)

# GA 參數 (demo 想更快可調小 POP/GEN; 想更準可調大)
GA_POP        = 16      # 族群大小
GA_GEN        = 20     # 演化世代數 (從8提高; 論文用70, 想更接近可再調大, 但會更慢)
GA_CX_PROB    = 0.7     # 交配機率
GA_MUT_PROB   = 0.2     # 突變機率
GA_ELITE      = 2       # 菁英保留數

# GA 適應度的「逐年驗證」設定:
#   GA 評估一組參數時, 會在訓練資料內用擴張視窗、一年一年地驗證並取平均。
#   INNER_VAL_YEARS = None 代表用「全部可用年份」逐年驗證 (最完整, 但較慢);
#   設成整數 K (例如 5) 則只用「最近 K 年」逐年驗證 (較快, 結果通常仍接近)。
INNER_VAL_YEARS = None

# SVR 超參數搜尋範圍 (GA 在這些範圍內找最佳值)
C_RANGE       = (-1.0, 2.0)    # 以 10 為底的指數: C     in [10^-1 , 10^2]
GAMMA_RANGE   = (-3.0, 0.0)    # 以 10 為底的指數: gamma in [10^-3 , 10^0]
EPS_RANGE     = (0.01, 0.5)    # epsilon (標準化後的目標單位)

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 1. 資料載入與前處理
# ============================================================
def load_data(path):
    """
    讀檔、前處理、回傳 (df, 特徵欄位名稱 list)。
    自動適應不同資料 (原始 top200 或爬來的資料):
      - 丟棄 年月 = DISCARD_YM 的紀錄 (爬來資料若沒有此年份則無影響)
      - 丟棄 Return 缺值的列 (無法訓練/評估)
      - 特徵 = 除了識別欄與輸出欄以外的所有欄位 (數量自動適應)
      - 丟棄「整欄全空」的特徵 (例如爬蟲沒抓到的欄位)
      - 其餘零星缺值用「同年中位數」補值 (跨年不洩漏), 全年皆空則用全體中位數
    """
    df = pd.read_excel(path)
    df = df[df["年月"] != DISCARD_YM].copy()
    df["year"] = (df["年月"] // 100).astype(int)        # 199712 -> 1997
    df = df.dropna(subset=["Return"]).reset_index(drop=True)

    non_feature = ["證券代碼", "簡稱", "年月", "year", "Return", "ReturnMean_year_Label"]
    feature_cols = [c for c in df.columns if c not in non_feature]
    # 丟掉整欄全空的特徵
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    # 零星缺值: 先用同年中位數補, 再用全體中位數補 (處理整年皆空的情況)
    for c in feature_cols:
        df[c] = df.groupby("year")[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())

    return df, feature_cols


# ============================================================
# 2. GA 個體與適應度
# ============================================================
class Individual:
    """一個 GA 個體 = 特徵遮罩 + 三個 SVR 超參數。"""
    def __init__(self, mask, logC, logG, eps):
        self.mask = mask          # np.array(bool), 長度 = 特徵數
        self.logC = logC          # float
        self.logG = logG          # float
        self.eps  = eps           # float
        self.fitness = -np.inf

    @property
    def params(self):
        return dict(C=10.0 ** self.logC,
                    gamma=10.0 ** self.logG,
                    epsilon=self.eps)


def random_individual(n_feat):
    mask = rng.random(n_feat) < 0.5
    if not mask.any():                 # 至少要選一個特徵
        mask[rng.integers(n_feat)] = True
    logC = rng.uniform(*C_RANGE)
    logG = rng.uniform(*GAMMA_RANGE)
    eps  = rng.uniform(*EPS_RANGE)
    return Individual(mask, logC, logG, eps)


def build_svr(ind):
    p = ind.params
    return SVR(kernel="rbf", C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"])


def portfolio_return_for_year(model, scaler_X, scaler_y,
                              train_year_df, test_year_df, feat, top_n):
    """訓練好的設定下, 在某個 (測試) 年份選股並回傳該年組合報酬。"""
    Xtr = scaler_X.transform(train_year_df[feat].values)
    ytr = scaler_y.transform(train_year_df[["Return"]].values).ravel()
    model.fit(Xtr, ytr)

    Xte = scaler_X.transform(test_year_df[feat].values)
    pred = model.predict(Xte)                       # 標準化空間的預測 (排序用即可)
    order = np.argsort(pred)[::-1][:top_n]          # 預測報酬最高的前 N 檔
    picks = test_year_df.iloc[order]
    return picks["Return"].mean() / 100.0           # 百分比 -> 小數


def evaluate(ind, train_df, feat, top_n):
    """
    GA 適應度: 只在「訓練資料」內部做時間切分驗證。
      - 取訓練資料中最後一個年份當內部驗證年, 其餘較早年份當內部訓練。
      - 適應度 = 內部驗證年的選股組合報酬。
      - 若訓練年份不足 2 年, 退化為 3-fold (非時間) 交叉驗證的負 MSE 排序代理。
    """
    sel = ind.mask
    feat_sel = [f for f, m in zip(feat, sel) if m]
    if len(feat_sel) == 0:
        return -np.inf

    years = sorted(train_df["year"].unique())
    if len(years) >= 2:
        # === 以「一年」為單位的擴張視窗內部驗證 (與整體 TV 同精神) ===
        # 訓練1997 -> 驗證1998; 訓練1997~1998 -> 驗證1999; ... 逐年驗證
        val_years = years[1:]
        if INNER_VAL_YEARS is not None:          # 只用最近 K 個年份當驗證, 控制執行時間
            val_years = val_years[-INNER_VAL_YEARS:]
        scores = []
        for vy in val_years:
            inner_tr = train_df[train_df["year"] < vy]
            inner_va = train_df[train_df["year"] == vy]
            sx = StandardScaler().fit(inner_tr[feat_sel].values)
            sy = StandardScaler().fit(inner_tr[["Return"]].values)
            model = build_svr(ind)
            scores.append(portfolio_return_for_year(model, sx, sy,
                                                    inner_tr, inner_va, feat_sel, top_n))
        return float(np.mean(scores))           # 各「一年單位」組合報酬的平均
    else:
        # 只有 1 個訓練年份(測試年=1998): 無法做逐年驗證, 退化為 3-fold
        d = train_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        fold_idx = np.array_split(np.arange(len(d)), 3)
        scores = []
        for i in range(3):
            va = d.iloc[fold_idx[i]]
            tr = d.iloc[np.concatenate([fold_idx[j] for j in range(3) if j != i])]
            sx = StandardScaler().fit(tr[feat_sel].values)
            sy = StandardScaler().fit(tr[["Return"]].values)
            model = build_svr(ind)
            scores.append(portfolio_return_for_year(model, sx, sy,
                                                    tr, va, feat_sel, top_n))
        return float(np.mean(scores))


# ============================================================
# 3. GA 主迴圈 (在給定訓練資料上找最佳設定)
# ============================================================
def tournament(pop, k=3):
    cand = rng.choice(pop, size=k, replace=False)
    return max(cand, key=lambda ind: ind.fitness)


def crossover(a, b, n_feat):
    """均勻交配: 特徵遮罩逐位互換, 超參數取雙親平均/隨機。"""
    cross_pts = rng.random(n_feat) < 0.5
    mask1 = np.where(cross_pts, a.mask, b.mask)
    mask2 = np.where(cross_pts, b.mask, a.mask)
    for m in (mask1, mask2):
        if not m.any():
            m[rng.integers(n_feat)] = True
    w = rng.random()
    c1 = Individual(mask1, w*a.logC+(1-w)*b.logC, w*a.logG+(1-w)*b.logG, w*a.eps+(1-w)*b.eps)
    c2 = Individual(mask2, w*b.logC+(1-w)*a.logC, w*b.logG+(1-w)*a.logG, w*b.eps+(1-w)*a.eps)
    return c1, c2


def mutate(ind, n_feat):
    # 特徵遮罩: 逐位以小機率翻轉
    flip = rng.random(n_feat) < (1.0 / n_feat)
    ind.mask = np.where(flip, ~ind.mask, ind.mask)
    if not ind.mask.any():
        ind.mask[rng.integers(n_feat)] = True
    # 超參數: 加高斯擾動並夾在範圍內
    ind.logC = float(np.clip(ind.logC + rng.normal(0, 0.3), *C_RANGE))
    ind.logG = float(np.clip(ind.logG + rng.normal(0, 0.3), *GAMMA_RANGE))
    ind.eps  = float(np.clip(ind.eps  + rng.normal(0, 0.05), *EPS_RANGE))
    return ind


def run_ga(train_df, feat, top_n, verbose=False):
    n_feat = len(feat)
    pop = [random_individual(n_feat) for _ in range(GA_POP)]
    for ind in pop:
        ind.fitness = evaluate(ind, train_df, feat, top_n)

    for gen in range(GA_GEN):
        pop.sort(key=lambda i: i.fitness, reverse=True)
        new_pop = pop[:GA_ELITE]                       # 菁英保留
        while len(new_pop) < GA_POP:
            p1, p2 = tournament(pop), tournament(pop)
            if rng.random() < GA_CX_PROB:
                c1, c2 = crossover(p1, p2, n_feat)
            else:
                c1 = Individual(p1.mask.copy(), p1.logC, p1.logG, p1.eps)
                c2 = Individual(p2.mask.copy(), p2.logC, p2.logG, p2.eps)
            if rng.random() < GA_MUT_PROB: c1 = mutate(c1, n_feat)
            if rng.random() < GA_MUT_PROB: c2 = mutate(c2, n_feat)
            c1.fitness = evaluate(c1, train_df, feat, top_n)
            c2.fitness = evaluate(c2, train_df, feat, top_n)
            new_pop.extend([c1, c2])
        pop = new_pop[:GA_POP]
        if verbose:
            best = max(pop, key=lambda i: i.fitness)
            print(f"    gen {gen+1:2d}: best inner return = {best.fitness:+.4f}, "
                  f"#feat={best.mask.sum()}")
    return max(pop, key=lambda i: i.fitness)


# ============================================================
# 4. Temporal Validation 主流程
# ============================================================
def run_temporal_validation(df, feat, top_n=TOP_N, verbose=True,
                            checkpoint_path="ga_svr_checkpoint.csv",
                            picks_path="ga_svr_picks.csv"):
    """
    擴張視窗 TV。支援斷點續跑:
      - 每跑完一個測試年份, 立刻把該年結果寫入 checkpoint_path。
      - 同時把該年「選了哪些股票」寫入 picks_path (每年 top_n 檔的明細)。
      - 重新執行時, 已完成的年份會自動跳過, 從上次停的地方接著跑。
      - 想從頭重跑, 把 checkpoint_path 與 picks_path 兩個檔刪掉即可。
    所以執行中可隨時按 Ctrl+C 中止, 之後再執行就會續跑。
    """
    years = sorted(df["year"].unique())
    test_years = years[1:]                  # 第一年只能當訓練, 從第二年開始測

    # --- 讀取既有 checkpoint, 找出已完成的年份 ---
    records, done_years = [], set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        records = ckpt.to_dict("records")
        done_years = set(ckpt["year"].tolist())
        if verbose and done_years:
            print(f"[續跑] 已完成年份 {sorted(done_years)}, 將跳過這些年份\n")

    # --- 讀取既有選股明細 (續跑時保留先前年份的選股) ---
    pick_records = []
    if picks_path and os.path.exists(picks_path):
        pick_records = pd.read_csv(picks_path).to_dict("records")

    # --- 解釋性輸出的累積容器 (續跑時一併讀回) ---
    EXPLAIN_PATH = "ga_svr_explain.csv"
    explain_records = []
    if os.path.exists(EXPLAIN_PATH):
        explain_records = pd.read_csv(EXPLAIN_PATH).to_dict("records")

    prev_picks = None
    for t in test_years:
        if t in done_years:                 # 已經算過 -> 跳過
            continue

        train_df = df[df["year"] < t]
        test_df  = df[df["year"] == t]

        if verbose:
            print(f"[TV] 測試年 {t} | 訓練年 {train_df['year'].min()}-{t-1} "
                  f"({train_df['year'].nunique()} 年, {len(train_df)} 筆)")

        best = run_ga(train_df, feat, top_n, verbose=verbose)
        feat_sel = [f for f, m in zip(feat, best.mask) if m]

        # 用「全部訓練年份」重新擬合最佳設定, 對測試年選股
        sx = StandardScaler().fit(train_df[feat_sel].values)
        sy = StandardScaler().fit(train_df[["Return"]].values)
        model = build_svr(best)
        model.fit(sx.transform(train_df[feat_sel].values),
                  sy.transform(train_df[["Return"]].values).ravel())

        pred = model.predict(sx.transform(test_df[feat_sel].values))
        order = np.argsort(pred)[::-1][:top_n]
        picks = test_df.iloc[order]
        pred_scores = pred[order]                       # 對應的預測分數(排序用)
        port_ret = picks["Return"].mean() / 100.0
        mkt_ret  = test_df["Return"].mean() / 100.0     # 同年全市場等權當基準
        hit = (picks["ReturnMean_year_Label"] == 1).mean()  # 選到「贏過平均」的比例

        # --- 記錄這一年選了哪些股票 (名次 / 代碼 / 簡稱 / 預測分數 / 實際報酬 / 是否贏過平均) ---
        for rank, ((_, row), score) in enumerate(zip(picks.iterrows(), pred_scores), start=1):
            pick_records.append(dict(
                year=t, rank=rank,
                證券代碼=row["證券代碼"],
                簡稱=str(row["簡稱"]).strip(),
                預測分數=round(float(score), 4),
                實際報酬=round(float(row["Return"]), 2),
                贏過平均=int(row["ReturnMean_year_Label"] == 1)))

        # 周轉率: 與前一年選股相比換掉多少
        pick_ids = set(picks["證券代碼"].tolist())
        if prev_picks is None:
            turnover = np.nan
        else:
            turnover = len(pick_ids - prev_picks) / top_n
        prev_picks = pick_ids

        records.append(dict(year=t, port_return=port_ret, mkt_return=mkt_ret,
                            hit_rate=hit, turnover=turnover,
                            n_feat=int(best.mask.sum()),
                            C=best.params["C"], gamma=best.params["gamma"],
                            epsilon=best.params["epsilon"]))

        # --- 解釋性: 為什麼選這些股票 ---
        # (a) 這年 GA 選用了哪些特徵  (b) 各特徵的 permutation importance
        # (c) 被選股票相對全市場的特徵輪廓 (z 分數: 正=偏高, 負=偏低)
        Xte_full = sx.transform(test_df[feat_sel].values)
        yte = sy.transform(test_df[["Return"]].values).ravel()
        perm = permutation_importance(model, Xte_full, yte,
                                      n_repeats=5, random_state=RANDOM_SEED)
        imp_map = {f: float(v) for f, v in zip(feat_sel, perm.importances_mean)}

        for f in feat:                                  # 16 個特徵都記一列
            used = f in feat_sel
            mu_all = test_df[f].mean()
            sd_all = test_df[f].std(ddof=0)
            z = (picks[f].mean() - mu_all) / sd_all if sd_all > 0 else 0.0
            explain_records.append(dict(
                year=t, feature=f,
                used_by_GA=int(used),
                importance=round(imp_map.get(f, 0.0), 4),
                pick_zscore=round(float(z), 3)))

        # --- 立刻存檔: 即使下一秒被中止, 這一年也不用重算 ---
        if checkpoint_path:
            pd.DataFrame(records).to_csv(checkpoint_path, index=False,
                                         encoding="utf-8-sig")
        if picks_path:
            pd.DataFrame(pick_records).to_csv(picks_path, index=False,
                                              encoding="utf-8-sig")
        pd.DataFrame(explain_records).to_csv(EXPLAIN_PATH, index=False,
                                             encoding="utf-8-sig")

        if verbose:
            print(f"      -> 組合報酬 {port_ret:+.2%} | 市場 {mkt_ret:+.2%} | "
                  f"命中率 {hit:.0%} | 選用特徵 {best.mask.sum()} 個 [已存檔]\n")

    res = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    return res


# ============================================================
# 5. 績效指標
# ============================================================
def performance_summary(res, label="GA+SVR", top_n=TOP_N, txn_cost=0.0):
    """
    報告完整報酬與風險指標。
      txn_cost > 0 時, 會用 (該年周轉率 × txn_cost) 扣掉交易成本, 另外報告淨值結果。
      第一年視為全額建倉, 周轉率以 1.0 計成本。
    """
    r = res["port_return"].values.copy()
    n = len(r)

    # 交易成本調整: 每年成本 = 周轉率 × 來回成本率
    if txn_cost > 0:
        turn = res["turnover"].fillna(1.0).values     # 首年全額建倉
        r = r - turn * txn_cost

    equity = np.cumprod(1 + r)                        # 淨值曲線 (起始 1)
    cum_return = equity[-1] - 1
    ann_return = equity[-1] ** (1 / n) - 1            # 年化 (資料本就是年頻)
    # 最大回撤
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    # 波動率 (年化, 年頻所以即為年報酬標準差)
    volatility = r.std(ddof=1)
    # Sharpe (年頻, rf=RF_ANNUAL)
    excess = r - RF_ANNUAL
    sharpe = excess.mean() / excess.std(ddof=1) if excess.std(ddof=1) > 0 else np.nan
    win_rate = (r > 0).mean()
    avg_turnover = res["turnover"].dropna().mean()

    tag = " [扣交易成本]" if txn_cost > 0 else ""
    print(f"\n===== {label}{tag} 績效總結 (out-of-sample {res['year'].min()}-{res['year'].max()}) =====")
    print(f"  年化報酬 (Annualized return)   : {ann_return:+.2%}")
    print(f"  累積報酬 (Cumulative return)   : {cum_return:+.2%}")
    print(f"  最大回撤 (Maximum drawdown)    : {mdd:.2%}")
    print(f"  波動率   (Volatility, 年化)     : {volatility:.2%}")
    print(f"  Sharpe ratio                  : {sharpe:.3f}")
    print(f"  勝率   (正報酬年份比例)         : {win_rate:.0%}")
    print(f"  平均周轉率 (Turnover)          : {avg_turnover:.0%}")
    print(f"  選股數量 (每年持有檔數)         : {top_n}")
    print(f"  平均每年命中率 (選到贏過平均者) : {res['hit_rate'].mean():.0%}")
    return dict(annualized=ann_return, cumulative=cum_return, mdd=mdd,
                volatility=volatility, sharpe=sharpe, win_rate=win_rate,
                turnover=avg_turnover, n_stocks=top_n)


# ============================================================
# 5c. 作業指定的 TV 方案: 訓練前 k 年 -> 測試其餘所有年份
# ============================================================
def _series_metrics(rets):
    """給一串「逐年報酬」, 算年化/累積/最大回撤/波動/Sharpe/勝率。"""
    rets = np.asarray(rets, dtype=float)
    n = len(rets)
    equity = np.cumprod(1 + rets)
    cum = equity[-1] - 1
    ann = equity[-1] ** (1 / n) - 1
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak).min()
    vol = rets.std(ddof=1) if n > 1 else np.nan
    sharpe = (rets.mean() / vol) if (n > 1 and vol > 0) else np.nan
    win = (rets > 0).mean()
    return dict(ann=ann, cum=cum, mdd=mdd, vol=vol, sharpe=sharpe, win=win)


def run_tv_validation(df, feat, top_n=TOP_N, verbose=True,
                      checkpoint_path="ga_svr_tv_checkpoint.csv",
                      detail_path="ga_svr_tv_detail.csv",
                      criteria_path="ga_svr_tv_criteria.csv"):
    """
    作業 TV 表格的標準切法 (擴張訓練 / 收縮測試):
      TV1 : 訓練 第1年       -> 測試 其餘所有年份
      TV2 : 訓練 前2年       -> 測試 其餘所有年份
      ...
      TVk : 訓練 前k年       -> 測試 其餘所有年份  (k = 1 .. 年份數-1)
    每個 TV:
      - 在「訓練年份」上跑 GA 找最佳 SVR 設定 (GA 內部仍只用訓練年逐年驗證, 無洩漏)
      - 用最佳設定在「測試期間」逐年選 top_n, 串接出該 TV 的績效
      - 另記錄該 TV 的「選股標準」: 選了哪些指標、各指標權重(正規化的重要度)、SVR參數
    回傳: 每個 TV 一列的績效表。支援以 TV 為單位的斷點續跑。
    """
    years = sorted(df["year"].unique())
    n = len(years)

    tv_records, detail_records, crit_records, done_tv = [], [], [], set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = pd.read_csv(checkpoint_path)
        tv_records = ck.to_dict("records")
        done_tv = set(ck["tv"].tolist())
        if verbose and done_tv:
            print(f"[續跑] 已完成 TV {sorted(done_tv)}, 將跳過\n")
    if detail_path and os.path.exists(detail_path):
        detail_records = pd.read_csv(detail_path).to_dict("records")
    if criteria_path and os.path.exists(criteria_path):
        crit_records = pd.read_csv(criteria_path).to_dict("records")

    for k in range(1, n):                       # k = 訓練年數
        if k in done_tv:
            continue
        train_years = years[:k]
        test_years  = years[k:]
        train_df = df[df["year"].isin(train_years)]

        if verbose:
            print(f"[TV{k}] 訓練 {train_years[0]}-{train_years[-1]} ({k}年) | "
                  f"測試 {test_years[0]}-{test_years[-1]} ({len(test_years)}年)")

        best = run_ga(train_df, feat, top_n, verbose=verbose)
        feat_sel = [f for f, m in zip(feat, best.mask) if m]
        sx = StandardScaler().fit(train_df[feat_sel].values)
        sy = StandardScaler().fit(train_df[["Return"]].values)
        model = build_svr(best)
        model.fit(sx.transform(train_df[feat_sel].values),
                  sy.transform(train_df[["Return"]].values).ravel())

        # 在測試期間逐年選股
        rets, turns, prev_ids = [], [], None
        for ty in test_years:
            tdf = df[df["year"] == ty]
            pred = model.predict(sx.transform(tdf[feat_sel].values))
            picks = tdf.iloc[np.argsort(pred)[::-1][:top_n]]
            pr = picks["Return"].mean() / 100.0
            mr = tdf["Return"].mean() / 100.0
            ids = set(picks["證券代碼"].tolist())
            tn = np.nan if prev_ids is None else len(ids - prev_ids) / top_n
            prev_ids = ids
            rets.append(pr)
            turns.append(tn)
            detail_records.append(dict(tv=k, year=ty,
                                       port_return=round(pr, 4),
                                       mkt_return=round(mr, 4)))

        m = _series_metrics(rets)
        tv_records.append(dict(
            tv=k,
            train_years=f"{train_years[0]}-{train_years[-1]}",
            n_train=k,
            test_years=f"{test_years[0]}-{test_years[-1]}",
            n_test=len(test_years),
            ann_return=round(m["ann"], 4),
            cum_return=round(m["cum"], 4),
            mdd=round(m["mdd"], 4),
            volatility=round(m["vol"], 4) if not np.isnan(m["vol"]) else np.nan,
            sharpe=round(m["sharpe"], 3) if not np.isnan(m["sharpe"]) else np.nan,
            win_rate=round(m["win"], 3),
            avg_turnover=round(float(np.nanmean(turns)), 3) if len(turns) > 1 else np.nan,
            n_feat=int(best.mask.sum()),
            C=round(best.params["C"], 4),
            gamma=round(best.params["gamma"], 5),
            epsilon=round(best.params["epsilon"], 4)))

        # === 該 TV 的「選股標準」: 各選用指標的權重 (正規化的 permutation importance) ===
        test_df = df[df["year"].isin(test_years)]
        Xt = sx.transform(test_df[feat_sel].values)
        yt = sy.transform(test_df[["Return"]].values).ravel()
        perm = permutation_importance(model, Xt, yt, n_repeats=5,
                                      random_state=RANDOM_SEED)
        imp = np.clip(perm.importances_mean, 0, None)        # 負的(沒幫助)歸零
        weights = imp / imp.sum() if imp.sum() > 0 else np.ones(len(feat_sel)) / len(feat_sel)
        order = np.argsort(weights)[::-1]
        for rank, idx in enumerate(order, start=1):
            crit_records.append(dict(
                tv=k, train_years=f"{train_years[0]}-{train_years[-1]}",
                rank=rank, 指標=feat_sel[idx],
                權重=round(float(weights[idx]), 4),
                重要度=round(float(perm.importances_mean[idx]), 4),
                C=round(best.params["C"], 4),
                gamma=round(best.params["gamma"], 5),
                epsilon=round(best.params["epsilon"], 4)))

        # 每完成一個 TV 就存檔 (可中斷續跑)
        if checkpoint_path:
            pd.DataFrame(tv_records).to_csv(checkpoint_path, index=False,
                                            encoding="utf-8-sig")
        if detail_path:
            pd.DataFrame(detail_records).to_csv(detail_path, index=False,
                                                encoding="utf-8-sig")
        if criteria_path:
            pd.DataFrame(crit_records).to_csv(criteria_path, index=False,
                                              encoding="utf-8-sig")
        if verbose:
            print(f"      -> 測試期 年化 {m['ann']:+.2%} | 累積 {m['cum']:+.2%} | "
                  f"回撤 {m['mdd']:.2%} | Sharpe {m['sharpe'] if np.isnan(m['sharpe']) else round(m['sharpe'],2)} "
                  f"[已存檔]\n")

    return pd.DataFrame(tv_records).sort_values("tv").reset_index(drop=True)


def print_tv_table(tv):
    """把每個 TV 的驗證結果印成整齊表格。"""
    print("\n===== TV 驗證結果 (每個 TV = 訓練前 k 年, 測試其餘年份) =====")
    hdr = (f"{'TV':>2} {'訓練年':>9} {'測試年':>9} {'年化':>8} "
           f"{'累積':>9} {'最大回撤':>9} {'波動':>7} {'Sharpe':>7} {'勝率':>6}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in tv.iterrows():
        sh = "  n/a" if pd.isna(r["sharpe"]) else f"{r['sharpe']:>6.2f}"
        vo = "  n/a" if pd.isna(r["volatility"]) else f"{r['volatility']:>6.1%}"
        print(f"{int(r['tv']):>2} {r['train_years']:>9} {r['test_years']:>9} "
              f"{r['ann_return']:>+7.1%} {r['cum_return']:>+8.1%} {r['mdd']:>8.1%} "
              f"{vo} {sh} {r['win_rate']:>5.0%}")
    print("-" * len(hdr))
    print(f"各 TV 平均: 年化 {tv['ann_return'].mean():+.2%} | "
          f"最大回撤 {tv['mdd'].mean():.2%} | "
          f"Sharpe {tv['sharpe'].dropna().mean():.2f}")


def print_tv_criteria(criteria_path="ga_svr_tv_criteria.csv"):
    """列出每個 TV 的選股標準: 選用指標、各指標權重、SVR 參數。"""
    if not os.path.exists(criteria_path):
        print("（找不到 ga_svr_tv_criteria.csv）")
        return
    c = pd.read_csv(criteria_path)
    print("\n===== 每個 TV 的選股標準 (選用指標 + 權重 + SVR 參數) =====")
    for tv_id, g in c.groupby("tv"):
        g = g.sort_values("rank")
        ty = g["train_years"].iloc[0]
        C, gm, ep = g["C"].iloc[0], g["gamma"].iloc[0], g["epsilon"].iloc[0]
        print(f"\nTV{int(tv_id)} (訓練 {ty}) | SVR: C={C}, gamma={gm}, epsilon={ep} | "
              f"選用 {len(g)} 個指標:")
        for _, r in g.iterrows():
            bar = "█" * max(1, int(round(r["權重"] * 30)))
            print(f"   {r['權重']:>6.1%}  {bar:<30} {r['指標']}")


# ============================================================
# 5b. 解釋性總結: 為什麼選這些股票
# ============================================================
def explain_summary(explain_path="ga_svr_explain.csv", n_years=None):
    """彙整各年的特徵使用、重要度與選股輪廓, 解釋模型的選股邏輯。"""
    if not os.path.exists(explain_path):
        print("（找不到 ga_svr_explain.csv, 先跑一次 TV）")
        return
    e = pd.read_csv(explain_path)
    if n_years is None:
        n_years = e["year"].nunique()

    agg = (e.groupby("feature")
             .agg(被選用年數=("used_by_GA", "sum"),
                  平均重要度=("importance", "mean"),
                  平均選股z分數=("pick_zscore", "mean"))
             .reset_index())
    agg["被選用比例"] = agg["被選用年數"] / n_years

    print("\n===== 模型選股邏輯 (為什麼選這些股票) =====")
    print(f"  共 {n_years} 個測試年份。下表彙整 16 個特徵:")

    print("\n[1] GA 最常選用的特徵 (依被選用年數排序):")
    top_used = agg.sort_values("被選用年數", ascending=False).head(8)
    for _, r in top_used.iterrows():
        print(f"    {r['feature']:<14} 用了 {int(r['被選用年數'])}/{n_years} 年 "
              f"({r['被選用比例']:.0%})")

    print("\n[2] 對預測影響最大的特徵 (平均 permutation importance):")
    top_imp = agg.sort_values("平均重要度", ascending=False).head(8)
    for _, r in top_imp.iterrows():
        print(f"    {r['feature']:<14} 重要度 {r['平均重要度']:+.3f}")

    print("\n[3] 被選股票的特徵輪廓 (相對全市場的平均 z 分數, 正=偏高/負=偏低):")
    prof = agg.reindex(agg["平均選股z分數"].abs().sort_values(ascending=False).index).head(8)
    for _, r in prof.iterrows():
        direction = "偏高" if r["平均選股z分數"] > 0 else "偏低"
        print(f"    {r['feature']:<14} z={r['平均選股z分數']:+.2f} ({direction})")

    return agg


# ============================================================
# 6. Demo 用: 在新測試檔上預測 (同格式)
# ============================================================
def train_full_and_predict(train_path, new_test_path, top_n=TOP_N):
    """
    用全部歷史資料 (train_path) 找最佳 GA+SVR 設定, 對 demo 新資料 (new_test_path)
    選股並回報組合報酬。new_test_path 需與訓練資料同欄位格式。
    """
    df, feat = load_data(train_path)
    best = run_ga(df, feat, top_n, verbose=False)
    feat_sel = [f for f, m in zip(feat, best.mask) if m]

    sx = StandardScaler().fit(df[feat_sel].values)
    sy = StandardScaler().fit(df[["Return"]].values)
    model = build_svr(best)
    model.fit(sx.transform(df[feat_sel].values),
              sy.transform(df[["Return"]].values).ravel())

    new_df, _ = load_data(new_test_path)
    out = []
    for yr, g in new_df.groupby("year"):
        pred = model.predict(sx.transform(g[feat_sel].values))
        picks = g.iloc[np.argsort(pred)[::-1][:top_n]]
        out.append(dict(year=yr,
                        port_return=picks["Return"].mean()/100.0,
                        picks=picks["簡稱"].str.strip().tolist()))
    return pd.DataFrame(out), best, feat_sel


# ============================================================
# 6b. 存模型 / 載入模型直接預測 (Demo 用, 免重新訓練)
# ============================================================
def train_and_save_model(data_path, model_path="ga_svr_model.pkl", top_n=TOP_N):
    """
    用「全部資料」跑一次 GA 找最佳設定, 訓練 SVR, 把整組模型存成檔。
    存的內容: 選用特徵、兩個 scaler、訓練好的 SVR、top_n、來源與特徵清單。
    Demo 前先執行一次, 現場就能用 predict_with_saved_model 秒出結果。
    """
    import joblib
    df, feat = load_data(data_path)
    print(f"用全部資料訓練最終模型: {data_path} ({df.shape[0]} 筆, {len(feat)} 特徵)")
    best = run_ga(df, feat, top_n, verbose=True)
    feat_sel = [f for f, m in zip(feat, best.mask) if m]

    sx = StandardScaler().fit(df[feat_sel].values)
    sy = StandardScaler().fit(df[["Return"]].values)
    model = build_svr(best)
    model.fit(sx.transform(df[feat_sel].values),
              sy.transform(df[["Return"]].values).ravel())

    bundle = dict(feat_sel=feat_sel, scaler_X=sx, scaler_y=sy, svr=model,
                  top_n=top_n, source=data_path,
                  params=best.params, all_features=feat)
    joblib.dump(bundle, model_path)
    print(f"模型已存檔: {model_path}")
    print(f"  選用 {len(feat_sel)} 個特徵 | C={best.params['C']:.3f}, "
          f"gamma={best.params['gamma']:.4f}, epsilon={best.params['epsilon']:.3f}")
    return model_path


def predict_with_saved_model(new_data_path, model_path="ga_svr_model.pkl",
                             out_path="ga_svr_demo_picks.csv"):
    """
    載入已存好的模型 (不重新訓練), 對新資料逐年選股。
      - 新資料需含模型所用的特徵欄位 (同格式即可)。
      - 若新資料有 Return 欄, 會一併算出每年組合報酬與整體績效。
    """
    import joblib
    if not os.path.exists(model_path):
        print(f"[錯誤] 找不到模型檔 {model_path}, 請先執行 train_and_save_model 或 'save' 模式。")
        return None
    b = joblib.load(model_path)
    feat_sel, sx, sy, model, top_n = (b["feat_sel"], b["scaler_X"],
                                      b["scaler_y"], b["svr"], b["top_n"])
    print(f"已載入模型 {model_path} (用 {b['source']} 訓練, {len(feat_sel)} 個特徵)")

    new_df, _ = load_data(new_data_path)
    missing = [f for f in feat_sel if f not in new_df.columns]
    if missing:
        print(f"[錯誤] 新資料缺少模型需要的特徵: {missing}")
        return None

    pick_rows, yearly = [], []
    has_ret = "Return" in new_df.columns
    has_lbl = "ReturnMean_year_Label" in new_df.columns
    prev_ids = None
    for yr, g in new_df.groupby("year"):
        pred = model.predict(sx.transform(g[feat_sel].values))
        picks = g.iloc[np.argsort(pred)[::-1][:top_n]]
        pr = picks["Return"].mean()/100.0 if has_ret else np.nan
        hit = (picks["ReturnMean_year_Label"] == 1).mean() if has_lbl else np.nan
        ids = set(picks["證券代碼"].tolist())
        tn = np.nan if prev_ids is None else len(ids - prev_ids) / top_n
        prev_ids = ids
        yearly.append(dict(year=yr, port_return=pr, turnover=tn, hit_rate=hit))
        for rank, (_, row) in enumerate(picks.iterrows(), start=1):
            pick_rows.append(dict(year=yr, rank=rank,
                                  證券代碼=row["證券代碼"],
                                  簡稱=str(row["簡稱"]).strip(),
                                  實際報酬=round(float(row["Return"]), 2) if has_ret else np.nan))
    picks_df = pd.DataFrame(pick_rows)
    picks_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"選股結果已存檔: {out_path}")

    print("\n===== 每年選股 =====")
    for yr, g in picks_df.groupby("year"):
        names = "、".join(g.sort_values("rank")["簡稱"].astype(str))
        line = f"{yr}: {names}"
        if has_ret:
            pr = [y["port_return"] for y in yearly if y["year"] == yr][0]
            line += f"   (組合報酬 {pr:+.2%})"
        print(line)

    if has_ret:
        ydf = pd.DataFrame(yearly)
        performance_summary(ydf, "GA+SVR (載入模型)", top_n=top_n)
    return picks_df


# ============================================================
# 7. 進入點
# ============================================================
if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    # 三種使用方式:
    #   1) python gasvr.py <資料檔>            -> 跑 TV 驗證 (預設)
    #   2) python gasvr.py <資料檔> save       -> 用全部資料訓練並存模型 ga_svr_model.pkl
    #   3) python gasvr.py <新資料檔> predict  -> 載入 ga_svr_model.pkl, 不重訓直接選股
    mode = "tv"
    if "save" in args:
        mode = "save"; args.remove("save")
    elif "predict" in args:
        mode = "predict"; args.remove("predict")
    DATA = args[0] if args else "top200.xlsx"

    if mode == "save":
        train_and_save_model(DATA, model_path="ga_svr_model.pkl", top_n=TOP_N)
        sys.exit(0)
    if mode == "predict":
        predict_with_saved_model(DATA, model_path="ga_svr_model.pkl")
        sys.exit(0)

    # ---- 預設: TV 驗證 ----
    df, feat = load_data(DATA)
    print(f"資料來源: {DATA}")
    print(f"資料: {df.shape[0]} 筆, {df['year'].nunique()} 個年份, {len(feat)} 個特徵")
    print(f"特徵: {feat}\n")

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    # ===== 作業指定的 TV 驗證: 訓練前 k 年 -> 測試其餘年份, k=1..11 =====
    tv = run_tv_validation(df, feat, top_n=TOP_N, verbose=True)
    print_tv_table(tv)
    print_tv_criteria("ga_svr_tv_criteria.csv")     # 每個 TV 的選股標準 + 權重
    tv.to_csv("ga_svr_tv_results.csv", index=False, encoding="utf-8-sig")
    print("\nTV 驗證結果已存檔: ga_svr_tv_results.csv")
    print("各 TV 測試期逐年明細已存檔: ga_svr_tv_detail.csv")
    print("各 TV 選股標準與指標權重已存檔: ga_svr_tv_criteria.csv")