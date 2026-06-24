# -*- coding: utf-8 -*-
"""
用「存好的模型」在一份測試資料上選股, 並跟「同期大盤」比較、畫圖。
讀模型 (.pkl) + 測試檔 (.xlsx), 自己算逐年報酬, 不依賴其他 CSV。

用法 (在有 gasvr.py / model_200.pkl / 測試檔 的資料夾執行):
  python compare_test.py top200_testing.xlsx --model model_200.pkl
  python compare_test.py top200_testing.xlsx --model model_200.pkl --out demo_compare.png
需要: pip install matplotlib joblib
"""
import sys, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import gasvr as G                      # 借用 load_data 的清理/補值邏輯

# 中文字型
for fp in ["C:/Windows/Fonts/msjh.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=fp).get_name()]
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def metrics(rets):
    r = np.asarray(rets, float); n = len(r)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    vol = r.std(ddof=1) if n > 1 else np.nan
    return dict(ann=eq[-1]**(1/n)-1, cum=eq[-1]-1,
                mdd=((eq-peak)/peak).min(),
                sharpe=(r.mean()/vol) if (n > 1 and vol > 0) else np.nan,
                win=(r > 0).mean())


def main(test_path, model_path="ga_svr_model.pkl", out_path="compare_test.png"):
    b = joblib.load(model_path)
    feat_sel, sx, sy, model, top_n = (b["feat_sel"], b["scaler_X"],
                                      b["scaler_y"], b["svr"], b["top_n"])
    print(f"已載入模型 {model_path} (用 {b['source']} 訓練, {len(feat_sel)} 特徵)")

    df, _ = G.load_data(test_path)
    missing = [f for f in feat_sel if f not in df.columns]
    if missing:
        print(f"[錯誤] 測試檔缺少模型需要的特徵: {missing}"); return

    rows = []
    for yr, g in df.groupby("year"):
        pred = model.predict(sx.transform(g[feat_sel].values))
        picks = g.iloc[np.argsort(pred)[::-1][:top_n]]
        rows.append(dict(year=yr,
                         model=picks["Return"].mean()/100.0,   # 模型選股報酬
                         market=g["Return"].mean()/100.0))     # 同期大盤(等權)
    d = pd.DataFrame(rows).sort_values("year")
    years = d["year"].values
    mm, bm = metrics(d["model"].values), metrics(d["market"].values)

    # 螢幕比較表
    print(f"\n===== 模型 vs 大盤 ({years[0]}-{years[-1]}, 共 {len(years)} 年) =====")
    print(f"{'指標':<12}{'模型':>12}{'大盤':>12}{'差距':>12}")
    print("-" * 48)
    for name, mv, bv in [("年化報酬", mm["ann"], bm["ann"]),
                         ("累積報酬", mm["cum"], bm["cum"]),
                         ("最大回撤", mm["mdd"], bm["mdd"]),
                         ("Sharpe ", mm["sharpe"], bm["sharpe"]),
                         ("勝率   ", mm["win"], bm["win"])]:
        if "Sharpe" in name:
            print(f"{name:<12}{mv:>12.2f}{bv:>12.2f}{mv-bv:>+12.2f}")
        else:
            print(f"{name:<12}{mv:>+12.1%}{bv:>+12.1%}{mv-bv:>+12.1%}")

    # 畫圖: 左=累積報酬曲線, 右=逐年報酬長條
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1.plot(years, (np.cumprod(1+d["model"].values)-1)*100, "o-", lw=2, label="模型選股")
    ax1.plot(years, (np.cumprod(1+d["market"].values)-1)*100, "s--", lw=2, label="大盤(等權)")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_title(f"累積報酬：模型 vs 大盤 ({years[0]}-{years[-1]})")
    ax1.set_xlabel("年份"); ax1.set_ylabel("累積報酬 (%)")
    ax1.legend(); ax1.grid(alpha=0.3)

    w = 0.38
    ax2.bar(years - w/2, d["model"].values*100, w, label="模型選股")
    ax2.bar(years + w/2, d["market"].values*100, w, label="大盤(等權)")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("逐年報酬：模型 vs 大盤")
    ax2.set_xlabel("年份"); ax2.set_ylabel("當年報酬 (%)")
    ax2.set_xticks(years); ax2.legend(); ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    print(f"\n已輸出 {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    model = "ga_svr_model.pkl"; out = "compare_test.png"
    if "--model" in args:
        i = args.index("--model"); model = args[i+1]; del args[i:i+2]
    if "--out" in args:
        i = args.index("--out"); out = args[i+1]; del args[i:i+2]
    test = args[0] if args else "top200_testing.xlsx"
    main(test, model, out)