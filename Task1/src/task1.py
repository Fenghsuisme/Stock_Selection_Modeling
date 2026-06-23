import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 💡 新增：用於繪製圖表的套件

# =====================================================================
# 1. 自訂連續型 ID3 決策樹節點與分類器 (Custom ID3 for Continuous Data)
# =====================================================================
class ID3Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature = feature      # 切分的財務特徵欄位名稱
        self.threshold = threshold  # 連續變數的切分臨界值 (Threshold)
        self.left = left            # 左子樹 (滿足 <= threshold)
        self.right = right          # 右子樹 (不滿足 > threshold)
        self.value = value          # 若為葉節點，代表預測類別 (1: 打敗大盤, -1: 未打敗大盤)
        self.is_leaf = is_leaf

class ContinuousID3TreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _calculate_entropy(self, y):
        """計算資訊熵 (Entropy)"""
        if len(y) == 0:
            return 0
        p_pos = np.sum(y == 1) / len(y)
        p_neg = np.sum(y == -1) / len(y)

        entropy = 0
        for p in [p_pos, p_neg]:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _best_split(self, X, y):
        """尋找能帶來最大 Information Gain 的組合"""
        best_gain = -1
        split_idx, split_thresh = None, None
        current_entropy = self._calculate_entropy(y)

        for col in X.columns:
            X_column = X[col].values
            unique_vals = np.sort(np.unique(X_column))
            if len(unique_vals) <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue

                entropy_left = self._calculate_entropy(y[left_mask])
                entropy_right = self._calculate_entropy(y[right_mask])
                w_left = np.sum(left_mask) / len(y)
                w_right = np.sum(right_mask) / len(y)
                child_entropy = w_left * entropy_left + w_right * entropy_right

                info_gain = current_entropy - child_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx = col
                    split_thresh = thresh

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            leaf_val = 1 if np.sum(y == 1) >= np.sum(y == -1) else -1
            return ID3Node(value=leaf_val, is_leaf=True)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            leaf_val = 1 if np.sum(y == 1) >= np.sum(y == -1) else -1
            return ID3Node(value=leaf_val, is_leaf=True)

        left_mask = X[feature] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return ID3Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, row):
        if node.is_leaf:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        else:
            return self._predict_row(node.right, row)

    def predict(self, X):
        return np.array([self._predict_row(self.root, row) for _, row in X.iterrows()])

    def print_tree(self):
        """公開調用的印樹接口"""
        self._export_tree_structure(self.root, depth=0, prefix="")

    # 💡 優化：修正原本重複列印與對齊問題，清晰標記出【是 <=】與【否 >】的分支流向
    def _export_tree_structure(self, node, depth=0, prefix=""):
        """遞迴走訪樹節點並清晰列印到終端機"""
        indent = "    " * depth
        
        if node.is_leaf:
            label_text = "🎯 買進 ( 1 )" if node.value == 1 else "❌ 淘汰 (-1)"
            print(f"{indent}{prefix}──> 最終預測: {label_text}")
            return
            
        print(f"{indent}{prefix}── [ 節點: {node.feature} ] 門檻: <= {node.threshold:.4f}")
        self._export_tree_structure(node.left, depth + 1, prefix="【是 <=】")
        self._export_tree_structure(node.right, depth + 1, prefix="【否  >】")

    # =====================================================================
    # 💡 功能：將樹結構轉換成結構化文字，以便匯出至 CSV
    # =====================================================================
    def get_tree_rules_text(self):
        """將決策樹的核心規則提取成一行文字摘要"""
        rules = []
        self._extract_rules(self.root, rules)
        return " | ".join(rules)

    def _extract_rules(self, node, rules):
        if node.is_leaf:
            return
        rules.append(f"{node.feature}(<={node.threshold:.2f})")
        self._extract_rules(node.left, rules)
        self._extract_rules(node.right, rules)


# =====================================================================
# 2. 投資組合績效計算邏輯 (Portfolio Evaluation)
# =====================================================================
def evaluate_portfolio(test_df, predictions):
    df_eval = test_df.copy()
    df_eval['Pred_Label'] = predictions
    yearly_returns = {}
    
    for year, group in df_eval.groupby('Year'):
        selected_stocks = group[group['Pred_Label'] == 1]
        if len(selected_stocks) == 0:
            yearly_returns[year] = 0.0
        else:
            yearly_returns[year] = selected_stocks['Return'].mean() / 100
            
    return pd.Series(yearly_returns)


# =====================================================================
# 3. 主執行流程：載入、預處理、擴展視窗驗證與 CSV 輸出 (Main Workflow)
# =====================================================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_filename = "top200.xlsx" 
    data_path = os.path.join(current_dir, "..", "data", data_filename)

    print(f"[*] 正在讀取財務資料集: {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    if "年月" in df.columns:
        df = df[df["年月"] != 200912]
        df["Year"] = (df["年月"] // 100).astype(int)
    else:
        raise KeyError("找不到『年月』欄位，請確認 Excel 表頭欄位名稱。")

    years = sorted(df["Year"].unique())

    features = df.columns[3:19].tolist()   
    target_col = df.columns[20]            
    df = df.rename(columns={df.columns[19]: "Return"}) 

    print("\n[+] 已自動識別並對齊 16 個特徵欄位。")
    print(f"[+] 資料集總年份範圍: {years[0]} 年 ~ {years[-1]} 年")

    performance_results = []
    tree_structure_results = []

    print("\n[+] 開始執行時間序列驗證 (Temporal Validation - Expanding Window)...")
    
    for i in range(1, len(years)):
        train_years = years[:i]
        test_years = years[i:]
        
        train_data = df[df["Year"].isin(train_years)]
        test_data = df[df["Year"].isin(test_years)]

        X_train = train_data[features]
        y_train = train_data[target_col]
        X_test = test_data[features]

        # 建立與訓練自訂 ID3 決策樹 (max_depth=2 層以便清晰解析核心決定變數)
        model = ContinuousID3TreeClassifier(max_depth=2, min_samples_split=5)
        model.fit(X_train, y_train)

        # 顯示該樹決定變數的詳細過程
        print("\n" + "-"*60)
        print(f" 🟢 階段 {f'T{i}':<3} | 訓練集數據年份範圍: {train_years[0]}~{train_years[-1]}")
        print("-" * 60)
        model.print_tree()
        print("-" * 60)

        # 預測整個測試區間
        preds = model.predict(X_test)

        # 計算測試區間內每年的投資組合報酬率序列
        returns_series = evaluate_portfolio(test_data, preds)
        
        # 計算回測績效指標
        cum_ret = (1 + returns_series).prod() - 1
        n_years = len(returns_series)
        ann_ret = (1 + cum_ret) ** (1 / n_years) - 1
        
        equity_curve = (1 + returns_series).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        mdd = drawdown.min()
        
        # 1. 儲存績效指標 
        performance_results.append({
            "Stage": f"T{i}",
            "Test_Interval": f"{test_years[0]}-{test_years[-1]}",
            "Cumulative_Return_%": round(cum_ret * 100, 2),
            "Annualized_Return_%": round(ann_ret * 100, 2),
            "Max_Drawdown_%": round(mdd * 100, 2)
        })

        # 2. 儲存決定變數規則文字 
        tree_structure_results.append({
            "Stage": f"T{i}",
            "Selected_Variables_and_Thresholds": model.get_tree_rules_text()
        })

        print(f" ↳ [測試表現] 預測 {test_years[0]}~{test_years[-1]} | 累積報酬: {cum_ret*100:.2f}%\n")

    # =====================================================================
    # 4. 將結果自動輸出到 CSV 檔案與建立視覺化圖表 (Output & Plotting)
    # =====================================================================
    # 建立輸出資料夾
    output_dir = os.path.join(current_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    perf_csv_path = os.path.join(output_dir, "task1_performance_summary.csv")
    tree_csv_path = os.path.join(output_dir, "task1_tree_structures.csv")
    chart_png_path = os.path.join(output_dir, "task1_performance_chart.png")

    # 轉為 DataFrame
    perf_df = pd.DataFrame(performance_results)
    tree_df = pd.DataFrame(tree_structure_results)

    # 匯出至 CSV
    perf_df.to_csv(perf_csv_path, index=False, encoding='utf-8-sig')
    tree_df.to_csv(tree_csv_path, index=False, encoding='utf-8-sig')

    # 💡 新增功能：將績效數據繪製成精美折線圖
    print("[*] 正在將績效成果轉換為趨勢對比圖表...")
    
    # 建立圖表畫布與大小設定
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 繪製三條代表不同績效數據的線（定義專屬顏色與標記點）
    plt.plot(perf_df["Stage"], perf_df["Cumulative_Return_%"], marker='o', color='#1f77b4', linewidth=2.5, label='Cumulative Return %')
    plt.plot(perf_df["Stage"], perf_df["Annualized_Return_%"], marker='s', color='#2ca02c', linewidth=2, label='Annualized Return %')
    plt.plot(perf_df["Stage"], perf_df["Max_Drawdown_%"], marker='v', color='#d62728', linewidth=2, label='Max Drawdown %')
    
    # 設定圖表標題與軸標籤
    plt.title("Model Performance Across Validation Stages (Task 1)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Validation Stage", fontsize=11, labelpad=10)
    plt.ylabel("Percentage (%)", fontsize=11, labelpad=10)
    
    # 建立網格背景、圖例說明，並優化版面邊界
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    plt.tight_layout()
    
    # 儲存圖表圖片
    plt.savefig(chart_png_path)
    plt.close()

    print("=" * 75)
    print(" 🎉 成果資料與圖表匯出成功！")
    print(f" 📁 績效報表已儲存至: {os.path.abspath(perf_csv_path)}")
    print(f" 📁 績效趨勢圖表已儲存: {os.path.abspath(chart_png_path)}")
    print(f" 📁 決策樹變數結構已儲存: {os.path.abspath(tree_csv_path)}")
    print(" 💡 提示：您可以直接把產出的 PNG 圖表貼進您的簡報 PPT 中！")
    print("=" * 75)

if __name__ == "__main__":
    main()