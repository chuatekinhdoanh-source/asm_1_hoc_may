import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOGISTIC REGRESSION - THRESHOLD OPTIMIZATION")
print("=" * 60)

train_data = pd.read_csv("train_model.csv")
test_data = pd.read_csv("test_model.csv")

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "Adj_Close",
    "Volume",
    "Return",
    "MA5",
    "MA20",
    "MA50",
    "EMA12",
    "EMA26",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "RSI",
    "Volatility20",
    "Volume_MA5",
    "Volume_Change",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Price_MA5_Ratio",
    "Price_MA20_Ratio"
]

X_train = train_data[features]
y_train = train_data["Target"]

X_test = test_data[features]
y_test = test_data["Target"]


# ============================================================
# 3. INTERNAL TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

train_internal_list = []
validation_list = []

for symbol in train_data["Symbol"].unique():

    symbol_data = train_data[
        train_data["Symbol"] == symbol
    ].sort_values("Date")

    split_index = int(len(symbol_data) * 0.8)

    train_internal_list.append(
        symbol_data.iloc[:split_index]
    )

    validation_list.append(
        symbol_data.iloc[split_index:]
    )

train_internal = pd.concat(
    train_internal_list,
    ignore_index=True
)

validation = pd.concat(
    validation_list,
    ignore_index=True
)

X_train_internal = train_internal[features]
y_train_internal = train_internal["Target"]

X_validation = validation[features]
y_validation = validation["Target"]

print("Train Internal:", X_train_internal.shape)
print("Validation:", X_validation.shape)
print("Final Train:", X_train.shape)
print("Test:", X_test.shape)


# ============================================================
# 4. VALIDATION MODEL
# ============================================================

print("\nĐang huấn luyện mô hình Validation...")

validation_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=3000,
        random_state=42
    ))
])

validation_model.fit(
    X_train_internal,
    y_train_internal
)

validation_probabilities = validation_model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# 5. THRESHOLD OPTIMIZATION - PORTFOLIO BASED
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION - PORTFOLIO BASED")
print("=" * 70)

# ------------------------------------------------------------
# 1. Chuẩn bị dữ liệu Validation để mô phỏng giao dịch
# ------------------------------------------------------------

validation_trading_data = validation.copy()

validation_trading_data["Probability_Up"] = (
    validation_probabilities
)

validation_trading_data["Date"] = pd.to_datetime(
    validation_trading_data["Date"]
)

# Đọc dữ liệu giá tương lai
feature_data = pd.read_csv("data_feature.csv")

feature_data["Date"] = pd.to_datetime(
    feature_data["Date"]
)

# Ghép giá ngày tiếp theo vào Validation
validation_trading_data = validation_trading_data.merge(
    feature_data[[
        "Symbol",
        "Date",
        "Next_Adj_Close"
    ]],
    on=["Symbol", "Date"],
    how="left"
)

# Loại bỏ những dòng thiếu giá tương lai
validation_trading_data = validation_trading_data.dropna(
    subset=["Adj_Close", "Next_Adj_Close", "Probability_Up"]
).copy()

# Tính lợi nhuận ngày tiếp theo
validation_trading_data["Next_Return"] = (
    validation_trading_data["Next_Adj_Close"]
    / validation_trading_data["Adj_Close"]
    - 1
)

NUM_STOCKS = 8

# ------------------------------------------------------------
# 2. Hàm đánh giá chiến lược trên Validation
# ------------------------------------------------------------

def evaluate_validation_strategy(
    threshold,
    validation_data
):

    data = validation_data.copy()

    # Tạo tín hiệu BUY / HOLD
    data["Signal"] = np.where(
        data["Probability_Up"] >= threshold,
        1,
        0
    )

    # Lợi nhuận nếu BUY
    data["Strategy_Return"] = (
        data["Signal"] * data["Next_Return"]
    )

    # Tính lợi nhuận danh mục mỗi ngày
    daily_portfolio = (
        data.groupby("Date")
        .agg(
            Strategy_Return=("Strategy_Return", "sum"),
            Buy_Count=("Signal", "sum"),
            Number_Of_Stocks=("Symbol", "nunique")
        )
        .reset_index()
    )

    # Phân bổ vốn cố định: 1/8 cho mỗi mã
    daily_portfolio["Strategy_Return"] = (
        daily_portfolio["Strategy_Return"]
        / NUM_STOCKS
    )

    daily_portfolio = daily_portfolio.sort_values(
        "Date"
    ).reset_index(drop=True)

    daily_returns = daily_portfolio["Strategy_Return"]

    # Tổng lợi nhuận
    total_return = (
        (1 + daily_returns).prod() - 1
    )

    # Lợi nhuận trung bình
    average_daily_return = daily_returns.mean()

    # Độ biến động
    volatility = daily_returns.std()

    # Sharpe Ratio
    sharpe_ratio = (
        average_daily_return / volatility * np.sqrt(252)
        if volatility > 0
        else 0
    )

    # Maximum Drawdown
    cumulative_returns = (
        (1 + daily_returns).cumprod()
    )

    cumulative_with_initial = pd.concat([
        pd.Series([1.0]),
        cumulative_returns.reset_index(drop=True)
    ])

    running_max = cumulative_with_initial.cummax()

    drawdown = (
        cumulative_with_initial / running_max - 1
    )

    max_drawdown = drawdown.min()

    # Số tín hiệu BUY
    total_buy_signals = int(data["Signal"].sum())

    total_signals = len(data)

    buy_rate = (
        total_buy_signals / total_signals
        if total_signals > 0
        else 0
    )

    # Win Rate trên các tín hiệu BUY
    buy_data = data[data["Signal"] == 1]

    win_rate = (
        (buy_data["Next_Return"] > 0).mean()
        if len(buy_data) > 0
        else 0
    )

    # Các chỉ số Classification
    predictions = data["Signal"]

    accuracy = accuracy_score(
        data["Target"],
        predictions
    )

    precision = precision_score(
        data["Target"],
        predictions,
        zero_division=0
    )

    recall = recall_score(
        data["Target"],
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        data["Target"],
        predictions,
        zero_division=0
    )

    return {
        "Threshold": threshold,
        "Total_Return": total_return,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "Daily_Volatility": volatility,
        "Buy_Signals": total_buy_signals,
        "Buy_Rate": buy_rate,
        "Win_Rate": win_rate,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Number_Of_Days": len(daily_returns)
    }


# ------------------------------------------------------------
# 3. Thử các threshold
# ------------------------------------------------------------

thresholds = np.arange(0.30, 0.71, 0.01)

threshold_results = []

for threshold in thresholds:

    result = evaluate_validation_strategy(
        threshold,
        validation_trading_data
    )

    threshold_results.append(result)


threshold_df = pd.DataFrame(
    threshold_results
)

# ------------------------------------------------------------
# 4. Giới hạn để tránh chiến lược BUY quá ít hoặc quá nhiều
# ------------------------------------------------------------

# Chỉ xét threshold có tỷ lệ BUY từ 5% đến 70%
valid_thresholds = threshold_df[
    (threshold_df["Buy_Rate"] >= 0.05) &
    (threshold_df["Buy_Rate"] <= 0.70)
].copy()

if len(valid_thresholds) == 0:

    print(
        "WARNING: Không có threshold đáp ứng điều kiện Buy Rate."
    )

    valid_thresholds = threshold_df.copy()


# ------------------------------------------------------------
# 5. Chọn threshold theo Sharpe Ratio
# ------------------------------------------------------------

valid_thresholds = valid_thresholds.sort_values(
    by=["Sharpe_Ratio", "Total_Return"],
    ascending=False
)

best_threshold = valid_thresholds.iloc[0]["Threshold"]

best_validation_sharpe = (
    valid_thresholds.iloc[0]["Sharpe_Ratio"]
)

best_validation_return = (
    valid_thresholds.iloc[0]["Total_Return"]
)

best_validation_f1 = (
    valid_thresholds.iloc[0]["F1"]
)

# ------------------------------------------------------------
# 6. Hiển thị kết quả
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST THRESHOLD - VALIDATION")
print("=" * 70)

print("Best Threshold:", round(best_threshold, 2))

print(
    "Validation Sharpe:",
    round(best_validation_sharpe, 4)
)

print(
    "Validation Total Return:",
    round(best_validation_return * 100, 2),
    "%"
)

print(
    "Validation F1:",
    round(best_validation_f1, 4)
)

print("\nTop 10 Threshold theo Sharpe Ratio:")

print(
    valid_thresholds.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# 7. Lưu toàn bộ kết quả
# ------------------------------------------------------------

threshold_df.to_csv(
    "threshold_validation_results.csv",
    index=False
)

valid_thresholds.to_csv(
    "threshold_validation_ranked.csv",
    index=False
)

print("\nĐã lưu:")
print("- threshold_validation_results.csv")
print("- threshold_validation_ranked.csv")


# ============================================================
# 6. TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAIN FINAL MODEL")
print("=" * 60)

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=3000,
        random_state=42
    ))
])

final_model.fit(
    X_train,
    y_train
)

print("Đã huấn luyện Final Model trên toàn bộ Train.")


# ============================================================
# 7. PREDICT TEST
# ============================================================

y_pred_proba = final_model.predict_proba(
    X_test
)[:, 1]

y_pred_05 = (
    y_pred_proba >= 0.5
).astype(int)

y_pred_optimized = (
    y_pred_proba >= best_threshold
).astype(int)


# ============================================================
# 8. CHECK PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("KIỂM TRA PHÂN PHỐI XÁC SUẤT")
print("=" * 60)

print("Min:", round(y_pred_proba.min(), 4))
print("Max:", round(y_pred_proba.max(), 4))
print("Mean:", round(y_pred_proba.mean(), 4))
print("Median:", round(np.median(y_pred_proba), 4))

print("\nCác khoảng xác suất:")

probability_ranges = [
    (0.0, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 1.0)
]

for lower, upper in probability_ranges:

    if upper == 1.0:
        count = np.sum(
            (y_pred_proba >= lower) &
            (y_pred_proba <= upper)
        )
    else:
        count = np.sum(
            (y_pred_proba >= lower) &
            (y_pred_proba < upper)
        )

    print(
        f"{lower:.1f} - {upper:.1f}: {count}"
    )


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_probability=None
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if y_probability is not None:
        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )
    else:
        roc_auc = np.nan

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }


# ============================================================
# 10. BASELINE
# ============================================================

majority_class = y_train.value_counts().idxmax()

baseline_predictions = np.full(
    len(y_test),
    majority_class
)

baseline_result = evaluate_model(
    "Baseline Majority Class",
    y_test,
    baseline_predictions
)


# ============================================================
# 11. EVALUATE MODELS
# ============================================================

result_05 = evaluate_model(
    "Logistic Regression Threshold 0.50",
    y_test,
    y_pred_05,
    y_pred_proba
)

result_optimized = evaluate_model(
    f"Logistic Regression Threshold {best_threshold:.2f}",
    y_test,
    y_pred_optimized,
    y_pred_proba
)

metrics_df = pd.DataFrame([
    baseline_result,
    result_05,
    result_optimized
])


# ============================================================
# 12. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred_optimized,
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred_optimized
)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nInterpretation:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# 15. SAVE PREDICTIONS
# ============================================================

prediction_data = test_data.copy()

prediction_data["Probability_Up"] = y_pred_proba
prediction_data["Prediction_05"] = y_pred_05
prediction_data["Prediction_Optimized"] = y_pred_optimized

prediction_data.to_csv(
    "model_predictions_optimized.csv",
    index=False
)


# ============================================================
# 16. SAVE METRICS
# ============================================================

metrics_df.to_csv(
    "model_metrics_optimized.csv",
    index=False
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    "logistic_regression_optimized.pkl"
)

with open(
    "best_threshold.txt",
    "w"
) as file:

    file.write(
        str(best_threshold)
    )


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING HOÀN TẤT")
print("=" * 60)

print("Best Threshold:", best_threshold)
print("Validation F1:", round(best_validation_f1, 4))

print(
    "Test Accuracy:",
    round(result_optimized["Accuracy"], 4)
)

print(
    "Test Precision:",
    round(result_optimized["Precision"], 4)
)

print(
    "Test Recall:",
    round(result_optimized["Recall"], 4)
)

print(
    "Test F1:",
    round(result_optimized["F1"], 4)
)

print(
    "Test ROC-AUC:",
    round(result_optimized["ROC_AUC"], 4)
)

print("\nFiles đã tạo:")
print("- logistic_regression_optimized.pkl")
print("- best_threshold.txt")
print("- model_predictions_optimized.csv")
print("- model_metrics_optimized.csv")
print("- threshold_validation_results.csv")

print("\nHoàn thành!")
# ============================================================
# 19. EVALUATE EACH STOCK SYMBOL
# ============================================================

print("\n" + "=" * 70)
print("ĐÁNH GIÁ LOGISTIC REGRESSION THEO TỪNG MÃ CỔ PHIẾU")
print("=" * 70)

symbol_results = []

for symbol in test_data["Symbol"].unique():

    mask = test_data["Symbol"] == symbol

    y_true_symbol = y_test[mask]
    y_proba_symbol = y_pred_proba[mask]
    y_pred_symbol = y_pred_optimized[mask]

    accuracy = accuracy_score(
        y_true_symbol,
        y_pred_symbol
    )

    precision = precision_score(
        y_true_symbol,
        y_pred_symbol,
        zero_division=0
    )

    recall = recall_score(
        y_true_symbol,
        y_pred_symbol,
        zero_division=0
    )

    f1 = f1_score(
        y_true_symbol,
        y_pred_symbol,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true_symbol,
        y_proba_symbol
    ) if len(np.unique(y_true_symbol)) == 2 else np.nan

    predict_up = np.sum(y_pred_symbol == 1)
    predict_down = np.sum(y_pred_symbol == 0)

    actual_up = np.sum(y_true_symbol == 1)
    actual_down = np.sum(y_true_symbol == 0)

    symbol_results.append({
        "Symbol": symbol,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Predict_Up": predict_up,
        "Predict_Down": predict_down,
        "Actual_Up": actual_up,
        "Actual_Down": actual_down
    })


symbol_metrics = pd.DataFrame(symbol_results)

symbol_metrics = symbol_metrics.sort_values(
    by="F1",
    ascending=False
)

print("\nKẾT QUẢ TỪNG MÃ:")
print(
    symbol_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 20. SAVE SYMBOL RESULTS
# ============================================================

symbol_metrics.to_csv(
    "model_metrics_by_symbol.csv",
    index=False
)

print("\nĐã lưu: model_metrics_by_symbol.csv")


# ============================================================
# 21. BEST AND WORST SYMBOL
# ============================================================

best_symbol = symbol_metrics.iloc[0]
worst_symbol = symbol_metrics.iloc[-1]

print("\n" + "=" * 70)
print("MÃ CỔ PHIẾU TỐT NHẤT VÀ KÉM NHẤT")
print("=" * 70)

print("\nMã tốt nhất theo F1:")
print("Symbol:", best_symbol["Symbol"])
print("F1:", round(best_symbol["F1"], 4))
print("Accuracy:", round(best_symbol["Accuracy"], 4))
print("Precision:", round(best_symbol["Precision"], 4))
print("Recall:", round(best_symbol["Recall"], 4))
print("ROC-AUC:", round(best_symbol["ROC_AUC"], 4))

print("\nMã thấp nhất theo F1:")
print("Symbol:", worst_symbol["Symbol"])
print("F1:", round(worst_symbol["F1"], 4))
print("Accuracy:", round(worst_symbol["Accuracy"], 4))
print("Precision:", round(worst_symbol["Precision"], 4))
print("Recall:", round(worst_symbol["Recall"], 4))
print("ROC-AUC:", round(worst_symbol["ROC_AUC"], 4))
# ============================================================
# 22. TRADING STRATEGY SIMULATION - CORRECT PORTFOLIO EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TRADING STRATEGY SIMULATION - PORTFOLIO")
print("=" * 70)

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Chuẩn bị dữ liệu mô phỏng
# ------------------------------------------------------------

# Dữ liệu test + xác suất dự đoán
trading_data = test_data.copy()

# Ghép xác suất dự đoán
trading_data["Probability_Up"] = y_pred_proba

# Ghép dữ liệu giá tương lai
feature_data = pd.read_csv("data_feature.csv")

feature_data["Date"] = pd.to_datetime(feature_data["Date"])
trading_data["Date"] = pd.to_datetime(trading_data["Date"])

trading_data = trading_data.merge(
    feature_data[[
        "Symbol",
        "Date",
        "Adj_Close",
        "Next_Adj_Close"
    ]],
    on=["Symbol", "Date"],
    how="left",
    suffixes=("", "_feature")
)

# Nếu Adj_Close đã có sẵn thì dùng cột đó
if "Adj_Close_feature" in trading_data.columns:
    trading_data["Adj_Close"] = trading_data["Adj_Close_feature"]

# Loại bỏ dữ liệu không có giá ngày tiếp theo
trading_data = trading_data.dropna(
    subset=["Adj_Close", "Next_Adj_Close", "Probability_Up"]
).copy()

# ------------------------------------------------------------
# 2. Tính lợi nhuận ngày tiếp theo
# ------------------------------------------------------------

trading_data["Next_Return"] = (
    trading_data["Next_Adj_Close"] / trading_data["Adj_Close"] - 1
)

# ------------------------------------------------------------
# 3. Tạo tín hiệu giao dịch
# ------------------------------------------------------------

THRESHOLD_05 = 0.50
THRESHOLD_OPTIMIZED = best_threshold

trading_data["Signal_05"] = np.where(
    trading_data["Probability_Up"] >= THRESHOLD_05,
    "BUY",
    "HOLD"
)

trading_data["Signal_Optimized"] = np.where(
    trading_data["Probability_Up"] >= THRESHOLD_OPTIMIZED,
    "BUY",
    "HOLD"
)

# ------------------------------------------------------------
# 4. Tính lợi nhuận từng cổ phiếu theo từng chiến lược
# ------------------------------------------------------------

# Threshold 0.50
trading_data["Strategy_Return_05"] = np.where(
    trading_data["Signal_05"] == "BUY",
    trading_data["Next_Return"],
    0
)

# Threshold tối ưu 0.38
trading_data["Strategy_Return_Optimized"] = np.where(
    trading_data["Signal_Optimized"] == "BUY",
    trading_data["Next_Return"],
    0
)

# Buy & Hold: luôn giữ cổ phiếu
trading_data["BuyHold_Return"] = trading_data["Next_Return"]

# ------------------------------------------------------------
# 5. Tính lợi nhuận danh mục theo NGÀY
# ------------------------------------------------------------

# Mỗi ngày phân bổ vốn đều cho 8 mã
NUM_STOCKS = 8

portfolio_daily = (
    trading_data.groupby("Date")
    .agg(
        Strategy_05_Daily_Return=(
            "Strategy_Return_05", "sum"
        ),
        Strategy_Optimized_Daily_Return=(
            "Strategy_Return_Optimized", "sum"
        ),
        BuyHold_Daily_Return=(
            "BuyHold_Return", "sum"
        ),
        Number_Of_Stocks=(
            "Symbol", "nunique"
        ),
        Strategy_05_Buy_Count=(
            "Signal_05",
            lambda x: (x == "BUY").sum()
        ),
        Strategy_Optimized_Buy_Count=(
            "Signal_Optimized",
            lambda x: (x == "BUY").sum()
        )
    )
    .reset_index()
)

# Chia cho 8 để giữ tỷ trọng vốn cố định 12.5% mỗi mã
portfolio_daily["Strategy_05_Daily_Return"] /= NUM_STOCKS
portfolio_daily["Strategy_Optimized_Daily_Return"] /= NUM_STOCKS
portfolio_daily["BuyHold_Daily_Return"] /= NUM_STOCKS

# ------------------------------------------------------------
# 6. Kiểm tra số lượng cổ phiếu mỗi ngày
# ------------------------------------------------------------

print("\nKiểm tra số cổ phiếu trong danh mục mỗi ngày:")

print(
    portfolio_daily["Number_Of_Stocks"]
    .value_counts()
    .sort_index()
)

if (portfolio_daily["Number_Of_Stocks"] != NUM_STOCKS).any():
    print(
        "\nWARNING: Một số ngày không có đủ 8 cổ phiếu."
        "\nCần kiểm tra dữ liệu trước khi kết luận hiệu quả danh mục."
    )

# ------------------------------------------------------------
# 7. Sắp xếp thời gian
# ------------------------------------------------------------

portfolio_daily = portfolio_daily.sort_values("Date").reset_index(drop=True)

# ------------------------------------------------------------
# 8. Tính lợi nhuận tích lũy
# ------------------------------------------------------------

portfolio_daily["Strategy_05_Cumulative_Return"] = (
    (1 + portfolio_daily["Strategy_05_Daily_Return"])
    .cumprod() - 1
)

portfolio_daily["Strategy_Optimized_Cumulative_Return"] = (
    (1 + portfolio_daily["Strategy_Optimized_Daily_Return"])
    .cumprod() - 1
)

portfolio_daily["BuyHold_Cumulative_Return"] = (
    (1 + portfolio_daily["BuyHold_Daily_Return"])
    .cumprod() - 1
)

# ------------------------------------------------------------
# 9. Hàm đánh giá chiến lược
# ------------------------------------------------------------

def evaluate_portfolio(
    strategy_name,
    daily_returns,
    total_buy_signals=None,
    win_rate=None
):

    daily_returns = daily_returns.dropna()

    total_return = (
        (1 + daily_returns).prod() - 1
    )

    average_daily_return = daily_returns.mean()

    volatility = daily_returns.std()

    # Sharpe Ratio đơn giản hóa, giả sử lãi suất phi rủi ro = 0
    sharpe_ratio = (
        average_daily_return / volatility * np.sqrt(252)
        if volatility != 0
        else 0
    )

    # Maximum Drawdown
    cumulative_returns = (
        (1 + daily_returns).cumprod()
    )

    running_max = cumulative_returns.cummax()

    drawdown = (
        cumulative_returns / running_max - 1
    )

    max_drawdown = drawdown.min()

    return {
        "Strategy": strategy_name,
        "Total_Return": total_return,
        "Average_Daily_Return": average_daily_return,
        "Daily_Volatility": volatility,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "Total_Buy_Signals": total_buy_signals,
        "Win_Rate": win_rate,
        "Number_Of_Days": len(daily_returns)
    }

# ------------------------------------------------------------
# 10. Win Rate của các giao dịch BUY
# ------------------------------------------------------------

buy_05 = trading_data[
    trading_data["Signal_05"] == "BUY"
]

buy_optimized = trading_data[
    trading_data["Signal_Optimized"] == "BUY"
]

win_rate_05 = (
    (buy_05["Next_Return"] > 0).mean()
    if len(buy_05) > 0
    else 0
)

win_rate_optimized = (
    (buy_optimized["Next_Return"] > 0).mean()
    if len(buy_optimized) > 0
    else 0
)

# ------------------------------------------------------------
# 11. Đánh giá 3 chiến lược
# ------------------------------------------------------------

portfolio_results = []

portfolio_results.append(
    evaluate_portfolio(
        "Logistic Regression Threshold 0.50",
        portfolio_daily["Strategy_05_Daily_Return"],
        len(buy_05),
        win_rate_05
    )
)

portfolio_results.append(
    evaluate_portfolio(
        f"Logistic Regression Threshold {best_threshold:.2f}",
        portfolio_daily["Strategy_Optimized_Daily_Return"],
        len(buy_optimized),
        win_rate_optimized
    )
)

portfolio_results.append(
    evaluate_portfolio(
        "Buy & Hold",
        portfolio_daily["BuyHold_Daily_Return"]
    )
)

portfolio_performance = pd.DataFrame(portfolio_results)

# ------------------------------------------------------------
# 12. Hiển thị kết quả
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("KẾT QUẢ ĐÁNH GIÁ DANH MỤC")
print("=" * 70)

print(
    portfolio_performance.to_string(index=False)
)

# ------------------------------------------------------------
# 13. Lưu kết quả
# ------------------------------------------------------------

portfolio_daily.to_csv(
    "portfolio_daily_returns.csv",
    index=False
)

portfolio_performance.to_csv(
    "portfolio_performance_corrected.csv",
    index=False
)

print("\nĐã lưu:")
print("- portfolio_daily_returns.csv")
print("- portfolio_performance_corrected.csv")

print("\nHoàn thành mô phỏng danh mục!")