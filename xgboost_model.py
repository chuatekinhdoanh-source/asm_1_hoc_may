# ============================================================
# XGBOOST - DỰ ĐOÁN XU HƯỚNG GIÁ CỔ PHIẾU NGÀY TIẾP THEO
# Dự án: ML phân tích và dự báo thị trường chứng khoán Việt Nam
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CẤU HÌNH
# ============================================================

RANDOM_STATE = 42

TRAIN_FILE = "train_model.csv"
TEST_FILE = "test_model.csv"
FEATURE_FILE = "data_feature.csv"

FEATURES = [
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

TARGET = "Target"

# Số cổ phiếu trong danh mục
N_STOCKS = 8

# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("XGBOOST - STOCK MARKET PREDICTION")
print("=" * 70)

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)
feature_data = pd.read_csv(FEATURE_FILE)

for df in [train_data, test_data, feature_data]:
    df["Date"] = pd.to_datetime(df["Date"])

print("\nĐã đọc dữ liệu thành công!")
print(f"Train: {train_data.shape}")
print(f"Test : {test_data.shape}")
print(f"Feature: {feature_data.shape}")

# Kiểm tra dữ liệu
missing_features = [f for f in FEATURES if f not in train_data.columns]

if missing_features:
    raise ValueError(f"Thiếu features: {missing_features}")

# ============================================================
# 3. CHIA TRAIN / VALIDATION THEO THỜI GIAN
#    Chia riêng từng mã cổ phiếu
# ============================================================

train_parts = []
validation_parts = []

for symbol in train_data["Symbol"].unique():

    symbol_data = train_data[
        train_data["Symbol"] == symbol
    ].sort_values("Date").copy()

    split_index = int(len(symbol_data) * 0.8)

    train_part = symbol_data.iloc[:split_index]
    validation_part = symbol_data.iloc[split_index:]

    train_parts.append(train_part)
    validation_parts.append(validation_part)

train_internal = pd.concat(
    train_parts,
    ignore_index=True
)

validation_data = pd.concat(
    validation_parts,
    ignore_index=True
)

X_train = train_internal[FEATURES]
y_train = train_internal[TARGET]

X_validation = validation_data[FEATURES]
y_validation = validation_data[TARGET]

print("\n" + "=" * 70)
print("TRAIN / VALIDATION")
print("=" * 70)

print(f"Internal Train: {len(train_internal)}")
print(f"Validation   : {len(validation_data)}")

print("\nTrain distribution:")
print(y_train.value_counts())

print("\nValidation distribution:")
print(y_validation.value_counts())

# ============================================================
# 4. TUNING HYPERPARAMETERS
# ============================================================

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [1.0]
}

configs = list(ParameterGrid(param_grid))

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

print(f"Số cấu hình cần thử: {len(configs)}")

tuning_results = []
best_auc = -1
best_params = None
best_validation_model = None

for i, params in enumerate(configs, start=1):

    print(f"\nĐang thử cấu hình {i}/{len(configs)}")
    print(params)

    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(
        X_train,
        y_train
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions_05 = (
        validation_probabilities >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_validation,
        validation_probabilities
    )

    accuracy = accuracy_score(
        y_validation,
        validation_predictions_05
    )

    precision = precision_score(
        y_validation,
        validation_predictions_05,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_predictions_05,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_predictions_05,
        zero_division=0
    )

    result = {
        **params,
        "Validation_AUC": auc,
        "Validation_Accuracy": accuracy,
        "Validation_Precision": precision,
        "Validation_Recall": recall,
        "Validation_F1": f1
    }

    tuning_results.append(result)

    print(f"AUC      : {auc:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1       : {f1:.4f}")

    if auc > best_auc:

        best_auc = auc
        best_params = params
        best_validation_model = model

# Lưu tuning results
tuning_df = pd.DataFrame(tuning_results)

tuning_df = tuning_df.sort_values(
    "Validation_AUC",
    ascending=False
)

tuning_df.to_csv(
    "xgboost_tuning_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)

print(best_params)
print(f"Best Validation AUC: {best_auc:.4f}")

# ============================================================
# 5. CHỌN THRESHOLD TRÊN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION - VALIDATION")
print("=" * 70)

validation_probabilities = (
    best_validation_model.predict_proba(X_validation)[:, 1]
)

validation_trading_data = validation_data[
    ["Date", "Symbol", "Adj_Close", TARGET]
].copy()

validation_trading_data["Probability"] = (
    validation_probabilities
)

# Lấy giá ngày tiếp theo từ data_feature
next_price_data = feature_data[
    ["Date", "Symbol", "Next_Adj_Close"]
].copy()

next_price_data["Date"] = pd.to_datetime(
    next_price_data["Date"]
)

validation_trading_data = validation_trading_data.merge(
    next_price_data,
    on=["Date", "Symbol"],
    how="left"
)

validation_trading_data["Next_Return"] = (
    validation_trading_data["Next_Adj_Close"]
    / validation_trading_data["Adj_Close"]
    - 1
)

validation_trading_data = validation_trading_data.dropna(
    subset=["Next_Return", "Probability"]
)

# ============================================================
# HÀM ĐÁNH GIÁ DANH MỤC
# ============================================================

def portfolio_metrics(
    data,
    threshold,
    n_stocks=N_STOCKS
):

    df = data.copy()

    df["Buy_Signal"] = (
        df["Probability"] >= threshold
    ).astype(int)

    df["Strategy_Return"] = (
        df["Next_Return"] * df["Buy_Signal"]
    )

    daily_strategy = (
        df.groupby("Date")["Strategy_Return"]
        .sum()
        / n_stocks
    )

    daily_strategy = daily_strategy.sort_index()

    total_return = (
        (1 + daily_strategy).prod() - 1
    )

    if daily_strategy.std() != 0:

        sharpe = (
            daily_strategy.mean()
            / daily_strategy.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0

    cumulative_returns = (
        1 + daily_strategy
    ).cumprod()

    running_max = cumulative_returns.cummax()

    drawdown = (
        cumulative_returns / running_max - 1
    )

    max_drawdown = drawdown.min()

    buy_count = df["Buy_Signal"].sum()

    if buy_count > 0:

        win_rate = (
            df.loc[
                df["Buy_Signal"] == 1,
                "Next_Return"
            ] > 0
        ).mean()

    else:

        win_rate = 0

    buy_rate = (
        df["Buy_Signal"].mean()
    )

    return {
        "Threshold": threshold,
        "Total_Return": total_return,
        "Sharpe": sharpe,
        "Max_Drawdown": max_drawdown,
        "Buy_Count": buy_count,
        "Buy_Rate": buy_rate,
        "Win_Rate": win_rate,
        "Daily_Returns": daily_strategy
    }


thresholds = np.arange(
    0.30,
    0.701,
    0.01
)

threshold_results = []

for threshold in thresholds:

    metrics = portfolio_metrics(
        validation_trading_data,
        threshold
    )

    # Loại các ngưỡng tạo tín hiệu quá ít hoặc quá nhiều
    if 0.05 <= metrics["Buy_Rate"] <= 0.70:

        threshold_results.append({
            "Threshold": metrics["Threshold"],
            "Total_Return": metrics["Total_Return"],
            "Sharpe": metrics["Sharpe"],
            "Max_Drawdown": metrics["Max_Drawdown"],
            "Buy_Count": metrics["Buy_Count"],
            "Buy_Rate": metrics["Buy_Rate"],
            "Win_Rate": metrics["Win_Rate"]
        })

threshold_df = pd.DataFrame(
    threshold_results
)

if threshold_df.empty:

    raise ValueError(
        "Không tìm được threshold phù hợp. "
        "Hãy kiểm tra xác suất dự đoán."
    )

threshold_df = threshold_df.sort_values(
    ["Sharpe", "Total_Return"],
    ascending=False
)

threshold_df.to_csv(
    "xgboost_threshold_ranked.csv",
    index=False
)

threshold_df.to_csv(
    "xgboost_threshold_results.csv",
    index=False
)

best_threshold = float(
    threshold_df.iloc[0]["Threshold"]
)

best_validation_metrics = threshold_df.iloc[0]

print("\nTop 10 Threshold:")

print(
    threshold_df.head(10).to_string(
        index=False
    )
)

print("\nThreshold tốt nhất:")
print(f"Threshold: {best_threshold:.2f}")
print(
    f"Validation Sharpe: "
    f"{best_validation_metrics['Sharpe']:.4f}"
)
print(
    f"Validation Return: "
    f"{best_validation_metrics['Total_Return']:.4%}"
)
print(
    f"Validation MaxDD: "
    f"{best_validation_metrics['Max_Drawdown']:.4%}"
)
print(
    f"Validation Win Rate: "
    f"{best_validation_metrics['Win_Rate']:.4%}"
)

# ============================================================
# 6. TRAIN MODEL CUỐI CÙNG TRÊN TOÀN BỘ TRAIN
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

final_model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)

final_model.fit(
    train_data[FEATURES],
    train_data[TARGET]
)

print("Đã huấn luyện Final Model!")

# ============================================================
# 7. DỰ ĐOÁN TEST
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

X_test = test_data[FEATURES]
y_test = test_data[TARGET]

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

# Dự đoán với threshold 0.50
predictions_05 = (
    test_probabilities >= 0.50
).astype(int)

# Dự đoán với threshold tối ưu
predictions_best = (
    test_probabilities >= best_threshold
).astype(int)

# ============================================================
# HÀM METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities
):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        )
    }


metrics_05 = calculate_metrics(
    y_test,
    predictions_05,
    test_probabilities
)

metrics_best = calculate_metrics(
    y_test,
    predictions_best,
    test_probabilities
)

print("\nTEST - THRESHOLD 0.50")

for name, value in metrics_05.items():

    print(f"{name}: {value:.4f}")

print("\nTEST - OPTIMIZED THRESHOLD")

print(f"Threshold: {best_threshold:.2f}")

for name, value in metrics_best.items():

    print(f"{name}: {value:.4f}")

print("\nConfusion Matrix - Optimized Threshold")

print(
    confusion_matrix(
        y_test,
        predictions_best
    )
)

# ============================================================
# 8. PER-SYMBOL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PER-SYMBOL TEST RESULTS")
print("=" * 70)

test_result_data = test_data.copy()

test_result_data["Probability"] = (
    test_probabilities
)

test_result_data["Prediction_05"] = (
    predictions_05
)

test_result_data["Prediction_Best"] = (
    predictions_best
)

symbol_results = []

for symbol in test_result_data["Symbol"].unique():

    symbol_data = test_result_data[
        test_result_data["Symbol"] == symbol
    ]

    y_symbol = symbol_data[TARGET]

    p_symbol = symbol_data[
        "Prediction_Best"
    ]

    prob_symbol = symbol_data[
        "Probability"
    ]

    result = {
        "Symbol": symbol,
        "Accuracy": accuracy_score(
            y_symbol,
            p_symbol
        ),
        "Precision": precision_score(
            y_symbol,
            p_symbol,
            zero_division=0
        ),
        "Recall": recall_score(
            y_symbol,
            p_symbol,
            zero_division=0
        ),
        "F1": f1_score(
            y_symbol,
            p_symbol,
            zero_division=0
        )
    }

    if y_symbol.nunique() > 1:

        result["ROC_AUC"] = roc_auc_score(
            y_symbol,
            prob_symbol
        )

    else:

        result["ROC_AUC"] = np.nan

    symbol_results.append(result)

symbol_results_df = pd.DataFrame(
    symbol_results
)

symbol_results_df = symbol_results_df.sort_values(
    "F1",
    ascending=False
)

print(
    symbol_results_df.to_string(
        index=False
    )
)

symbol_results_df.to_csv(
    "xgboost_metrics_by_symbol.csv",
    index=False
)

# ============================================================
# 9. PORTFOLIO BACKTEST TRÊN TEST
# ============================================================

print("\n" + "=" * 70)
print("PORTFOLIO BACKTEST - TEST")
print("=" * 70)

test_trading_data = test_result_data[
    ["Date", "Symbol", "Adj_Close", "Probability"]
].copy()

test_trading_data = test_trading_data.merge(
    next_price_data,
    on=["Date", "Symbol"],
    how="left"
)

test_trading_data["Next_Return"] = (
    test_trading_data["Next_Adj_Close"]
    / test_trading_data["Adj_Close"]
    - 1
)

test_trading_data = test_trading_data.dropna(
    subset=["Next_Return"]
)

# ============================================================
# STRATEGY 0.50
# ============================================================

strategy_05 = portfolio_metrics(
    test_trading_data,
    0.50
)

# ============================================================
# STRATEGY OPTIMIZED THRESHOLD
# ============================================================

strategy_best = portfolio_metrics(
    test_trading_data,
    best_threshold
)

# ============================================================
# BUY & HOLD
# ============================================================

test_trading_data["Buy_Hold_Return"] = (
    test_trading_data["Next_Return"]
)

daily_buy_hold = (
    test_trading_data.groupby("Date")[
        "Buy_Hold_Return"
    ].sum()
    / N_STOCKS
)

daily_buy_hold = daily_buy_hold.sort_index()

buy_hold_total_return = (
    (1 + daily_buy_hold).prod() - 1
)

buy_hold_sharpe = (
    daily_buy_hold.mean()
    / daily_buy_hold.std()
    * np.sqrt(252)
    if daily_buy_hold.std() != 0
    else 0
)

buy_hold_cumulative = (
    1 + daily_buy_hold
).cumprod()

buy_hold_running_max = (
    buy_hold_cumulative.cummax()
)

buy_hold_drawdown = (
    buy_hold_cumulative
    / buy_hold_running_max
    - 1
)

buy_hold_max_drawdown = (
    buy_hold_drawdown.min()
)

# ============================================================
# HIỂN THỊ KẾT QUẢ
# ============================================================

portfolio_results = pd.DataFrame([
    {
        "Strategy": "XGBoost Threshold 0.50",
        "Threshold": 0.50,
        "Total_Return": strategy_05["Total_Return"],
        "Sharpe": strategy_05["Sharpe"],
        "Max_Drawdown": strategy_05["Max_Drawdown"],
        "Buy_Count": strategy_05["Buy_Count"],
        "Buy_Rate": strategy_05["Buy_Rate"],
        "Win_Rate": strategy_05["Win_Rate"]
    },

    {
        "Strategy": "XGBoost Optimized",
        "Threshold": best_threshold,
        "Total_Return": strategy_best["Total_Return"],
        "Sharpe": strategy_best["Sharpe"],
        "Max_Drawdown": strategy_best["Max_Drawdown"],
        "Buy_Count": strategy_best["Buy_Count"],
        "Buy_Rate": strategy_best["Buy_Rate"],
        "Win_Rate": strategy_best["Win_Rate"]
    },

    {
        "Strategy": "Buy & Hold",
        "Threshold": np.nan,
        "Total_Return": buy_hold_total_return,
        "Sharpe": buy_hold_sharpe,
        "Max_Drawdown": buy_hold_max_drawdown,
        "Buy_Count": len(test_trading_data),
        "Buy_Rate": 1.0,
        "Win_Rate": (
            test_trading_data["Next_Return"] > 0
        ).mean()
    }
])

print("\nPORTFOLIO PERFORMANCE:")

print(
    portfolio_results.to_string(
        index=False
    )
)

portfolio_results.to_csv(
    "xgboost_portfolio_performance.csv",
    index=False
)

# ============================================================
# 10. LƯU DAILY RETURNS
# ============================================================

daily_returns_df = pd.DataFrame({
    "Date": daily_buy_hold.index,
    "Buy_Hold_Return": daily_buy_hold.values
})

strategy_best_daily = (
    strategy_best["Daily_Returns"]
)

strategy_05_daily = (
    strategy_05["Daily_Returns"]
)

daily_returns_df = daily_returns_df.merge(
    pd.DataFrame({
        "Date": strategy_best_daily.index,
        "XGBoost_Optimized_Return":
            strategy_best_daily.values
    }),
    on="Date",
    how="left"
)

daily_returns_df = daily_returns_df.merge(
    pd.DataFrame({
        "Date": strategy_05_daily.index,
        "XGBoost_Threshold_050_Return":
            strategy_05_daily.values
    }),
    on="Date",
    how="left"
)

daily_returns_df = daily_returns_df.fillna(0)

daily_returns_df.to_csv(
    "xgboost_portfolio_daily_returns.csv",
    index=False
)

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": final_model.feature_importances_
})

feature_importance_df = (
    feature_importance_df.sort_values(
        "Importance",
        ascending=False
    )
)

print(
    feature_importance_df.to_string(
        index=False
    )
)

feature_importance_df.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)

# ============================================================
# 12. LƯU PREDICTIONS
# ============================================================

predictions_df = test_data[
    ["Date", "Symbol", TARGET]
].copy()

predictions_df["Probability"] = (
    test_probabilities
)

predictions_df["Prediction_050"] = (
    predictions_05
)

predictions_df["Prediction_Optimized"] = (
    predictions_best
)

predictions_df["Optimized_Threshold"] = (
    best_threshold
)

predictions_df.to_csv(
    "xgboost_predictions.csv",
    index=False
)

# ============================================================
# 13. LƯU TEST METRICS
# ============================================================

test_metrics_df = pd.DataFrame([
    {
        "Model": "XGBoost",
        "Threshold": 0.50,
        **metrics_05
    },

    {
        "Model": "XGBoost",
        "Threshold": best_threshold,
        **metrics_best
    }
])

test_metrics_df.to_csv(
    "xgboost_test_metrics.csv",
    index=False
)

# ============================================================
# 14. LƯU MODEL VÀ CẤU HÌNH
# ============================================================

joblib.dump(
    final_model,
    "xgboost_optimized.pkl"
)

with open(
    "xgboost_best_threshold.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        str(best_threshold)
    )

with open(
    "xgboost_best_params.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        str(best_params)
    )

# ============================================================
# 15. TỔNG KẾT
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST COMPLETED")
print("=" * 70)

print(f"\nBest Parameters: {best_params}")
print(f"Best Validation AUC: {best_auc:.4f}")
print(f"Best Threshold: {best_threshold:.2f}")

print("\nTest Results - Optimized Threshold:")

print(
    f"Accuracy : {metrics_best['Accuracy']:.4f}"
)

print(
    f"Precision: {metrics_best['Precision']:.4f}"
)

print(
    f"Recall   : {metrics_best['Recall']:.4f}"
)

print(
    f"F1       : {metrics_best['F1']:.4f}"
)

print(
    f"ROC-AUC  : {metrics_best['ROC_AUC']:.4f}"
)

print("\nPortfolio Test:")

print(
    f"Total Return: "
    f"{strategy_best['Total_Return']:.4%}"
)

print(
    f"Sharpe: "
    f"{strategy_best['Sharpe']:.4f}"
)

print(
    f"Max Drawdown: "
    f"{strategy_best['Max_Drawdown']:.4%}"
)

print(
    f"Win Rate: "
    f"{strategy_best['Win_Rate']:.4%}"
)

print(
    f"Buy Signals: "
    f"{strategy_best['Buy_Count']}"
)

print("\nCác file đã được lưu:")
print("- xgboost_optimized.pkl")
print("- xgboost_best_threshold.txt")
print("- xgboost_best_params.txt")
print("- xgboost_predictions.csv")
print("- xgboost_test_metrics.csv")
print("- xgboost_metrics_by_symbol.csv")
print("- xgboost_portfolio_performance.csv")
print("- xgboost_portfolio_daily_returns.csv")
print("- xgboost_feature_importance.csv")
print("- xgboost_tuning_results.csv")
print("- xgboost_threshold_results.csv")
print("- xgboost_threshold_ranked.csv")

print("\nHOÀN TẤT!")