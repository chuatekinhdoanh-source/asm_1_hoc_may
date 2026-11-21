import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("RANDOM FOREST - STOCK MARKET PREDICTION")
print("=" * 70)

train_data = pd.read_csv("train_model.csv")
test_data = pd.read_csv("test_model.csv")
feature_data = pd.read_csv("data_feature.csv")

for df in [train_data, test_data, feature_data]:
    df["Date"] = pd.to_datetime(df["Date"])

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

target = "Target"

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("\nSố Features:", len(features))
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# 3. CHIA TRAIN INTERNAL / VALIDATION THEO THỜI GIAN
# ============================================================

train_internal_list = []
validation_list = []

for symbol in train_data["Symbol"].unique():

    symbol_data = (
        train_data[train_data["Symbol"] == symbol]
        .sort_values("Date")
    )

    split_index = int(len(symbol_data) * 0.8)

    train_internal_list.append(symbol_data.iloc[:split_index])
    validation_list.append(symbol_data.iloc[split_index:])

train_internal = pd.concat(train_internal_list, ignore_index=True)
validation = pd.concat(validation_list, ignore_index=True)

X_train_internal = train_internal[features]
y_train_internal = train_internal[target]

X_validation = validation[features]
y_validation = validation[target]

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Train Internal:", X_train_internal.shape)
print("Validation:", X_validation.shape)
print("Final Train:", X_train.shape)
print("Test:", X_test.shape)


# ============================================================
# 4. RANDOM FOREST - HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST - HYPERPARAMETER TUNING")
print("=" * 70)

parameter_grid = {
    "n_estimators": [200, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt"]
}

parameter_list = list(ParameterGrid(parameter_grid))

best_validation_auc = -1
best_params = None
best_validation_model = None

tuning_results = []

for i, params in enumerate(parameter_list, start=1):

    model = RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train_internal, y_train_internal)

    validation_probabilities = model.predict_proba(X_validation)[:, 1]

    validation_auc = roc_auc_score(
        y_validation,
        validation_probabilities
    )

    validation_predictions = (
        validation_probabilities >= 0.5
    ).astype(int)

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    tuning_results.append({
        "Parameters": str(params),
        "Validation_AUC": validation_auc,
        "Validation_F1": validation_f1
    })

    print(
        f"[{i}/{len(parameter_list)}] "
        f"AUC={validation_auc:.4f} | "
        f"F1={validation_f1:.4f} | "
        f"Params={params}"
    )

    if validation_auc > best_validation_auc:

        best_validation_auc = validation_auc
        best_params = params
        best_validation_model = model


# Lưu kết quả tuning

tuning_results_df = pd.DataFrame(tuning_results)

tuning_results_df = tuning_results_df.sort_values(
    "Validation_AUC",
    ascending=False
)

tuning_results_df.to_csv(
    "random_forest_tuning_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("BEST RANDOM FOREST PARAMETERS")
print("=" * 70)

print("Best Parameters:", best_params)
print(f"Best Validation AUC: {best_validation_auc:.4f}")

print("\nTop 10 Hyperparameter Results:")

print(
    tuning_results_df.head(10).to_string(index=False)
)


# ============================================================
# 5. TỐI ƯU THRESHOLD THEO CHIẾN LƯỢC GIAO DỊCH
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION - PORTFOLIO BASED")
print("=" * 70)

validation_probabilities = (
    best_validation_model.predict_proba(X_validation)[:, 1]
)

validation_trading_data = validation.copy()

validation_trading_data["Probability"] = (
    validation_probabilities
)

# Lấy giá ngày kế tiếp để tính lợi nhuận

next_price_data = feature_data[
    ["Symbol", "Date", "Next_Adj_Close"]
].copy()

validation_trading_data = validation_trading_data.merge(
    next_price_data,
    on=["Symbol", "Date"],
    how="left"
)

validation_trading_data["Next_Return"] = (
    validation_trading_data["Next_Adj_Close"]
    / validation_trading_data["Adj_Close"]
    - 1
)

validation_trading_data = validation_trading_data.dropna(
    subset=["Next_Return"]
)


def evaluate_validation_strategy(data, threshold):

    strategy_data = data.copy()

    strategy_data["Buy_Signal"] = (
        strategy_data["Probability"] >= threshold
    ).astype(int)

    strategy_data["Strategy_Return"] = (
        strategy_data["Next_Return"]
        * strategy_data["Buy_Signal"]
    )

    # 8 mã cổ phiếu, mỗi mã chiếm 1/8 danh mục.
    daily_returns = (
        strategy_data.groupby("Date")["Strategy_Return"]
        .sum() / 8
    )

    total_return = (
        (1 + daily_returns).prod() - 1
    )

    daily_volatility = daily_returns.std()

    if daily_volatility > 0:

        sharpe_ratio = (
            daily_returns.mean()
            / daily_volatility
            * np.sqrt(252)
        )

    else:

        sharpe_ratio = 0

    cumulative_returns = (
        1 + daily_returns
    ).cumprod()

    running_max = cumulative_returns.cummax()

    drawdown = (
        cumulative_returns / running_max - 1
    )

    max_drawdown = drawdown.min()

    buy_signals = strategy_data["Buy_Signal"].sum()

    buy_rate = (
        buy_signals / len(strategy_data)
        if len(strategy_data) > 0
        else 0
    )

    bought_data = strategy_data[
        strategy_data["Buy_Signal"] == 1
    ]

    if len(bought_data) > 0:

        win_rate = (
            bought_data["Next_Return"] > 0
        ).mean()

    else:

        win_rate = 0

    predictions = (
        strategy_data["Probability"] >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        strategy_data["Target"],
        predictions
    )

    precision = precision_score(
        strategy_data["Target"],
        predictions,
        zero_division=0
    )

    recall = recall_score(
        strategy_data["Target"],
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        strategy_data["Target"],
        predictions,
        zero_division=0
    )

    return {
        "Threshold": threshold,
        "Total_Return": total_return,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "Daily_Volatility": daily_volatility,
        "Buy_Signals": buy_signals,
        "Buy_Rate": buy_rate,
        "Win_Rate": win_rate,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Number_Of_Days": len(daily_returns)
    }


threshold_results = []

for threshold in np.arange(0.30, 0.701, 0.01):

    result = evaluate_validation_strategy(
        validation_trading_data,
        threshold
    )

    threshold_results.append(result)

threshold_results_df = pd.DataFrame(
    threshold_results
)

threshold_results_df.to_csv(
    "random_forest_threshold_results.csv",
    index=False
)

# Giới hạn tỷ lệ mua để tránh chiến lược mua quá ít
# hoặc gần như mua tất cả cổ phiếu.

valid_thresholds = threshold_results_df[
    (threshold_results_df["Buy_Rate"] >= 0.05)
    &
    (threshold_results_df["Buy_Rate"] <= 0.70)
]

if len(valid_thresholds) == 0:

    valid_thresholds = threshold_results_df.copy()

valid_thresholds = valid_thresholds.sort_values(
    ["Sharpe_Ratio", "Total_Return"],
    ascending=False
)

best_threshold_row = valid_thresholds.iloc[0]

best_threshold = float(
    best_threshold_row["Threshold"]
)

print("\n" + "=" * 70)
print("BEST THRESHOLD - VALIDATION")
print("=" * 70)

print(f"Best Threshold: {best_threshold:.2f}")
print(
    f"Validation Sharpe: "
    f"{best_threshold_row['Sharpe_Ratio']:.4f}"
)
print(
    f"Validation Total Return: "
    f"{best_threshold_row['Total_Return'] * 100:.2f}%"
)
print(
    f"Validation F1: "
    f"{best_threshold_row['F1']:.4f}"
)

print("\nTop 10 Threshold:")

print(
    valid_thresholds.head(10).to_string(index=False)
)

valid_thresholds.to_csv(
    "random_forest_threshold_ranked.csv",
    index=False
)


# ============================================================
# 6. HUẤN LUYỆN FINAL MODEL TRÊN TOÀN BỘ TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAIN FINAL RANDOM FOREST MODEL")
print("=" * 70)

final_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

final_model.fit(X_train, y_train)

print("Đã huấn luyện Final Random Forest trên toàn bộ Train.")


# ============================================================
# 7. DỰ ĐOÁN TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

y_test_pred_050 = (
    y_test_pred_proba >= 0.50
).astype(int)

y_test_pred_optimized = (
    y_test_pred_proba >= best_threshold
).astype(int)


def calculate_metrics(y_true, y_pred, y_proba):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, y_proba)
    }


metrics_050 = calculate_metrics(
    y_test,
    y_test_pred_050,
    y_test_pred_proba
)

metrics_optimized = calculate_metrics(
    y_test,
    y_test_pred_optimized,
    y_test_pred_proba
)

test_results = pd.DataFrame([
    {
        "Model": "Random Forest Threshold 0.50",
        **metrics_050
    },
    {
        "Model": f"Random Forest Threshold {best_threshold:.2f}",
        **metrics_optimized
    }
])

print(test_results.to_string(index=False))

test_results.to_csv(
    "random_forest_test_metrics.csv",
    index=False
)


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - OPTIMIZED THRESHOLD")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_test_pred_optimized,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_test_pred_optimized
)

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nInterpretation:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# 9. ĐÁNH GIÁ TỪNG MÃ CỔ PHIẾU
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST - EVALUATION BY STOCK")
print("=" * 70)

test_evaluation_data = test_data.copy()

test_evaluation_data["Probability"] = (
    y_test_pred_proba
)

test_evaluation_data["Prediction"] = (
    y_test_pred_optimized
)

symbol_results = []

for symbol in test_evaluation_data["Symbol"].unique():

    symbol_data = test_evaluation_data[
        test_evaluation_data["Symbol"] == symbol
    ]

    symbol_y_true = symbol_data["Target"]

    symbol_y_pred = symbol_data["Prediction"]

    symbol_y_proba = symbol_data["Probability"]

    symbol_auc = roc_auc_score(
        symbol_y_true,
        symbol_y_proba
    )

    result = {
        "Symbol": symbol,
        "Accuracy": accuracy_score(
            symbol_y_true,
            symbol_y_pred
        ),
        "Precision": precision_score(
            symbol_y_true,
            symbol_y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            symbol_y_true,
            symbol_y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            symbol_y_true,
            symbol_y_pred,
            zero_division=0
        ),
        "ROC_AUC": symbol_auc,
        "Predict_Up": int(symbol_y_pred.sum()),
        "Predict_Down": int(
            len(symbol_y_pred) - symbol_y_pred.sum()
        ),
        "Actual_Up": int(symbol_y_true.sum()),
        "Actual_Down": int(
            len(symbol_y_true) - symbol_y_true.sum()
        )
    }

    symbol_results.append(result)

symbol_results_df = pd.DataFrame(
    symbol_results
)

symbol_results_df = symbol_results_df.sort_values(
    "F1",
    ascending=False
)

print(symbol_results_df.to_string(index=False))

symbol_results_df.to_csv(
    "random_forest_metrics_by_symbol.csv",
    index=False
)

print("\nBest stock by F1:")
print(symbol_results_df.iloc[0].to_string())

print("\nWorst stock by F1:")
print(symbol_results_df.iloc[-1].to_string())


# ============================================================
# 10. PORTFOLIO BACKTEST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST - PORTFOLIO BACKTEST")
print("=" * 70)

portfolio_data = test_data.copy()

portfolio_data["Probability"] = (
    y_test_pred_proba
)

portfolio_data["Buy_Signal"] = (
    portfolio_data["Probability"] >= best_threshold
).astype(int)

portfolio_data = portfolio_data.merge(
    feature_data[
        ["Symbol", "Date", "Next_Adj_Close"]
    ],
    on=["Symbol", "Date"],
    how="left"
)

portfolio_data["Next_Return"] = (
    portfolio_data["Next_Adj_Close"]
    / portfolio_data["Adj_Close"]
    - 1
)

portfolio_data = portfolio_data.dropna(
    subset=["Next_Return"]
)

portfolio_data["Strategy_Return"] = (
    portfolio_data["Next_Return"]
    * portfolio_data["Buy_Signal"]
)

portfolio_data["Buy_Hold_Return"] = (
    portfolio_data["Next_Return"]
)


# ============================================================
# 11. DAILY PORTFOLIO RETURNS
# ============================================================

# Equal weight 8 mã cổ phiếu.
# Không mua: phần vốn đó được coi là tiền mặt, lợi nhuận = 0.

daily_strategy_returns = (
    portfolio_data.groupby("Date")["Strategy_Return"]
    .sum() / 8
)

daily_buy_hold_returns = (
    portfolio_data.groupby("Date")["Buy_Hold_Return"]
    .sum() / 8
)


def calculate_portfolio_performance(daily_returns):

    daily_returns = daily_returns.dropna()

    total_return = (
        (1 + daily_returns).prod() - 1
    )

    average_daily_return = daily_returns.mean()

    daily_volatility = daily_returns.std()

    if daily_volatility > 0:

        sharpe_ratio = (
            average_daily_return
            / daily_volatility
            * np.sqrt(252)
        )

    else:

        sharpe_ratio = 0

    cumulative_returns = (
        1 + daily_returns
    ).cumprod()

    running_max = cumulative_returns.cummax()

    drawdown = (
        cumulative_returns / running_max - 1
    )

    max_drawdown = drawdown.min()

    return {
        "Total_Return": total_return,
        "Average_Daily_Return": average_daily_return,
        "Daily_Volatility": daily_volatility,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "Number_Of_Days": len(daily_returns)
    }


strategy_performance = calculate_portfolio_performance(
    daily_strategy_returns
)

buy_hold_performance = calculate_portfolio_performance(
    daily_buy_hold_returns
)


# ============================================================
# 12. WIN RATE / BUY SIGNALS
# ============================================================

total_buy_signals = int(
    portfolio_data["Buy_Signal"].sum()
)

bought_data = portfolio_data[
    portfolio_data["Buy_Signal"] == 1
]

if len(bought_data) > 0:

    win_rate = (
        bought_data["Next_Return"] > 0
    ).mean()

else:

    win_rate = 0


portfolio_results = pd.DataFrame([
    {
        "Strategy": (
            f"Random Forest Threshold {best_threshold:.2f}"
        ),
        **strategy_performance,
        "Total_Buy_Signals": total_buy_signals,
        "Win_Rate": win_rate
    },
    {
        "Strategy": "Buy & Hold",
        **buy_hold_performance,
        "Total_Buy_Signals": np.nan,
        "Win_Rate": np.nan
    }
])

print("\n" + "=" * 70)
print("PORTFOLIO PERFORMANCE")
print("=" * 70)

print(
    portfolio_results.to_string(index=False)
)

portfolio_results.to_csv(
    "random_forest_portfolio_performance.csv",
    index=False
)


# ============================================================
# 13. LƯU DAILY RETURNS
# ============================================================

daily_portfolio_data = pd.DataFrame({
    "Date": daily_strategy_returns.index,
    "Random_Forest_Return": daily_strategy_returns.values,
    "Buy_Hold_Return": daily_buy_hold_returns.values
})

daily_portfolio_data.to_csv(
    "random_forest_portfolio_daily_returns.csv",
    index=False
)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST - FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": final_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print(feature_importance.to_string(index=False))

feature_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)


# ============================================================
# 15. LƯU MODEL VÀ THRESHOLD
# ============================================================

joblib.dump(
    final_model,
    "random_forest_optimized.pkl"
)

with open(
    "random_forest_best_threshold.txt",
    "w"
) as f:

    f.write(str(best_threshold))

with open(
    "random_forest_best_params.txt",
    "w"
) as f:

    f.write(str(best_params))


# ============================================================
# 16. LƯU DỰ ĐOÁN TEST
# ============================================================

prediction_results = test_data[
    ["Date", "Symbol", "Target"]
].copy()

prediction_results["Probability_Up"] = (
    y_test_pred_proba
)

prediction_results["Prediction_050"] = (
    y_test_pred_050
)

prediction_results["Prediction_Optimized"] = (
    y_test_pred_optimized
)

prediction_results["Threshold"] = (
    best_threshold
)

prediction_results.to_csv(
    "random_forest_predictions.csv",
    index=False
)


# ============================================================
# 17. HOÀN TẤT
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 70)

print("Best Hyperparameters:", best_params)
print("Best Threshold:", best_threshold)

print("\nFiles đã tạo:")

files_created = [
    "random_forest_optimized.pkl",
    "random_forest_best_threshold.txt",
    "random_forest_best_params.txt",
    "random_forest_predictions.csv",
    "random_forest_test_metrics.csv",
    "random_forest_metrics_by_symbol.csv",
    "random_forest_portfolio_performance.csv",
    "random_forest_portfolio_daily_returns.csv",
    "random_forest_feature_importance.csv",
    "random_forest_tuning_results.csv",
    "random_forest_threshold_results.csv",
    "random_forest_threshold_ranked.csv"
]

for file_name in files_created:
    print("-", file_name)

print("\nHoàn thành toàn bộ Random Forest pipeline!")