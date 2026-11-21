import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("RANDOM FOREST - STOCK DIRECTION PREDICTION")
print("=" * 65)

train_data = pd.read_csv("train_model.csv")
test_data = pd.read_csv("test_model.csv")

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "Close",
    "Adj_Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "Trading_Value",
    "Negotiated_Volume",
    "Negotiated_Value",
    "Return",
    "MA5",
    "MA20",
    "MA50",
    "Price_MA5_Ratio",
    "Price_MA20_Ratio",
    "RSI14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volatility20",
    "Volume_MA5",
    "Volume_MA20",
    "Volume_Change",
    "Volume_Ratio",
    "Daily_Range",
    "Daily_Range_Pct",
    "Open_Close_Pct"
]

X_train = train_data[features]
y_train = train_data["Target"]

X_test = test_data[features]
y_test = test_data["Target"]


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nĐang huấn luyện Random Forest...")

model.fit(X_train, y_train)

print("Huấn luyện hoàn tất!")


# ============================================================
# 4. PREDICTION
# ============================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n" + "=" * 65)
print("RANDOM FOREST EVALUATION")
print("=" * 65)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, predictions)

print(cm)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Không tăng (0)", "Tăng (1)"],
        zero_division=0
    )
)


# ============================================================
# 8. PREDICTION COUNTS
# ============================================================

print("\nSố lượng dự đoán:")

print(
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. PROBABILITY
# ============================================================

print("\nXác suất tăng trung bình:")

print(round(probabilities.mean(), 4))


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 65)
print("TOP 15 FEATURE IMPORTANCE")
print("=" * 65)

print(
    feature_importance.head(15).round(4).to_string(index=False)
)


# ============================================================
# 11. ANALYSIS BY SYMBOL
# ============================================================

test_results = test_data[
    ["Date", "Symbol", "Target"]
].copy()

test_results["Prediction"] = predictions
test_results["Probability_Up"] = probabilities

symbol_results = []

for symbol, group in test_results.groupby("Symbol"):

    symbol_results.append({
        "Symbol": symbol,
        "Samples": len(group),
        "Accuracy": accuracy_score(
            group["Target"],
            group["Prediction"]
        ),
        "Precision": precision_score(
            group["Target"],
            group["Prediction"],
            zero_division=0
        ),
        "Recall": recall_score(
            group["Target"],
            group["Prediction"],
            zero_division=0
        ),
        "F1": f1_score(
            group["Target"],
            group["Prediction"],
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            group["Target"],
            group["Probability_Up"]
        )
    })

symbol_df = pd.DataFrame(symbol_results)

print("\n" + "=" * 65)
print("RESULTS BY SYMBOL")
print("=" * 65)

print(
    symbol_df.sort_values(
        "ROC_AUC",
        ascending=False
    ).round(4).to_string(index=False)
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

test_results.to_csv(
    "random_forest_predictions.csv",
    index=False
)

symbol_df.to_csv(
    "random_forest_by_symbol.csv",
    index=False
)

feature_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

metrics = pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc
}])

metrics.to_csv(
    "random_forest_metrics.csv",
    index=False
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    "random_forest_model.pkl"
)


# ============================================================
# 14. FINISH
# ============================================================

print("\n" + "=" * 65)
print("HOÀN TẤT RANDOM FOREST")
print("=" * 65)

print("Đã lưu:")
print("- random_forest_predictions.csv")
print("- random_forest_by_symbol.csv")
print("- random_forest_feature_importance.csv")
print("- random_forest_metrics.csv")
print("- random_forest_model.pkl")