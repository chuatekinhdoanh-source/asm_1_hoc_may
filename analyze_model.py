import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Đọc dữ liệu dự đoán
df = pd.read_csv("model_predictions_optimized.csv")

print("=" * 70)
print("PHÂN TÍCH HIỆU QUẢ MÔ HÌNH THEO TỪNG CỔ PHIẾU")
print("=" * 70)

results = []

for symbol, group in df.groupby("Symbol"):

    y_true = group["Target"]
    y_pred = group["Prediction_Threshold_05"]
    probabilities = group["Probability_Up"]

    results.append({
        "Symbol": symbol,
        "Samples": len(group),
        "Actual_Up_Rate": y_true.mean(),
        "Predicted_Up_Rate": y_pred.mean(),
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
        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        )
    })

result_df = pd.DataFrame(results)

print("\nKẾT QUẢ THEO TỪNG MÃ:")
print(
    result_df.sort_values(
        "ROC_AUC",
        ascending=False
    ).round(4).to_string(index=False)
)

print("\n" + "=" * 70)
print("THỐNG KÊ XÁC SUẤT DỰ ĐOÁN")
print("=" * 70)

print(df["Probability_Up"].describe().round(4))

print("\nPhân phối xác suất dự đoán:")
print(
    pd.cut(
        df["Probability_Up"],
        bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1],
        include_lowest=True
    ).value_counts().sort_index()
)

result_df.to_csv(
    "analysis_by_symbol.csv",
    index=False
)

print("\nĐã lưu: analysis_by_symbol.csv")