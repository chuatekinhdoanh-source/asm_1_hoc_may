# ============================================================
# PREDICT STOCK - LOGISTIC REGRESSION
# Dự đoán xu hướng giá cổ phiếu ngày tiếp theo
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

# ============================================================
# 1. CẤU HÌNH
# ============================================================

MODEL_FILE = "logistic_regression_optimized.pkl"
THRESHOLD_FILE = "best_threshold.txt"
DATA_FILE = "data_feature.csv"

THRESHOLD_DEFAULT = 0.48

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

STOCKS = [
    "ACB", "BID", "CTG", "MBB",
    "STB", "TCB", "VCB", "VPB"
]

# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 65)
print("STOCK PREDICTION SYSTEM")
print("=" * 65)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Không tìm thấy model: {MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

if os.path.exists(THRESHOLD_FILE):

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        threshold = float(file.read().strip())

else:

    threshold = THRESHOLD_DEFAULT

print(f"\nModel: {MODEL_FILE}")
print(f"Threshold: {threshold}")

# ============================================================
# 3. LOAD DATA
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu: {DATA_FILE}"
    )

data = pd.read_csv(DATA_FILE)

data["Date"] = pd.to_datetime(
    data["Date"]
)

# Kiểm tra features
missing_features = [
    feature
    for feature in FEATURES
    if feature not in data.columns
]

if missing_features:

    raise ValueError(
        f"Thiếu features: {missing_features}"
    )

print("\nĐã load dữ liệu Feature Engineering.")
print(f"Số dòng dữ liệu: {len(data)}")

# ============================================================
# 4. LẤY DỮ LIỆU MỚI NHẤT CỦA TỪNG MÃ
# ============================================================

latest_data = (
    data.sort_values("Date")
    .groupby("Symbol")
    .tail(1)
    .copy()
)

latest_data = latest_data.dropna(
    subset=FEATURES
)

latest_data = latest_data.sort_values(
    "Symbol"
)

# ============================================================
# 5. DỰ ĐOÁN
# ============================================================

X_latest = latest_data[FEATURES]

probabilities = model.predict_proba(
    X_latest
)[:, 1]

latest_data["Probability_Up"] = probabilities

latest_data["Probability_Percent"] = (
    latest_data["Probability_Up"] * 100
)

latest_data["Threshold"] = threshold

latest_data["Signal"] = np.where(
    latest_data["Probability_Up"] >= threshold,
    "TÍN HIỆU TĂNG",
    "CHƯA CÓ TÍN HIỆU TĂNG"
)

# ============================================================
# 6. PHÂN TÍCH CÁC CHỈ BÁO
# ============================================================

def analyze_stock(row):

    comments = []

    # RSI
    if row["RSI"] < 30:

        comments.append(
            "RSI thấp - vùng có thể bị bán quá mức"
        )

    elif row["RSI"] > 70:

        comments.append(
            "RSI cao - vùng có thể bị mua quá mức"
        )

    else:

        comments.append(
            "RSI nằm trong vùng trung tính"
        )

    # MA5
    if row["Adj_Close"] > row["MA5"]:

        comments.append(
            "Giá cao hơn MA5 - xu hướng ngắn hạn tích cực"
        )

    else:

        comments.append(
            "Giá thấp hơn MA5 - xu hướng ngắn hạn yếu"
        )

    # MA20
    if row["Adj_Close"] > row["MA20"]:

        comments.append(
            "Giá cao hơn MA20"
        )

    else:

        comments.append(
            "Giá thấp hơn MA20"
        )

    # MACD
    if row["MACD"] > row["MACD_Signal"]:

        comments.append(
            "MACD đang cao hơn đường Signal"
        )

    else:

        comments.append(
            "MACD đang thấp hơn đường Signal"
        )

    # Volume
    if row["Volume"] > row["Volume_MA5"]:

        comments.append(
            "Khối lượng cao hơn trung bình 5 phiên"
        )

    else:

        comments.append(
            "Khối lượng thấp hơn trung bình 5 phiên"
        )

    return " | ".join(comments)


latest_data["Technical_Analysis"] = (
    latest_data.apply(
        analyze_stock,
        axis=1
    )
)

# ============================================================
# 7. XẾP HẠNG CỔ PHIẾU
# ============================================================

latest_data = latest_data.sort_values(
    "Probability_Up",
    ascending=False
)

latest_data["Rank"] = range(
    1,
    len(latest_data) + 1
)

# ============================================================
# 8. HIỂN THỊ BẢNG
# ============================================================

print("\n" + "=" * 65)
print("PREDICTION RESULTS")
print("=" * 65)

display_columns = [
    "Symbol",
    "Date",
    "Adj_Close",
    "Probability_Percent",
    "Signal",
    "RSI",
    "MACD",
    "Rank"
]

print(
    latest_data[
        display_columns
    ].to_string(
        index=False
    )
)

# ============================================================
# 9. CHỌN MÃ CỔ PHIẾU ĐỂ PHÂN TÍCH
# ============================================================

print("\n" + "=" * 65)
print("CHI TIẾT PHÂN TÍCH CỔ PHIẾU")
print("=" * 65)

symbol_input = input(
    "\nNhập mã cổ phiếu cần phân tích (ACB/BID/CTG/MBB/STB/TCB/VCB/VPB): "
).strip().upper()

selected = latest_data[
    latest_data["Symbol"] == symbol_input
]

if selected.empty:

    print(
        "\nKhông tìm thấy mã cổ phiếu "
        "hoặc mã chưa có dữ liệu."
    )

else:

    stock = selected.iloc[0]

    print("\n" + "=" * 65)
    print(f"PHÂN TÍCH MÃ: {symbol_input}")
    print("=" * 65)

    print(f"Ngày dữ liệu: {stock['Date'].date()}")

    print(
        f"Giá gần nhất: "
        f"{stock['Adj_Close']:,.2f}"
    )

    print(
        f"Xác suất tăng: "
        f"{stock['Probability_Percent']:.2f}%"
    )

    print(
        f"Threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Tín hiệu: "
        f"{stock['Signal']}"
    )

    print(
        f"Xếp hạng: "
        f"{int(stock['Rank'])}/{len(latest_data)}"
    )

    print("\nPhân tích kỹ thuật:")

    analysis_items = (
        stock["Technical_Analysis"].split(" | ")
    )

    for item in analysis_items:

        print(f"- {item}")

    print("\nCác chỉ báo:")

    print(
        f"RSI: {stock['RSI']:.2f}"
    )

    print(
        f"MACD: {stock['MACD']:.4f}"
    )

    print(
        f"MACD Signal: "
        f"{stock['MACD_Signal']:.4f}"
    )

    print(
        f"MA5: {stock['MA5']:,.2f}"
    )

    print(
        f"MA20: {stock['MA20']:,.2f}"
    )

    print(
        f"MA50: {stock['MA50']:,.2f}"
    )

    print(
        f"Volume: {stock['Volume']:,.0f}"
    )

    print(
        f"Volume MA5: "
        f"{stock['Volume_MA5']:,.0f}"
    )

    # --------------------------------------------------------
    # CẢNH BÁO
    # --------------------------------------------------------

    print("\nCẢNH BÁO:")

    print(
        "- Đây là dự đoán xác suất từ mô hình Machine Learning."
    )

    print(
        "- Xác suất không phải cam kết giá sẽ tăng."
    )

    print(
        "- Kết quả chỉ sử dụng dữ liệu lịch sử đã có."
    )

    print(
        "- Không sử dụng dự đoán này như lời khuyên đầu tư."
    )

# ============================================================
# 10. LƯU KẾT QUẢ
# ============================================================

output_columns = [
    "Symbol",
    "Date",
    "Adj_Close",
    "Probability_Up",
    "Probability_Percent",
    "Threshold",
    "Signal",
    "Rank",
    "RSI",
    "MA5",
    "MA20",
    "MA50",
    "MACD",
    "MACD_Signal",
    "Volume",
    "Volume_MA5",
    "Technical_Analysis"
]

latest_data[
    output_columns
].to_csv(
    "stock_prediction_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("ĐÃ LƯU KẾT QUẢ")
print("=" * 65)

print(
    "File: stock_prediction_results.csv"
)

print("\nHOÀN TẤT!")