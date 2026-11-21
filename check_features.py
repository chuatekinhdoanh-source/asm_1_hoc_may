import pandas as pd
import numpy as np

# =========================
# 1. Đọc dữ liệu Feature
# =========================
df = pd.read_csv("data_feature.csv")

print("=" * 60)
print("KIỂM TRA DATASET FEATURE")
print("=" * 60)

print("\nKích thước dữ liệu:", df.shape)
print("\nDanh sách cột:")
print(df.columns.tolist())

# =========================
# 2. Kiểm tra Missing Values
# =========================
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("Không có Missing Values")
else:
    print(missing)

# =========================
# 3. Kiểm tra Infinite Values
# =========================
print("\n" + "=" * 60)
print("INFINITE VALUES")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

inf_count = np.isinf(numeric_df).sum()

inf_count = inf_count[inf_count > 0]

if len(inf_count) == 0:
    print("Không có Infinite Values")
else:
    print(inf_count)

# =========================
# 4. Kiểm tra dữ liệu trùng
# =========================
print("\n" + "=" * 60)
print("DUPLICATE")
print("=" * 60)

print("Số dòng trùng:", df.duplicated().sum())

# =========================
# 5. Kiểm tra Target
# =========================
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["Target"].value_counts())

print("\nTỷ lệ Target:")
print(df["Target"].value_counts(normalize=True) * 100)

# =========================
# 6. Kiểm tra từng mã cổ phiếu
# =========================
print("\n" + "=" * 60)
print("THỐNG KÊ THEO SYMBOL")
print("=" * 60)

for symbol, group in df.groupby("Symbol"):

    group = group.sort_values("Date")

    print(f"\n--- {symbol} ---")

    print("Số dòng:", len(group))
    print("Ngày đầu:", group["Date"].min())
    print("Ngày cuối:", group["Date"].max())

    print("Tỷ lệ Target = 1:",
          round(group["Target"].mean() * 100, 2), "%")

# =========================
# 7. Kiểm tra các Feature
# =========================
print("\n" + "=" * 60)
print("THỐNG KÊ FEATURE")
print("=" * 60)

features = [
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

for feature in features:

    if feature in df.columns:

        print(f"\n{feature}")

        print("Min:", df[feature].min())
        print("Max:", df[feature].max())
        print("Mean:", df[feature].mean())
        print("Std:", df[feature].std())

# =========================
# 8. Kiểm tra tương quan Feature với Target
# =========================
print("\n" + "=" * 60)
print("FEATURE CORRELATION WITH TARGET")
print("=" * 60)

correlation = df[features + ["Target"]].corr()["Target"]

correlation = correlation.drop("Target")

correlation = correlation.sort_values(
    key=lambda x: abs(x),
    ascending=False
)

print(correlation)

# =========================
# 9. Kết luận tổng quan
# =========================
print("\n" + "=" * 60)
print("HOÀN TẤT KIỂM TRA")
print("=" * 60)