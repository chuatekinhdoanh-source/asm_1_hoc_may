import pandas as pd
import re

df = pd.read_csv("data_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Symbol", "Date"]).copy()

# Giá phiên trước
df["Previous_Close"] = df.groupby("Symbol")["Close"].shift(1)

# Tính biến động dựa trên Close
df["Calculated_Change"] = df["Close"] - df["Previous_Close"]

df["Calculated_Return_Pct"] = (
    df["Close"] / df["Previous_Close"] - 1
) * 100

# Lấy phần trăm Change từ dữ liệu nguồn
def extract_change_percent(value):
    if pd.isna(value):
        return None

    match = re.search(r"\(([-+]?\d+,\d+)%\)", str(value))

    if match:
        return float(match.group(1).replace(",", "."))

    return None

df["Source_Change_Pct"] = df["Change"].apply(
    extract_change_percent
)

# So sánh các trường
df["Pct_Difference"] = (
    df["Calculated_Return_Pct"] - df["Source_Change_Pct"]
).abs()

# Lọc các dòng có sai lệch lớn
anomalies = df[
    (df["Calculated_Return_Pct"].abs() > 20) |
    (df["Pct_Difference"] > 2)
].copy()

print("=" * 100)
print("KIỂM TRA TÍNH NHẤT QUÁN GIÁ")
print("=" * 100)

columns = [
    "Symbol",
    "Date",
    "Previous_Close",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
    "Calculated_Return_Pct",
    "Source_Change_Pct",
    "Pct_Difference",
    "Source_File"
]

print(anomalies[columns].to_string(index=False))

print("\nTổng số dòng bất thường:", len(anomalies))

print("\nHOÀN TẤT")