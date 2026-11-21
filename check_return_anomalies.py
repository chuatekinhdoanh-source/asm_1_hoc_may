import pandas as pd

df = pd.read_csv("data_feature.csv")
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Symbol", "Date"]).copy()

# Giá đóng cửa phiên trước
df["Previous_Close"] = df.groupby("Symbol")["Close"].shift(1)

# Return tự tính lại
df["Return_Recheck"] = (
    df["Close"] / df["Previous_Close"] - 1
)

# Lọc những phiên giảm trên 20%
anomalies = df[df["Return"] < -0.20].copy()

print("=" * 90)
print("KIỂM TRA RETURN GIẢM TRÊN 20%")
print("=" * 90)

columns = [
    "Symbol",
    "Date",
    "Previous_Close",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
    "Change",
    "Return",
    "Return_Recheck",
    "Source_File"
]

print(anomalies[columns].to_string(index=False))

print("\n" + "=" * 90)
print("KIỂM TRA RETURN CÓ KHỚP VỚI GIÁ ĐÓNG CỬA KHÔNG")
print("=" * 90)

difference = (
    anomalies["Return"] - anomalies["Return_Recheck"]
).abs()

print("Sai lệch Return lớn nhất:", difference.max())

print("\n" + "=" * 90)
print("KIỂM TRA CHANGE")
print("=" * 90)

print(
    anomalies[
        ["Symbol", "Date", "Change", "Return", "Return_Recheck"]
    ].to_string(index=False)
)

print("\nHOÀN TẤT")