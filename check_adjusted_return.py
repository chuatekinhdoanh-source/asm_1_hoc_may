import pandas as pd

df = pd.read_csv("data_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Symbol", "Date"])

# Tính lợi suất từ giá điều chỉnh
df["Adj_Return"] = (
    df.groupby("Symbol")["Adj_Close"]
    .pct_change()
)

# Trích phần % từ cột Change
df["Source_Change_Pct"] = (
    df["Change"]
    .astype(str)
    .str.extract(r"\(([-+]?\d+[.,]?\d*)%")
)

df["Source_Change_Pct"] = (
    df["Source_Change_Pct"]
    .str.replace(",", ".", regex=False)
    .astype(float) / 100
)

# So sánh
df["Difference"] = (
    df["Adj_Return"] - df["Source_Change_Pct"]
)

print("=" * 80)
print("KIỂM TRA ADJ_CLOSE SO VỚI CHANGE NGUỒN")
print("=" * 80)

valid = df.dropna(subset=["Adj_Return", "Source_Change_Pct"])

print(f"Tổng số dòng kiểm tra: {len(valid)}")

print(
    "Sai lệch trung bình tuyệt đối:",
    valid["Difference"].abs().mean()
)

print(
    "Sai lệch trung vị:",
    valid["Difference"].abs().median()
)

print(
    "Số dòng sai lệch trên 1%:",
    (valid["Difference"].abs() > 0.01).sum()
)

print("\n10 dòng sai lệch lớn nhất:")

print(
    valid.assign(
        Abs_Difference=valid["Difference"].abs()
    )
    .sort_values("Abs_Difference", ascending=False)
    [
        [
            "Symbol",
            "Date",
            "Close",
            "Adj_Close",
            "Change",
            "Adj_Return",
            "Source_Change_Pct",
            "Difference"
        ]
    ]
    .head(10)
    .to_string(index=False)
)