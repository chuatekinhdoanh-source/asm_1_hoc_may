import pandas as pd

df = pd.read_csv("data_feature.csv")
df["Date"] = pd.to_datetime(df["Date"])

print("=" * 70)
print("1. KIỂM TRA RETURN BẤT THƯỜNG")
print("=" * 70)

extreme_return = df.sort_values("Return").head(10)

print("\n10 Return thấp nhất:")
print(extreme_return[
    ["Symbol", "Date", "Close", "Return", "Target", "Source_File"]
].to_string(index=False))

print("\n10 Return cao nhất:")
print(
    df.sort_values("Return", ascending=False).head(10)[
        ["Symbol", "Date", "Close", "Return", "Target", "Source_File"]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("2. KIỂM TRA VOLUME_CHANGE BẤT THƯỜNG")
print("=" * 70)

print("\n10 Volume_Change lớn nhất:")
print(df.sort_values("Volume_Change", ascending=False).head(10)[
    ["Symbol", "Date", "Volume", "Volume_Change",
     "Volume_Ratio", "Target", "Source_File"]
].to_string(index=False))


print("\n" + "=" * 70)
print("3. KIỂM TRA TARGET CÓ ĐƯỢC TẠO ĐÚNG KHÔNG")
print("=" * 70)

df = df.sort_values(["Symbol", "Date"])

df["Next_Close_Check"] = (
    df.groupby("Symbol")["Close"].shift(-1)
)

df["Target_Check"] = (
    df["Next_Close_Check"] > df["Close"]
).astype(int)

# Chỉ kiểm tra những dòng có Next_Close
valid = df.dropna(subset=["Next_Close"])

wrong_next_close = (
    (valid["Next_Close"] - valid["Next_Close_Check"]).abs() > 0.000001
)

wrong_target = (
    valid["Target"] != valid["Target_Check"]
)

print("Số Next_Close sai:", wrong_next_close.sum())
print("Số Target sai:", wrong_target.sum())

if wrong_next_close.sum() > 0:
    print("\nCác dòng Next_Close sai:")
    print(valid[wrong_next_close][
        ["Symbol", "Date", "Close", "Next_Close",
         "Next_Close_Check"]
    ].head(20).to_string(index=False))

if wrong_target.sum() > 0:
    print("\nCác dòng Target sai:")
    print(valid[wrong_target][
        ["Symbol", "Date", "Close", "Next_Close",
         "Target", "Target_Check"]
    ].head(20).to_string(index=False))


print("\n" + "=" * 70)
print("4. KIỂM TRA HIGH, LOW, OPEN, CLOSE")
print("=" * 70)

invalid_price = df[
    (df["High"] < df["Low"]) |
    (df["High"] < df["Open"]) |
    (df["High"] < df["Close"]) |
    (df["Low"] > df["Open"]) |
    (df["Low"] > df["Close"])
]

print("Số dòng giá bất thường:", len(invalid_price))

if len(invalid_price) > 0:
    print(invalid_price[
        ["Symbol", "Date", "Open", "High", "Low", "Close"]
    ].head(20).to_string(index=False))


print("\n" + "=" * 70)
print("5. KIỂM TRA TARGET THEO TỪNG MÃ")
print("=" * 70)

print(
    df.groupby("Symbol")["Target"]
    .agg(["count", "mean"])
    .assign(Target_Up_Percent=lambda x: x["mean"] * 100)
)

print("\nHOÀN TẤT AUDIT DATA")