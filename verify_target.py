import pandas as pd

df = pd.read_csv("data_feature.csv")
df["Date"] = pd.to_datetime(df["Date"])

# 1. Kiểm tra Target trực tiếp bằng Next_Close
df["Target_Verify"] = (
    df["Next_Close"] > df["Close"]
).astype(int)

wrong = df[df["Target"] != df["Target_Verify"]]

print("=" * 60)
print("KIỂM TRA TARGET")
print("=" * 60)

print("Số Target không khớp Next_Close:", len(wrong))

if len(wrong) > 0:
    print(wrong[
        ["Symbol", "Date", "Close", "Next_Close",
         "Target", "Target_Verify"]
    ].to_string(index=False))


# 2. Kiểm tra các Return giảm mạnh
print("\n" + "=" * 60)
print("RETURN GIẢM MẠNH")
print("=" * 60)

extreme = df[df["Return"] < -0.20].sort_values("Return")

print(extreme[
    ["Symbol", "Date", "Open", "High", "Low", "Close",
     "Return", "Volume", "Source_File"]
].to_string(index=False))


# 3. Kiểm tra ngày khối lượng tăng bất thường
print("\n" + "=" * 60)
print("VOLUME CHANGE BẤT THƯỜNG")
print("=" * 60)

volume_extreme = df[df["Volume_Change"] > 10].sort_values(
    "Volume_Change", ascending=False
)

print(volume_extreme[
    ["Symbol", "Date", "Close", "Volume",
     "Volume_Change", "Volume_Ratio", "Source_File"]
].to_string(index=False))


print("\nHOÀN TẤT VERIFY")