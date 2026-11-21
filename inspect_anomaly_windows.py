import pandas as pd

df = pd.read_csv("data_feature.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

# Những phiên Return giảm mạnh cần điều tra
anomalies = df[df["Return"] < -0.15]

print("=" * 100)
print("KIỂM TRA CÁC PHIÊN TRƯỚC VÀ SAU RETURN BẤT THƯỜNG")
print("=" * 100)

for _, anomaly in anomalies.iterrows():

    symbol = anomaly["Symbol"]
    date = anomaly["Date"]

    group = df[df["Symbol"] == symbol].reset_index(drop=True)

    position = group[group["Date"] == date].index[0]

    start = max(0, position - 3)
    end = min(len(group), position + 4)

    window = group.iloc[start:end]

    print("\n" + "=" * 100)
    print(f"MÃ: {symbol} | NGÀY BẤT THƯỜNG: {date.date()}")
    print("=" * 100)

    columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Change",
        "Return",
        "Volume",
        "Source_File"
    ]

    print(window[columns].to_string(index=False))


print("\nHOÀN TẤT KIỂM TRA")