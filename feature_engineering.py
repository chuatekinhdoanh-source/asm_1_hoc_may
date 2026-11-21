import pandas as pd
import numpy as np

# =========================================================
# 1. CONFIGURATION
# =========================================================

INPUT_FILE = "data_clean.csv"
OUTPUT_FILE = "data_feature.csv"

print("=" * 60)
print("FEATURE ENGINEERING - STOCK MARKET")
print("=" * 60)


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("Dữ liệu ban đầu:", df.shape)
print("Các cột ban đầu:", df.columns.tolist())


# =========================================================
# 3. IDENTIFY STOCK SYMBOL COLUMN
# =========================================================

# Dataset hiện tại sử dụng Original_Symbol
if "Original_Symbol" in df.columns:
    SYMBOL_COLUMN = "Original_Symbol"

elif "Symbol" in df.columns:
    SYMBOL_COLUMN = "Symbol"

else:
    raise ValueError(
        "Không tìm thấy cột mã cổ phiếu "
        "(Original_Symbol hoặc Symbol)"
    )

# Chuẩn hóa tên mã cổ phiếu
df[SYMBOL_COLUMN] = df[SYMBOL_COLUMN].astype(str).str.strip()

# Tạo thêm Symbol để tương thích với các file khác trong dự án
if SYMBOL_COLUMN == "Original_Symbol":
    df["Symbol"] = df["Original_Symbol"]


# =========================================================
# 4. CLEAN AND PREPARE DATA
# =========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

numeric_columns = [
    "Close",
    "Adj_Close",
    "Open",
    "High",
    "Low",
    "Volume",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# Các cột quan trọng phải có
required_columns = [
    SYMBOL_COLUMN,
    "Date",
    "Adj_Close",
    "Volume",
]

df = df.dropna(
    subset=required_columns
)

# Giá điều chỉnh phải lớn hơn 0
df = df[
    (df["Adj_Close"] > 0) &
    (df["Volume"] >= 0)
]

# Sắp xếp theo từng mã và thời gian
df = df.sort_values(
    by=[SYMBOL_COLUMN, "Date"]
).reset_index(drop=True)

print("Sau khi làm sạch:", df.shape)
print("Số mã cổ phiếu:", df[SYMBOL_COLUMN].nunique())
print("Danh sách mã:", df[SYMBOL_COLUMN].unique())


# =========================================================
# 5. FEATURE ENGINEERING FUNCTION
# =========================================================

def create_features(stock_data):

    stock_data = stock_data.copy()

    stock_data = stock_data.sort_values(
        "Date"
    ).reset_index(drop=True)

    price = stock_data["Adj_Close"]

    # -----------------------------------------------------
    # RETURN
    # -----------------------------------------------------

    stock_data["Return"] = price.pct_change()

    # -----------------------------------------------------
    # MOVING AVERAGES
    # -----------------------------------------------------

    stock_data["MA5"] = price.rolling(5).mean()
    stock_data["MA20"] = price.rolling(20).mean()
    stock_data["MA50"] = price.rolling(50).mean()

    # -----------------------------------------------------
    # EXPONENTIAL MOVING AVERAGES
    # -----------------------------------------------------

    stock_data["EMA12"] = price.ewm(
        span=12,
        adjust=False
    ).mean()

    stock_data["EMA26"] = price.ewm(
        span=26,
        adjust=False
    ).mean()

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    stock_data["MACD"] = (
        stock_data["EMA12"] -
        stock_data["EMA26"]
    )

    stock_data["MACD_Signal"] = (
        stock_data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    stock_data["MACD_Hist"] = (
        stock_data["MACD"] -
        stock_data["MACD_Signal"]
    )

    # -----------------------------------------------------
    # RSI 14
    # -----------------------------------------------------

    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    stock_data["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

    stock_data["Volatility20"] = (
        stock_data["Return"]
        .rolling(20)
        .std()
    )

    # -----------------------------------------------------
    # VOLUME FEATURES
    # -----------------------------------------------------

    stock_data["Volume_MA5"] = (
        stock_data["Volume"]
        .rolling(5)
        .mean()
    )

    stock_data["Volume_Change"] = (
        stock_data["Volume"]
        .pct_change()
    )

    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------

    stock_data["Lag_1"] = (
        stock_data["Return"].shift(1)
    )

    stock_data["Lag_2"] = (
        stock_data["Return"].shift(2)
    )

    stock_data["Lag_3"] = (
        stock_data["Return"].shift(3)
    )

    # -----------------------------------------------------
    # PRICE / MOVING AVERAGE RATIO
    # -----------------------------------------------------

    stock_data["Price_MA5_Ratio"] = (
        price / stock_data["MA5"]
    )

    stock_data["Price_MA20_Ratio"] = (
        price / stock_data["MA20"]
    )

    # -----------------------------------------------------
    # NEXT DAY ADJUSTED CLOSE
    # -----------------------------------------------------

    stock_data["Next_Adj_Close"] = (
        price.shift(-1)
    )

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------
    # 1: Giá điều chỉnh ngày tiếp theo tăng
    # 0: Giá điều chỉnh ngày tiếp theo không tăng

    stock_data["Target"] = (
        stock_data["Next_Adj_Close"] >
        stock_data["Adj_Close"]
    ).astype(float)

    # Ngày cuối cùng của mỗi mã không có giá ngày sau
    stock_data.loc[
        stock_data["Next_Adj_Close"].isna(),
        "Target"
    ] = np.nan

    return stock_data


# =========================================================
# 6. APPLY FEATURE ENGINEERING SEPARATELY BY STOCK
# =========================================================

feature_data_list = []

for symbol, stock_data in df.groupby(
    SYMBOL_COLUMN,
    sort=False
):

    print(f"Đang xử lý mã: {symbol}")

    stock_features = create_features(
        stock_data
    )

    feature_data_list.append(
        stock_features
    )

# Ghép dữ liệu các mã lại
df_feature = pd.concat(
    feature_data_list,
    ignore_index=True
)

print(
    "Sau Feature Engineering:",
    df_feature.shape
)


# =========================================================
# 7. REMOVE INVALID FEATURE ROWS
# =========================================================

feature_columns = [
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
    "Price_MA20_Ratio",
    "Next_Adj_Close",
    "Target",
]

# Loại giá trị vô hạn
df_feature = df_feature.replace(
    [np.inf, -np.inf],
    np.nan
)

# Chỉ loại các hàng thiếu đặc trưng cần thiết
df_feature = df_feature.dropna(
    subset=feature_columns
)

# Sắp xếp dữ liệu đầu ra
df_feature = df_feature.sort_values(
    by=[SYMBOL_COLUMN, "Date"]
).reset_index(drop=True)


# =========================================================
# 8. SAVE DATA
# =========================================================

df_feature.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 9. RESULTS
# =========================================================

print("\n" + "=" * 60)
print("HOÀN THÀNH FEATURE ENGINEERING")
print("=" * 60)

print("Kích thước dữ liệu:", df_feature.shape)

print("\nPhân bố Target:")
print(df_feature["Target"].value_counts())

print("\nTỷ lệ Target:")
print(
    df_feature["Target"]
    .value_counts(normalize=True)
)

print("\nSố lượng dữ liệu theo mã:")
print(
    df_feature.groupby(SYMBOL_COLUMN).size()
)

print("\nCác mã cổ phiếu:")
print(
    df_feature[SYMBOL_COLUMN]
    .unique()
)

print("\nCác cột dữ liệu:")
print(
    df_feature.columns.tolist()
)

print("\nKiểm tra cột Symbol:")
print("Symbol tồn tại:", "Symbol" in df_feature.columns)

print("\nKiểm tra cột Original_Symbol:")
print(
    "Original_Symbol tồn tại:",
    "Original_Symbol" in df_feature.columns
)

print("\nĐã lưu file:", OUTPUT_FILE)