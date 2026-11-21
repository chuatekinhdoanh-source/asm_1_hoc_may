import pandas as pd

# =========================================================
# 1. CONFIGURATION
# =========================================================

INPUT_FILE = "data_feature.csv"

TRAIN_RATIO = 0.8

print("=" * 60)
print("PREPARE TRAIN / TEST DATA")
print("=" * 60)


# =========================================================
# 2. LOAD FEATURE DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Xác định cột mã cổ phiếu
if "Symbol" in df.columns:
    SYMBOL_COLUMN = "Symbol"

elif "Original_Symbol" in df.columns:
    SYMBOL_COLUMN = "Original_Symbol"

else:
    raise ValueError(
        "Không tìm thấy cột mã cổ phiếu"
    )

# Đảm bảo Symbol tồn tại
if "Symbol" not in df.columns:
    df["Symbol"] = df[SYMBOL_COLUMN]

# Chuẩn hóa mã
df["Symbol"] = df["Symbol"].astype(str).str.strip()

# Sắp xếp
df = df.sort_values(
    ["Symbol", "Date"]
).reset_index(drop=True)

print("Tổng dữ liệu:", df.shape)
print("Số mã:", df["Symbol"].nunique())
print("Danh sách mã:", df["Symbol"].unique())


# =========================================================
# 3. DEFINE FEATURES
# =========================================================

# Các feature đồng bộ với feature_engineering.py
feature_columns = [
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
    "Price_MA20_Ratio",
]

# Kiểm tra feature tồn tại
missing_features = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Thiếu các feature: {missing_features}"
    )

# Kiểm tra Target
if "Target" not in df.columns:
    raise ValueError(
        "Không tìm thấy cột Target"
    )

print("\nSố lượng Features:", len(feature_columns))
print("Features:", feature_columns)


# =========================================================
# 4. SPLIT BY TIME FOR EACH STOCK
# =========================================================

train_list = []
test_list = []

for symbol, stock_data in df.groupby(
    "Symbol",
    sort=False
):

    stock_data = stock_data.sort_values(
        "Date"
    ).copy()

    split_index = int(
        len(stock_data) * TRAIN_RATIO
    )

    train_stock = stock_data.iloc[
        :split_index
    ].copy()

    test_stock = stock_data.iloc[
        split_index:
    ].copy()

    train_list.append(train_stock)
    test_list.append(test_stock)

    print("\n" + "-" * 50)
    print("Mã cổ phiếu:", symbol)
    print("Tổng dữ liệu:", len(stock_data))
    print("Train:", len(train_stock))
    print("Test:", len(test_stock))

    print(
        "Train:",
        train_stock["Date"].min(),
        "->",
        train_stock["Date"].max()
    )

    print(
        "Test:",
        test_stock["Date"].min(),
        "->",
        test_stock["Date"].max()
    )


# =========================================================
# 5. COMBINE TRAIN / TEST
# =========================================================

train_data = pd.concat(
    train_list,
    ignore_index=True
)

test_data = pd.concat(
    test_list,
    ignore_index=True
)


# =========================================================
# 6. SORT DATA
# =========================================================

train_data = train_data.sort_values(
    ["Date", "Symbol"]
).reset_index(drop=True)

test_data = test_data.sort_values(
    ["Date", "Symbol"]
).reset_index(drop=True)


# =========================================================
# 7. CREATE MODEL DATA
# =========================================================

model_columns = (
    feature_columns
    + ["Target", "Symbol", "Date"]
)

train_model = train_data[
    model_columns
].copy()

test_model = test_data[
    model_columns
].copy()


# =========================================================
# 8. CHECK DATA
# =========================================================

print("\n" + "=" * 60)
print("KIỂM TRA TRAIN / TEST")
print("=" * 60)

print("\nTrain shape:", train_model.shape)
print("Test shape:", test_model.shape)

print("\nMissing values TRAIN:")
print(train_model.isnull().sum().sum())

print("\nMissing values TEST:")
print(test_model.isnull().sum().sum())

print("\nTarget TRAIN:")
print(
    train_model["Target"]
    .value_counts()
    .sort_index()
)

print("\nTarget TEST:")
print(
    test_model["Target"]
    .value_counts()
    .sort_index()
)

print("\nTỷ lệ Target TRAIN:")
print(
    train_model["Target"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)

print("\nTỷ lệ Target TEST:")
print(
    test_model["Target"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)


# =========================================================
# 9. SAVE FILES
# =========================================================

train_data.to_csv(
    "train_data.csv",
    index=False,
    encoding="utf-8-sig"
)

test_data.to_csv(
    "test_data.csv",
    index=False,
    encoding="utf-8-sig"
)

train_model.to_csv(
    "train_model.csv",
    index=False,
    encoding="utf-8-sig"
)

test_model.to_csv(
    "test_model.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 10. SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT HOÀN TẤT")
print("=" * 60)

print("Train data:", train_data.shape)
print("Test data:", test_data.shape)

print("\nTrain Model:", train_model.shape)
print("Test Model:", test_model.shape)

print("\nSố lượng dữ liệu TRAIN theo mã:")
print(
    train_data.groupby("Symbol").size()
)

print("\nSố lượng dữ liệu TEST theo mã:")
print(
    test_data.groupby("Symbol").size()
)

print("\nCác file đã tạo:")
print("- train_data.csv")
print("- test_data.csv")
print("- train_model.csv")
print("- test_model.csv")

print("\nHoàn thành!")