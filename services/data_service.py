from pathlib import Path
import pandas as pd

# =========================================
# PATH CONFIGURATION
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_excel" / "raw"


# =========================================
# LOAD EXCEL DATA
# =========================================

def load_stock_data():
    """
    Đọc tất cả file Excel trong data_excel/raw,
    ghép thành một DataFrame chung.
    """

    excel_files = list(RAW_DIR.glob("*.xlsx"))

    if not excel_files:
        raise FileNotFoundError(
            f"Không tìm thấy file Excel trong: {RAW_DIR}"
        )

    all_data = []

    for file in excel_files:
        try:
            df = pd.read_excel(file)

            if df.empty:
                print(f"Bỏ qua file rỗng: {file.name}")
                continue

            df["File nguồn"] = file.name
            all_data.append(df)

            print(f"Đã đọc: {file.name} | {len(df)} dòng")

        except Exception as e:
            print(f"Lỗi đọc {file.name}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Ghép dữ liệu
    data = pd.concat(all_data, ignore_index=True)

    # =====================================
    # DATA CLEANING
    # =====================================

    # Chuẩn hóa ngày
    data["Ngày"] = pd.to_datetime(
        data["Ngày"],
        dayfirst=True,
        errors="coerce"
    )

    # Chuẩn hóa mã cổ phiếu
    data["Mã"] = (
        data["Mã"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Các cột số
    numeric_columns = [
        "Giá đóng cửa",
        "Giá điều chỉnh",
        "Khối lượng khớp lệnh",
        "Giá trị khớp lệnh",
        "Khối lượng thỏa thuận",
        "Giá trị thỏa thuận",
        "Giá mở cửa",
        "Giá cao nhất",
        "Giá thấp nhất"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # Loại bỏ dòng không có ngày hoặc mã
    data = data.dropna(subset=["Ngày", "Mã"])

    # Loại bỏ mã không hợp lệ
    data = data[data["Mã"] != "NAN"]

    # Loại bỏ dữ liệu trùng mã và ngày
    data = data.drop_duplicates(
        subset=["Mã", "Ngày"],
        keep="first"
    )

    # Sắp xếp dữ liệu
    data = data.sort_values(
        by=["Mã", "Ngày"]
    ).reset_index(drop=True)

    print("\n===== DATA LOADED SUCCESSFULLY =====")
    print(f"Tổng bản ghi: {len(data)}")
    print(f"Số mã cổ phiếu: {data['Mã'].nunique()}")
    print(f"Các mã: {sorted(data['Mã'].unique())}")

    return data


# =========================================
# DATA SUMMARY
# =========================================

def get_data_summary(data):
    """
    Thống kê tổng quan dataset.
    """

    if data.empty:
        return {}

    return {
        "total_records": int(len(data)),
        "total_stocks": int(data["Mã"].nunique()),
        "stocks": sorted(data["Mã"].unique().tolist()),
        "start_date": data["Ngày"].min(),
        "end_date": data["Ngày"].max(),
        "missing_values": int(data.isnull().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "total_columns": int(len(data.columns))
    }


# =========================================
# GET DATA BY STOCK SYMBOL
# =========================================

def get_stock_data(data, symbol):
    """
    Lấy dữ liệu của một mã cổ phiếu.
    """

    symbol = str(symbol).strip().upper()

    stock_data = data[
        data["Mã"] == symbol
    ].copy()

    return stock_data.sort_values(
        by="Ngày"
    ).reset_index(drop=True)


# =========================================
# STOCK PRICE + MA5 + MA20 + VOLUME
# =========================================

def get_stock_price_data(data, symbol):
    """
    Trả dữ liệu biểu đồ:
    - Ngày
    - Giá đóng cửa
    - Khối lượng
    - MA5
    - MA20

    NaN được chuyển thành None để JSON trả null.
    """

    stock_data = get_stock_data(data, symbol)

    if stock_data.empty:
        return {
            "dates": [],
            "prices": [],
            "volumes": [],
            "ma5": [],
            "ma20": []
        }

    # =====================================
    # CALCULATE MOVING AVERAGES
    # =====================================

    stock_data["MA5"] = (
        stock_data["Giá đóng cửa"]
        .rolling(window=5)
        .mean()
    )

    stock_data["MA20"] = (
        stock_data["Giá đóng cửa"]
        .rolling(window=20)
        .mean()
    )

    # =====================================
    # CONVERT NaN TO JSON NULL
    # =====================================

    def clean_list(series):
        """
        Chuyển NaN/NaT thành None.
        """

        values = series.astype(object)

        values = values.where(
            pd.notna(values),
            None
        )

        return values.tolist()

    # =====================================
    # RETURN DATA
    # =====================================

    return {
        "dates": stock_data["Ngày"]
            .dt.strftime("%Y-%m-%d")
            .tolist(),

        "prices": clean_list(
            stock_data["Giá đóng cửa"]
        ),

        "volumes": clean_list(
            stock_data["Khối lượng khớp lệnh"]
        ),

        "ma5": clean_list(
            stock_data["MA5"].round(2)
        ),

        "ma20": clean_list(
            stock_data["MA20"].round(2)
        )
    }


# =========================================
# GET LATEST STOCK INFORMATION
# =========================================

def get_latest_stock_info(data, symbol):
    """
    Lấy thông tin phiên giao dịch mới nhất.
    """

    stock_data = get_stock_data(data, symbol)

    if stock_data.empty:
        return None

    latest = stock_data.iloc[-1]

    return {
        "symbol": symbol.upper(),
        "date": latest["Ngày"].strftime("%d/%m/%Y"),
        "close": float(latest["Giá đóng cửa"])
            if pd.notna(latest["Giá đóng cửa"])
            else None,

        "open": float(latest["Giá mở cửa"])
            if pd.notna(latest["Giá mở cửa"])
            else None,

        "high": float(latest["Giá cao nhất"])
            if pd.notna(latest["Giá cao nhất"])
            else None,

        "low": float(latest["Giá thấp nhất"])
            if pd.notna(latest["Giá thấp nhất"])
            else None,

        "volume": int(latest["Khối lượng khớp lệnh"])
            if pd.notna(latest["Khối lượng khớp lệnh"])
            else None
    }


# =========================================
# GET STOCK LIST
# =========================================

def get_stock_list(data):
    """
    Trả danh sách mã cổ phiếu.
    """

    if data.empty:
        return []

    return sorted(
        data["Mã"].dropna().unique().tolist()
    )