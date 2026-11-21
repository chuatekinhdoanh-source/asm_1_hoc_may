
from flask import Flask, render_template, jsonify
from pathlib import Path
import pandas as pd
import joblib

from services.data_service import (
    load_stock_data,
    get_data_summary,
    get_stock_data,
    get_stock_price_data,
    get_latest_stock_info
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# LOAD HISTORICAL DATA
# =====================================================

print("Đang tải dữ liệu cổ phiếu...")

df = load_stock_data()
summary = get_data_summary(df)

print("Đã tải dữ liệu thành công!")
print("Danh sách mã:", summary.get("stocks", []))


# =====================================================
# MACHINE LEARNING CONFIGURATION
# =====================================================

MODEL_PATH = BASE_DIR / "logistic_regression_optimized.pkl"
FEATURE_DATA_PATH = BASE_DIR / "data_feature.csv"
THRESHOLD_PATH = BASE_DIR / "best_threshold.txt"

FEATURES = [
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
    "Price_MA20_Ratio"
]


# =====================================================
# DASHBOARD
# =====================================================

@app.route("/")
def dashboard():

    stocks = summary.get("stocks", [])

    default_stock = "VCB"

    if default_stock not in stocks and stocks:
        default_stock = stocks[0]

    stock_data = get_stock_price_data(
        df,
        default_stock
    )

    latest = get_latest_stock_info(
        df,
        default_stock
    )

    return render_template(
        "dashboard.html",
        summary=summary,
        stocks=stocks,
        selected_stock=default_stock,
        stock_data=stock_data,
        latest=latest
    )


# =====================================================
# API STOCK DATA
# =====================================================

@app.route("/api/stock/<symbol>")
def stock_api(symbol):

    symbol = symbol.strip().upper()

    if symbol not in summary.get("stocks", []):

        return jsonify({
            "error": f"Không tìm thấy mã cổ phiếu: {symbol}"
        }), 404

    stock_data = get_stock_price_data(
        df,
        symbol
    )

    stock_df = get_stock_data(
        df,
        symbol
    )

    if stock_df.empty:

        return jsonify({
            "error": f"Mã {symbol} không có dữ liệu"
        }), 404

    latest = get_latest_stock_info(
        df,
        symbol
    )

    return jsonify({

        "symbol": symbol,

        "dates": stock_data["dates"],
        "prices": stock_data["prices"],
        "volumes": stock_data["volumes"],
        "ma5": stock_data["ma5"],
        "ma20": stock_data["ma20"],

        "latest_price": latest["close"],
        "latest_date": latest["date"],
        "latest_open": latest["open"],
        "latest_high": latest["high"],
        "latest_low": latest["low"],
        "latest_volume": latest["volume"],

        "records": len(stock_df)

    })


# =====================================================
# API STOCK LIST
# =====================================================

@app.route("/api/stocks")
def stocks_api():

    return jsonify({
        "stocks": summary.get("stocks", []),
        "total_stocks": summary.get("total_stocks", 0)
    })


# =====================================================
# API DATA SUMMARY
# =====================================================

@app.route("/api/summary")
def summary_api():

    return jsonify({

        "total_records": summary.get("total_records", 0),
        "total_stocks": summary.get("total_stocks", 0),
        "stocks": summary.get("stocks", []),

        "start_date": summary["start_date"].strftime("%d/%m/%Y"),
        "end_date": summary["end_date"].strftime("%d/%m/%Y"),

        "missing_values": summary.get("missing_values", 0),
        "duplicate_rows": summary.get("duplicate_rows", 0),
        "total_columns": summary.get("total_columns", 0)

    })


# =====================================================
# API MACHINE LEARNING PREDICTIONS
# =====================================================

@app.route("/api/predictions")
def predictions_api():

    try:

        # ---------------------------------------------
        # 1. CHECK FILES
        # ---------------------------------------------

        if not MODEL_PATH.exists():

            return jsonify({
                "error": "Không tìm thấy Logistic Regression model."
            }), 500

        if not FEATURE_DATA_PATH.exists():

            return jsonify({
                "error": "Không tìm thấy data_feature.csv."
            }), 500

        if not THRESHOLD_PATH.exists():

            return jsonify({
                "error": "Không tìm thấy best_threshold.txt."
            }), 500


        # ---------------------------------------------
        # 2. LOAD MODEL
        # ---------------------------------------------

        model = joblib.load(MODEL_PATH)

        with open(
            THRESHOLD_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            threshold = float(file.read().strip())


        # ---------------------------------------------
        # 3. LOAD FEATURE DATA
        # ---------------------------------------------

        feature_data = pd.read_csv(
            FEATURE_DATA_PATH
        )

        feature_data["Date"] = pd.to_datetime(
            feature_data["Date"],
            errors="coerce"
        )

        feature_data["Symbol"] = (
            feature_data["Symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
        )


        # ---------------------------------------------
        # 4. VALIDATE FEATURES
        # ---------------------------------------------

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in feature_data.columns
        ]

        if missing_features:

            return jsonify({
                "error": "Dataset thiếu các feature cần thiết.",
                "missing_features": missing_features
            }), 500


        # ---------------------------------------------
        # 5. GET LATEST ROW OF EACH STOCK
        # ---------------------------------------------

        latest_data = (
            feature_data
            .dropna(subset=["Date"])
            .sort_values(["Symbol", "Date"])
            .groupby("Symbol")
            .tail(1)
            .copy()
        )


        # ---------------------------------------------
        # 6. PREDICT PROBABILITY
        # ---------------------------------------------

        X = latest_data[FEATURES].copy()

        X = X.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )

        if X.isnull().any().any():

            return jsonify({
                "error": "Feature dữ liệu mới nhất chứa giá trị thiếu."
            }), 500


        probabilities = model.predict_proba(X)[:, 1]

        latest_data["Probability_Up"] = probabilities

        latest_data["Prediction"] = (
            latest_data["Probability_Up"] >= threshold
        ).astype(int)


        # ---------------------------------------------
        # 7. BUILD RESULTS
        # ---------------------------------------------

        results = []

        latest_date = latest_data["Date"].max()

        for _, row in latest_data.iterrows():

            probability = float(
                row["Probability_Up"]
            )

            prediction = int(
                row["Prediction"]
            )

            results.append({

                "symbol": str(row["Symbol"]),

                "date": row["Date"].strftime(
                    "%Y-%m-%d"
                ),

                "price": float(
                    row["Adj_Close"]
                ),

                "probability_up": round(
                    probability * 100,
                    2
                ),

                "threshold": round(
                    threshold * 100,
                    2
                ),

                "prediction": prediction,

                "signal": (
                    "Đạt ngưỡng mô hình"
                    if prediction == 1
                    else "Chưa đạt ngưỡng"
                ),

                "rsi": float(row["RSI"]),

                "macd": float(row["MACD"]),

                "macd_signal": float(
                    row["MACD_Signal"]
                ),

                "ma5": float(row["MA5"]),

                "ma20": float(row["MA20"]),

                "volume": int(row["Volume"]),

                "volume_ma5": float(
                    row["Volume_MA5"]
                )

            })


        # ---------------------------------------------
        # 8. SORT BY PROBABILITY
        # ---------------------------------------------

        results = sorted(
            results,
            key=lambda x: x["probability_up"],
            reverse=True
        )

        for index, item in enumerate(
            results,
            start=1
        ):

            item["rank"] = index


        # ---------------------------------------------
        # 9. DATA FRESHNESS
        # ---------------------------------------------

        today = pd.Timestamp.today().normalize()

        data_date = latest_date.normalize()

        data_age = (
            today - data_date
        ).days


        # ---------------------------------------------
        # 10. RESPONSE
        # ---------------------------------------------

        return jsonify({

            "model": "Logistic Regression",

            "threshold": threshold,

            "latest_data_date": (
                latest_date.strftime("%Y-%m-%d")
            ),

            "data_age_days": int(data_age),

            "data_is_stale": (
                data_age > 5
            ),

            "total_stocks": len(results),

            "predictions": results

        })


    except Exception as error:

        print("Prediction API error:", error)

        return jsonify({
            "error": str(error)
        }), 500


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
