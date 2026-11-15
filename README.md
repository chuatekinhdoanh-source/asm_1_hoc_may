# Stock Market Machine Learning

## Giới thiệu
Dự án ứng dụng các thuật toán học máy để phân tích và dự đoán xu hướng thị trường chứng khoán Việt Nam.

## Phạm vi dữ liệu
Nghiên cứu 8 mã cổ phiếu ngân hàng:
- VCB
- BID
- CTG
- TCB
- MBB
- ACB
- VPB
- STB

## Thuật toán
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

## Cấu trúc dự án
- `data_excel/`: Dữ liệu Excel ban đầu.
- `data_raw.csv`: Dữ liệu tổng hợp.
- `data_clean.csv`: Dữ liệu sau làm sạch.
- `data_feature.csv`: Dữ liệu sau tạo đặc trưng.
- `data_model.csv`: Dữ liệu phục vụ huấn luyện.
- `giai_doan_1.ipynb`: Notebook thực hiện phân tích và xây dựng mô hình.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Thực hiện
Mở file `giai_doan_1.ipynb` bằng Jupyter Notebook hoặc VS Code để chạy các bước xử lý dữ liệu và huấn luyện mô hình.