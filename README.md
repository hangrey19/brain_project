# 🧠 Brain Tumor MRI Segmentation (U-Net + Mini Xception + Attention Gate)

Dự án nghiên cứu và xây dựng mô hình Deep Learning phân đoạn khối u não từ ảnh chụp MRI 2D.

## 📌 Tổng quan dự án
- **Bài toán**: Medical Image Segmentation (Binary Segmentation)[cite: 1].
- **Dataset**: LGG MRI Segmentation Dataset (Kaggle)[cite: 1].
- **Đóng góp chính**: 
  - Kết hợp **Mini Xception Encoder** với **Attention Gate** giúp tối ưu tham số (chỉ ~523K params)[cite: 1].
  - Áp dụng **Combined Loss (Dice Loss + Binary Cross-Entropy)** để khắc phục mất cân bằng dữ liệu[cite: 1].

## 📊 Kết quả huấn luyện
- **Validation Accuracy**: ~99.5%[cite: 1]
- **Validation Dice Coefficient**: ~0.64[cite: 1]

## 🛠️ Cấu trúc dự án
- `BrainProject.ipynb`: Toàn bộ Pipeline xử lý dữ liệu, định nghĩa mô hình và đánh giá.
- `requirements.txt`: Các thư viện phụ thuộc.

## 🚀 Cách chạy dự án
```bash
pip install -r requirements.txt
jupyter notebook BrainProject.ipynb