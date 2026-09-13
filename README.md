# 🧠 Brain Tumor MRI Segmentation với U-Net + Attention + Mini-Xception

Dự án deep learning phân đoạn (segmentation) khối u não từ ảnh MRI, xây dựng và huấn luyện end-to-end trên bộ dữ liệu công khai **LGG MRI Segmentation** (Kaggle). Mục tiêu: tự động khoanh vùng khối u trên ảnh MRI, hỗ trợ bài toán chẩn đoán hình ảnh y tế.

---

## 🎯 Tổng quan bài toán

Cho một ảnh MRI não (grayscale), mô hình dự đoán một **mask nhị phân** đánh dấu vùng nào là khối u (tumor) và vùng nào là mô lành. Đây là bài toán **semantic segmentation** kinh điển trong y tế, đòi hỏi:

- Mô hình phải nhạy với các vùng nhỏ, ranh giới mờ (khối u thường chỉ chiếm một phần nhỏ diện tích ảnh).
- Cân bằng giữa độ chính xác pixel-level và khả năng khái quát hoá trên bệnh nhân mới.

---

## 📊 Dữ liệu

- **Nguồn:** [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Kaggle), ảnh MRI não bệnh nhân u thần kinh đệm bậc thấp (Low-Grade Glioma) kèm mask khối u do bác sĩ gán nhãn.
- Sử dụng dữ liệu của **100 bệnh nhân** đầu tiên (giới hạn do tài nguyên tính toán).
- Ảnh gốc định dạng `.tif`, mỗi lát cắt não có ảnh MRI tương ứng và mask nhị phân.
- Resize toàn bộ về **256×256**, chuẩn hoá pixel về [0,1].
- Chia tập **train/validation theo tỉ lệ 80/20**.

---

## 🏗️ Kiến trúc mô hình

Xây dựng **2 phiên bản U-Net** để so sánh, đi từ baseline đến kiến trúc nâng cao:

### Phiên bản 1 — U-Net cải tiến (baseline)
- Encoder–decoder chuẩn với **BatchNorm + LeakyReLU + Dropout** ở mỗi block để giảm overfitting.
- **Attention Gate** ở decoder giúp mô hình tập trung vào vùng có khối u thay vì học đều trên toàn ảnh.
- Kernel regularization (L2) trên các lớp Conv2D.

### Phiên bản 2 — U-Net + Mini-Xception + Attention (mô hình chính)
- Thay các block Conv2D thường bằng **Mini-Xception block**: `SeparableConv2D` (depthwise + pointwise) kết hợp **residual connection**, giúp mô hình học đặc trưng hiệu quả hơn với ít tham số hơn.
- Giữ nguyên **Attention Gate** ở decoder.
- **Hàm loss kết hợp:** `0.5 × Dice Loss + 0.5 × Binary Cross-Entropy` — Dice Loss giải quyết vấn đề mất cân bằng lớp (vùng khối u chiếm tỉ lệ nhỏ so với nền), BCE ổn định gradient ở giai đoạn đầu huấn luyện.
- Theo dõi thêm metric **Dice Coefficient** bên cạnh accuracy trong lúc huấn luyện.

**Callbacks sử dụng:** `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` để tự động dừng đúng lúc, giảm learning rate khi validation loss chững lại, và luôn lưu lại checkpoint tốt nhất.

---

## 📈 Kết quả huấn luyện

Mô hình U-Net + Mini-Xception + Attention huấn luyện 50 epoch, đạt tại epoch tốt nhất:

| Metric | Train | Validation |
|---|---|---|
| Accuracy | ~99.7% | ~99.6% |
| Dice Coefficient | ~0.82 | ~0.64 |
| Loss (Dice + BCE) | ~0.19 | ~0.27 |

> Lưu ý: Accuracy pixel-level cao là điều dễ hiểu do phần lớn ảnh là nền (background), nên **Dice Coefficient** mới là chỉ số phản ánh sát chất lượng phân đoạn vùng khối u thật sự.

---

## 🔧 Công nghệ sử dụng

| Thành phần | Công cụ |
|---|---|
| Framework Deep Learning | TensorFlow / Keras |
| Xử lý ảnh | OpenCV, SimpleITK |
| Tiền xử lý & trực quan hoá | NumPy, Matplotlib |
| Chia dữ liệu | scikit-learn |
| Nguồn dữ liệu | Kaggle API |

---

## 🔁 Pipeline dự án

1. **Thu thập dữ liệu** — tải bộ dữ liệu LGG MRI Segmentation qua Kaggle API.
2. **Tổ chức dữ liệu** — tách ảnh MRI và mask tương ứng vào cấu trúc thư mục `images/` và `masks/`.
3. **Tiền xử lý** — resize 256×256, chuẩn hoá, nhị phân hoá mask.
4. **Trực quan hoá kiểm tra** — hiển thị overlay ảnh MRI + vùng khối u tô màu để xác thực dữ liệu trước khi huấn luyện.
5. **Xây dựng mô hình** — U-Net với Attention Gate, sau đó nâng cấp bằng Mini-Xception block.
6. **Huấn luyện** — tối ưu bằng Adam, loss kết hợp Dice + BCE, callbacks tự động điều chỉnh learning rate và lưu checkpoint tốt nhất.
7. **Đánh giá & trực quan hoá** — so sánh ảnh gốc / mask thật / mask dự đoán trên tập validation.
8. **Lưu & tải lại mô hình** — export `.h5`, kiểm tra khả năng load lại kèm custom loss/metric.

---

## 📁 Cấu trúc thư mục dự án

```
brain_project/
├── data/
│   ├── images/{train,val}/
│   └── masks/{train,val}/
├── models/
│   └── unet_brain_mri.h5
├── results/
└── src/
```

---

## 🚀 Hướng phát triển tiếp theo

- Tính toán thêm **Dice Score / IoU trên toàn bộ tập validation** (không chỉ theo dõi lúc train) để đánh giá khách quan hơn.
- Áp dụng **data augmentation** (xoay, lật, thay đổi độ sáng) để tăng khả năng khái quát hoá khi dữ liệu bệnh nhân còn hạn chế.
- Thử nghiệm **transfer learning** với backbone pretrained (ResNet, EfficientNet) cho encoder.
- Đóng gói mô hình thành **API suy luận (Flask/FastAPI)** hoặc demo web để bác sĩ/người dùng có thể tải ảnh MRI lên và nhận kết quả phân đoạn trực tiếp.
- Mở rộng dữ liệu huấn luyện lên toàn bộ bệnh nhân trong bộ dữ liệu thay vì giới hạn 100 ca.

---

## 👤 Tác giả

Dự án cá nhân thực hiện với mục tiêu ứng dụng Deep Learning vào bài toán y tế thực tế — từ thu thập dữ liệu, xây dựng kiến trúc mô hình, đến huấn luyện và đánh giá.

*Notebook đầy đủ: `BrainProject.ipynb`*
