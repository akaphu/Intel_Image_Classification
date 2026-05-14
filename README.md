# Phân loại Cảnh trong Ảnh Intel

Repository này triển khai một pipeline phân loại ảnh hoàn chỉnh cho bộ dữ liệu **Intel Image Classification**. Mục tiêu là phân loại ảnh cảnh tự nhiên thành sáu danh mục:

```text
buildings
forest
glacier
mountain
sea
street
```

Dự án so sánh hai hướng tiếp cận chính:

1. **Học máy truyền thống với đặc trưng thủ công**
   - Đặc trưng pixel thô / ảnh được làm phẳng
   - Đặc trưng biểu đồ màu RGB
   - Đặc trưng HOG
   - Đặc trưng kết hợp HOG + biểu đồ RGB
   - Logistic Regression, KNN và SVM

2. **Học sâu cho phân loại ảnh**
   - Simple CNN được huấn luyện từ đầu
   - ResNet18 với transfer learning
   - Fine-tuning ResNet18
   - Phân tích lỗi và trực quan hóa Grad-CAM

Mô hình mạnh nhất trong thí nghiệm hiện tại là **ResNet18 Fine-tuned**, đạt khoảng **93.33% test accuracy** và **93.44% test Macro F1** trên tập kiểm thử.

---

## Mục lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Bộ dữ liệu](#bộ-dữ-liệu)
- [Mục tiêu dự án](#mục-tiêu-dự-án)
- [Phương pháp thực hiện](#phương-pháp-thực-hiện)
- [Các mô hình đã triển khai](#các-mô-hình-đã-triển-khai)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cấu trúc Repository](#cấu-trúc-repository)
- [Thiết lập môi trường](#thiết-lập-môi-trường)
- [Cách chạy](#cách-chạy)
- [Các file đầu ra](#các-file-đầu-ra)
- [Khả năng giải thích mô hình](#khả-năng-giải-thích-mô-hình)
- [Những phát hiện chính](#những-phát-hiện-chính)
- [Hạn chế](#hạn-chế)
- [Hướng phát triển trong tương lai](#hướng-phát-triển-trong-tương-lai)

---

## Tổng quan dự án

Dự án này tập trung vào bài toán **phân loại ảnh cảnh tự nhiên**. Với một ảnh đầu vào, mô hình dự đoán một trong sáu danh mục cảnh: `buildings`, `forest`, `glacier`, `mountain`, `sea`, hoặc `street`.

Dự án được thiết kế như một quy trình học máy đầy đủ:

```text
Khám phá bộ dữ liệu
→ Tiền xử lý dữ liệu
→ Chia tập train/validation/test
→ Trích xuất đặc trưng thủ công
→ Huấn luyện mô hình ML truyền thống
→ Tinh chỉnh siêu tham số
→ Phân tích lỗi
→ Trực quan hóa đặc trưng
→ Huấn luyện CNN
→ Transfer learning với ResNet18
→ Fine-tuning
→ Đánh giá mô hình
→ Trực quan hóa Grad-CAM
→ So sánh mô hình cuối cùng
```

Một mục đích quan trọng của dự án không chỉ là đạt độ chính xác cao, mà còn là hiểu **vì sao các mô hình khác nhau có hiệu năng khác nhau** trong bài toán phân loại ảnh cảnh.

---

## Bộ dữ liệu

Dự án sử dụng bộ dữ liệu **Intel Image Classification**, bao gồm ảnh cảnh thuộc sáu lớp:

| Lớp | Mô tả |
|---|---|
| `buildings` | Cảnh đô thị và kiến trúc |
| `forest` | Cảnh rừng và thảm thực vật |
| `glacier` | Cảnh tuyết, băng và sông băng |
| `mountain` | Cảnh quan núi |
| `sea` | Cảnh đại dương, bờ biển và nước |
| `street` | Cảnh đường phố và đường bộ |

### Chia bộ dữ liệu

Tất cả các mô hình sử dụng cùng một cách chia dữ liệu để đảm bảo so sánh công bằng.

| Tập dữ liệu | Số lượng ảnh |
|---|---:|
| Train | 11,212 |
| Validation | 2,804 |
| Test | 3,000 |

Việc sử dụng cùng một cách chia dữ liệu là quan trọng vì mọi mô hình đều được đánh giá trên cùng một tập kiểm thử. Điều này giúp việc so sánh giữa các mô hình học máy truyền thống và mô hình học sâu trở nên đáng tin cậy hơn.

---

## Mục tiêu dự án

Dự án có bốn mục tiêu chính:

1. **Xây dựng một pipeline phân loại ảnh hoàn chỉnh**
   - Tải và kiểm tra dữ liệu ảnh
   - Làm sạch và tiền xử lý ảnh
   - Tạo các tập train/validation/test
   - Lưu metadata và các đầu ra có thể tái sử dụng

2. **Đánh giá các phương pháp học máy truyền thống**
   - Trích xuất đặc trưng ảnh thủ công
   - Huấn luyện các mô hình Logistic Regression, KNN và SVM
   - So sánh đặc trưng pixel thô, biểu đồ màu, HOG và đặc trưng kết hợp
   - Cải thiện hiệu năng thông qua tinh chỉnh

3. **Đánh giá các phương pháp học sâu**
   - Huấn luyện Simple CNN từ đầu
   - Sử dụng ResNet18 như một mô hình transfer learning
   - Fine-tune ResNet18 để cải thiện hiệu năng phân loại

4. **Phân tích hành vi của mô hình**
   - Sử dụng ma trận nhầm lẫn
   - Kiểm tra metric theo từng lớp
   - Xác định các cặp lớp thường bị nhầm lẫn
   - Trực quan hóa các ảnh bị phân loại sai
   - Sử dụng Grad-CAM để hiểu các vùng ảnh mà mô hình chú ý

---

## Phương pháp thực hiện

### 1. Khám phá dữ liệu

Bộ dữ liệu trước tiên được phân tích để hiểu:

- Số lượng lớp
- Số lượng ảnh trong mỗi lớp
- Ảnh ví dụ từ mỗi danh mục
- Chất lượng ảnh và các vấn đề dữ liệu có thể có
- Phân bố lớp

Bước này đảm bảo dữ liệu phù hợp cho cả thí nghiệm học máy truyền thống và học sâu.

---

### 2. Tiền xử lý dữ liệu

Giai đoạn tiền xử lý bao gồm:

- Kiểm tra tính hợp lệ của file ảnh
- Xóa hoặc loại trừ ảnh có vấn đề khi cần thiết
- Tạo ánh xạ lớp nhất quán
- Chia bộ dữ liệu thành các tập train, validation và test
- Lưu metadata cho các thí nghiệm sau
- Tính toán thống kê chuẩn hóa khi cần

Các file metadata quan trọng bao gồm:

```text
data/metadata/class_to_idx.json
data/metadata/idx_to_class.json
data/metadata/preprocessing_summary.json
```

Các file này giúp pipeline có thể tái lập và ngăn lỗi không khớp chỉ số lớp giữa các notebook.

---

### 3. Trích xuất đặc trưng thủ công

Các mô hình học máy truyền thống không thể trực tiếp hiểu cấu trúc không gian và ngữ nghĩa của ảnh giống như CNN. Vì vậy, trích xuất đặc trưng thủ công được sử dụng trước khi huấn luyện các mô hình truyền thống.

Dự án sử dụng các đặc trưng sau:

#### Đặc trưng pixel được làm phẳng

Ảnh được thay đổi kích thước và làm phẳng thành một vector một chiều.

**Ưu điểm:**

- Dễ triển khai
- Giữ lại thông tin pixel thô

**Hạn chế:**

- Số chiều rất cao
- Nhạy cảm với thay đổi vị trí, ánh sáng và nền
- Không biểu diễn rõ ràng hình dạng hoặc kết cấu

---

#### Biểu đồ màu RGB

Đặc trưng biểu đồ RGB biểu diễn phân bố màu của ảnh.

**Ưu điểm:**

- Hữu ích cho các lớp có mẫu màu đặc trưng
- Hỗ trợ phân biệt các cảnh như forest, sea, glacier và mountain
- Số chiều thấp hơn so với pixel thô

**Hạn chế:**

- Không giữ lại vị trí đối tượng
- Không thể biểu diễn đầy đủ hình dạng hoặc cấu trúc cảnh
- Có thể nhầm lẫn các lớp có màu sắc tương tự

---

#### Đặc trưng HOG

HOG, hay Histogram of Oriented Gradients, ghi nhận thông tin cạnh và gradient từ ảnh.

**Ưu điểm:**

- Tốt trong việc biểu diễn hình dạng, đường viền và kết cấu
- Bền vững hơn so với đặc trưng pixel thô
- Hữu ích cho các cấu trúc cảnh như buildings, streets và mountains

**Hạn chế:**

- Mất thông tin màu sắc
- Không biểu diễn ý nghĩa ngữ nghĩa cấp cao
- Vẫn yếu hơn các đặc trưng sâu được CNN học tự động

---

#### Đặc trưng kết hợp HOG + Biểu đồ RGB

Thí nghiệm SVM sử dụng biểu diễn kết hợp:

```text
Vector đặc trưng cuối cùng = Đặc trưng HOG + Đặc trưng biểu đồ RGB
```

Sự kết hợp này được sử dụng vì HOG và biểu đồ RGB ghi nhận các loại thông tin bổ sung cho nhau:

| Loại đặc trưng | Ghi nhận |
|---|---|
| HOG | Hình dạng, cạnh, gradient, kết cấu |
| Biểu đồ RGB | Phân bố màu sắc |

Việc sử dụng **SVM + HOG + Biểu đồ RGB** mạnh hơn so với chỉ dùng HOG trong nhiều bài toán phân loại ảnh vì các danh mục cảnh thường phụ thuộc vào cả cấu trúc và màu sắc. Ví dụ:

- `forest` thường có phân bố màu xanh lá mạnh
- `sea` thường chứa các vùng chiếm ưu thế màu xanh dương
- `glacier` thường chứa các tông trắng, xám và xanh dương
- `buildings` và `street` thường chứa nhiều cạnh mạnh và cấu trúc hình học

Điều này làm cho biểu diễn đặc trưng kết hợp cung cấp nhiều thông tin hơn cho phân loại dựa trên SVM.

---

### 4. Huấn luyện học máy truyền thống

Dự án đánh giá một số mô hình học máy truyền thống:

| Mô hình | Đầu vào đặc trưng | Mục đích |
|---|---|---|
| Logistic Regression | Đặc trưng thủ công | Baseline tuyến tính mạnh |
| KNN | Đặc trưng thủ công | Baseline dựa trên khoảng cách |
| SVM | HOG + Biểu đồ RGB | Bộ phân loại dựa trên margin sử dụng đặc trưng trực quan kết hợp |

Các mô hình truyền thống hữu ích vì chúng cung cấp các baseline dễ diễn giải và giúp thể hiện vai trò của feature engineering trong phân loại ảnh.

---

### 5. Huấn luyện học sâu

Phần học sâu của dự án sử dụng hai kiến trúc chính:

1. **Simple CNN**
2. **ResNet18**

Khác với các mô hình học máy truyền thống, các mô hình dựa trên CNN có thể học đặc trưng trực quan trực tiếp từ dữ liệu ảnh.

---

## Các mô hình đã triển khai

### Logistic Regression

Logistic Regression được sử dụng như một baseline truyền thống mạnh. Mặc dù là mô hình tuyến tính, nó có thể hoạt động tốt khi kết hợp với các đặc trưng thủ công giàu thông tin như HOG và biểu đồ màu.

Vai trò của nó trong dự án là cung cấp một mốc so sánh đơn giản, ổn định và dễ diễn giải.

---

### K-Nearest Neighbors

KNN được sử dụng như một baseline dựa trên khoảng cách. Nó phân loại một ảnh dựa trên nhãn của các vector đặc trưng gần nhất.

KNN hữu ích để so sánh, nhưng có thể kém hiệu quả hơn trên các đặc trưng ảnh có số chiều cao vì các thước đo khoảng cách trở nên kém đáng tin cậy hơn khi số chiều tăng.

---

### SVM + HOG + Biểu đồ RGB

SVM được huấn luyện bằng biểu diễn đặc trưng kết hợp:

```text
Đặc trưng HOG + đặc trưng biểu đồ RGB
```

Mô hình này được thiết kế để sử dụng cả thông tin dựa trên cấu trúc và dựa trên màu sắc.

Động lực rất rõ ràng:

- HOG ghi nhận hướng cạnh, gradient và hình dạng cục bộ.
- Biểu đồ RGB ghi nhận phân bố màu tổng thể.
- SVM học một biên quyết định với margin tối đa giữa các lớp.

Điều này làm cho **SVM + HOG + Biểu đồ RGB** trở thành một hướng tiếp cận học máy truyền thống mạnh cho phân loại cảnh.

---

### Simple CNN

Simple CNN được sử dụng làm mô hình học sâu baseline.

Nó học đặc trưng trực quan trực tiếp từ ảnh thông qua các lớp tích chập. So với các mô hình truyền thống, nó không yêu cầu trích xuất đặc trưng thủ công như HOG hoặc biểu đồ màu.

Vai trò của nó là trả lời câu hỏi:

```text
Một CNN được huấn luyện từ đầu có thể hoạt động tốt đến mức nào trên bộ dữ liệu này?
```

---

### ResNet18 Feature Extraction

ResNet18 trước tiên được sử dụng như một bộ trích xuất đặc trưng. Ở giai đoạn này, hầu hết các lớp pretrained được đóng băng, và chỉ lớp phân loại cuối cùng được huấn luyện cho sáu lớp cảnh.

Cách tiếp cận này hữu ích vì các đặc trưng pretrained từ ImageNet đã chứa các biểu diễn trực quan mạnh như cạnh, kết cấu, bộ phận đối tượng và mẫu cảnh.

---

### ResNet18 Fine-tuned

Ở giai đoạn fine-tuning, một phần mạng ResNet18 được mở khóa và huấn luyện tiếp trên bộ dữ liệu phân loại cảnh.

Điều này cho phép mô hình thích nghi các đặc trưng pretrained với bộ dữ liệu mục tiêu hiệu quả hơn.

Trong thí nghiệm hiện tại, **ResNet18 Fine-tuned** đạt hiệu năng tổng thể tốt nhất.

---

## Kết quả thực nghiệm

### Kết quả học sâu

| Mô hình | Test Accuracy | Test Macro F1 | Vai trò |
|---|---:|---:|---|
| Simple CNN | 0.8783 | 0.8793 | Baseline học sâu |
| ResNet18 Fine-tuned | 0.9333 | 0.9344 | Mô hình tốt nhất |

### Nhận xét chính

ResNet18 Fine-tuned vượt trội đáng kể so với Simple CNN.

So với Simple CNN:

- Macro F1 cải thiện khoảng **5.5 điểm phần trăm**
- Số dự đoán sai giảm từ **365** xuống **200** trên tập kiểm thử 3,000 ảnh
- Mô hình trở nên ổn định hơn giữa các lớp có hình ảnh tương tự

Kết quả này cho thấy lợi thế của transfer learning và fine-tuning trong các tác vụ phân loại ảnh.

---

## Cấu trúc Repository

Một cấu trúc GitHub được đề xuất cho dự án này được trình bày dưới đây:

```text
intel_image_project/
│
├── data/
│   ├── raw/
│   │   ├── seg_train/
│   │   ├── seg_test/
│   │   └── seg_pred/
│   │
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── metadata/
│       ├── class_to_idx.json
│       ├── idx_to_class.json
│       └── preprocessing_summary.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_traditional_ml_baselines.ipynb
│   ├── 04_traditional_ml_tuning.ipynb
│   ├── 05_feature_visualization.ipynb
│   ├── 06_deep_learning_train.ipynb
│   ├── 07_logistic_regression_error_analysis.ipynb
│   ├── 08_svm_hog_rgb_error_analysis.ipynb
│   └── 09_deep_learning_evaluation_gradcam.ipynb
│
├── models/
│   ├── traditional_ml/
│   │   ├── best_baseline_model.joblib
│   │   ├── best_tuned_model.joblib
│   │   └── svm_hog_rgb_model.joblib
│   │
│   └── deep_learning/
│       ├── simple_cnn_best.pth
│       ├── resnet18_feature_extractor_best.pth
│       └── resnet18_finetuned_best.pth
│
├── outputs/
│   ├── features/
│   │   └── traditional_ml/
│   │       ├── train_hog_features.npz
│   │       ├── val_hog_features.npz
│   │       ├── test_hog_features.npz
│   │       ├── train_rgb_hist_features.npz
│   │       ├── val_rgb_hist_features.npz
│   │       ├── test_rgb_hist_features.npz
│   │       ├── train_hog_rgb_features.npz
│   │       ├── val_hog_rgb_features.npz
│   │       └── test_hog_rgb_features.npz
│   │
│   ├── predictions/
│   │   ├── logistic_regression_test_predictions.csv
│   │   ├── svm_hog_rgb_test_predictions.csv
│   │   ├── simple_cnn_test_predictions.csv
│   │   ├── resnet18_feature_extractor_test_predictions.csv
│   │   └── resnet18_finetuned_test_predictions.csv
│   │
│   ├── metrics/
│   │   ├── traditional_ml_baselines_metrics.csv
│   │   ├── traditional_ml_tuning_results.csv
│   │   ├── logistic_regression_test_metrics.json
│   │   ├── svm_hog_rgb_test_metrics.json
│   │   ├── simple_cnn_test_metrics.json
│   │   ├── resnet18_finetuned_test_metrics.json
│   │   ├── cnn_model_comparison.csv
│   │   └── final_model_comparison.csv
│   │
│   └── figures/
│       ├── dataset/
│       ├── features/
│       ├── traditional_ml/
│       ├── deep_learning/
│       ├── gradcam/
│       └── final/
│
├── reports/
│   └── final_report/
│
├── requirements.txt
└── README.md
```

---

## Thiết lập môi trường

### 1. Clone Repository

```bash
git clone https://github.com/your-username/intel-image-scene-classification.git
cd intel-image-scene-classification
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường:

```bash
# Windows
venv\Scripts\activate
```

```bash
# Linux / macOS
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Các dependencies được đề xuất:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scikit-image
opencv-python
Pillow
tqdm
joblib
torch
torchvision
```

---

## Cách chạy

Các notebook nên được thực thi theo thứ tự.

### Bước 1: Khám phá bộ dữ liệu

```text
notebooks/01_eda.ipynb
```

Notebook này kiểm tra bộ dữ liệu, phân bố lớp và các ảnh mẫu.

---

### Bước 2: Tiền xử lý dữ liệu

```text
notebooks/02_preprocessing.ipynb
```

Notebook này tạo:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
data/metadata/class_to_idx.json
data/metadata/idx_to_class.json
```

---

### Bước 3: Huấn luyện các baseline ML truyền thống

```text
notebooks/03_traditional_ml_baselines.ipynb
```

Notebook này trích xuất đặc trưng thủ công và huấn luyện các mô hình baseline:

- Logistic Regression
- KNN
- SVM

Các loại đặc trưng bao gồm:

- Đặc trưng pixel được làm phẳng
- Đặc trưng biểu đồ RGB
- Đặc trưng HOG
- Đặc trưng HOG + biểu đồ RGB

---

### Bước 4: Tinh chỉnh các mô hình ML truyền thống

```text
notebooks/04_traditional_ml_tuning.ipynb
```

Notebook này tinh chỉnh siêu tham số và lưu mô hình học máy truyền thống tốt nhất.

---

### Bước 5: Trực quan hóa đặc trưng thủ công

```text
notebooks/05_feature_visualization.ipynb
```

Notebook này trực quan hóa:

- Ảnh gốc
- Ảnh grayscale
- Các kênh RGB
- Bản đồ cạnh Sobel
- Trực quan hóa HOG
- Biểu đồ màu RGB

Bước này quan trọng vì nó giải thích sự khác biệt giữa các đặc trưng được thiết kế thủ công và các đặc trưng được CNN học tự động.

---

### Bước 6: Huấn luyện các mô hình học sâu

```text
notebooks/06_deep_learning_train.ipynb
```

Notebook này huấn luyện:

- Simple CNN
- ResNet18 Feature Extractor
- ResNet18 Fine-tuned

Nó lưu checkpoints mô hình, lịch sử huấn luyện, metrics và các file dự đoán.

---

### Bước 7: Phân tích lỗi Logistic Regression

```text
notebooks/07_logistic_regression_error_analysis.ipynb
```

Notebook này thực hiện:

- Tính toán metric tổng thể
- Phân tích metric theo từng lớp
- Trực quan hóa ma trận nhầm lẫn
- Phân tích các cặp lớp nhầm lẫn nhiều nhất
- Kiểm tra ảnh bị phân loại sai

---

### Bước 8: Phân tích lỗi SVM + HOG + RGB

```text
notebooks/08_svm_hog_rgb_error_analysis.ipynb
```

Notebook này tập trung vào mô hình SVM sử dụng đặc trưng thủ công kết hợp:

```text
HOG + RGB histogram
```

Nó tạo ra:

- Dự đoán kiểm thử của SVM
- Metrics tổng thể của SVM
- Metrics theo từng lớp của SVM
- Ma trận nhầm lẫn của SVM
- Các cặp lớp bị nhầm lẫn nhiều nhất của SVM
- Ví dụ bị phân loại sai của SVM
- So sánh với Logistic Regression

---

### Bước 9: Đánh giá các mô hình học sâu và tạo Grad-CAM

```text
notebooks/09_deep_learning_evaluation_gradcam.ipynb
```

Notebook này đánh giá các mô hình dựa trên CNN và tạo:

- Đường cong huấn luyện
- Ma trận nhầm lẫn
- So sánh F1 theo từng lớp
- Các cặp lớp bị nhầm lẫn nhiều nhất
- Ví dụ bị phân loại sai
- Trực quan hóa Grad-CAM
- So sánh mô hình cuối cùng

---

## Các file đầu ra

### Các file mô hình quan trọng

| File | Mô tả |
|---|---|
| `models/traditional_ml/best_tuned_model.joblib` | Mô hình ML truyền thống được tinh chỉnh tốt nhất |
| `models/traditional_ml/svm_hog_rgb_model.joblib` | Mô hình SVM được huấn luyện với đặc trưng HOG + biểu đồ RGB |
| `models/deep_learning/simple_cnn_best.pth` | Checkpoint Simple CNN tốt nhất |
| `models/deep_learning/resnet18_feature_extractor_best.pth` | Checkpoint ResNet18 feature extraction tốt nhất |
| `models/deep_learning/resnet18_finetuned_best.pth` | Checkpoint ResNet18 fine-tuned tốt nhất |

---

### Các file metrics quan trọng

| File | Mô tả |
|---|---|
| `outputs/metrics/traditional_ml_baselines_metrics.csv` | So sánh baseline cho các mô hình ML truyền thống |
| `outputs/metrics/traditional_ml_tuning_results.csv` | Kết quả tinh chỉnh siêu tham số |
| `outputs/metrics/logistic_regression_test_metrics.json` | Metrics kiểm thử cuối cùng của Logistic Regression |
| `outputs/metrics/svm_hog_rgb_test_metrics.json` | Metrics kiểm thử cuối cùng của SVM + HOG + RGB |
| `outputs/metrics/simple_cnn_test_metrics.json` | Metrics kiểm thử của Simple CNN |
| `outputs/metrics/resnet18_finetuned_test_metrics.json` | Metrics kiểm thử của ResNet18 Fine-tuned |
| `outputs/metrics/cnn_model_comparison.csv` | So sánh các mô hình dựa trên CNN |
| `outputs/metrics/final_model_comparison.csv` | So sánh cuối cùng giữa tất cả các mô hình được chọn |

---

### Các file hình ảnh quan trọng

| File | Mô tả |
|---|---|
| `outputs/figures/features/original_samples_grid.png` | Ảnh mẫu từ bộ dữ liệu |
| `outputs/figures/features/handcrafted_feature_comparison_panel.png` | So sánh ảnh gốc, grayscale, cạnh và đặc trưng HOG |
| `outputs/figures/features/color_histogram_examples.png` | Ví dụ biểu đồ RGB |
| `outputs/figures/traditional_ml/confusion_matrix_logistic_regression.png` | Ma trận nhầm lẫn của Logistic Regression |
| `outputs/figures/traditional_ml/confusion_matrix_svm_hog_rgb.png` | Ma trận nhầm lẫn của SVM + HOG + RGB |
| `outputs/figures/deep_learning/training_curves_resnet18_finetuned.png` | Đường cong fine-tuning ResNet18 |
| `outputs/figures/deep_learning/resnet18_finetuned_confusion_matrix.png` | Ma trận nhầm lẫn của ResNet18 Fine-tuned |
| `outputs/figures/deep_learning/per_class_f1_score_comparison.png` | So sánh F1 theo từng lớp |
| `outputs/figures/gradcam/resnet18_finetuned_gradcam_correct_examples.png` | Grad-CAM cho các dự đoán đúng |
| `outputs/figures/gradcam/resnet18_finetuned_gradcam_misclassified_examples.png` | Grad-CAM cho các dự đoán sai |

---

## Khả năng giải thích mô hình

Dự án sử dụng nhiều kỹ thuật để diễn giải hành vi của mô hình.

### Ma trận nhầm lẫn

Ma trận nhầm lẫn được sử dụng để xác định những lớp nào thường bị nhầm lẫn.

Điều này đặc biệt hữu ích cho phân loại cảnh vì một số lớp có chung các mẫu trực quan tương tự. Ví dụ:

- `mountain` và `glacier` có thể cùng có tuyết, đá và kết cấu cảnh quan.
- `buildings` và `street` có thể cùng có cấu trúc đô thị.
- `sea` và `glacier` có thể cùng có các vùng màu xanh dương hoặc trắng.

---

### Các cặp lớp bị nhầm lẫn nhiều nhất

Phân tích các cặp lớp bị nhầm lẫn nhiều nhất xác định những lỗi thường gặp nhất từ nhãn thật sang nhãn dự đoán.

Điều này giúp trả lời các câu hỏi như:

```text
Lớp nào có khả năng bị mô hình nhầm với lớp khác nhất?
Vì sao sự nhầm lẫn này xảy ra về mặt trực quan?
Lỗi có phải do màu sắc, kết cấu, bố cục đối tượng hay sự tương đồng nền không?
```

---

### Ví dụ bị phân loại sai

Các ví dụ bị phân loại sai được sử dụng để kiểm tra các ảnh thực tế mà mô hình dự đoán không đúng.

Điều này giúp giải thích liệu lỗi đó có hợp lý hay không. Trong nhiều trường hợp, dự đoán sai xảy ra vì:

- Ảnh chứa các yếu tố cảnh trộn lẫn
- Đối tượng chính không nổi bật về mặt thị giác
- Ánh sáng hoặc độ tương phản bất thường
- Hai lớp có kết cấu hoặc màu sắc tương tự
- Bản thân ảnh mơ hồ ngay cả đối với con người

---

### Grad-CAM

Grad-CAM được sử dụng cho các mô hình dựa trên CNN để làm nổi bật các vùng ảnh đóng góp nhiều nhất cho dự đoán.

Nó giúp trả lời:

```text
Mô hình có đang nhìn vào đúng vùng không?
Mô hình đang tập trung vào nền cảnh hay các đối tượng không liên quan?
Vì sao mô hình phân loại ảnh này đúng hoặc sai?
```

Grad-CAM làm cho phần học sâu của dự án dễ diễn giải hơn và cung cấp bằng chứng trực quan mạnh hơn cho phân tích mô hình.

---

## Những phát hiện chính

1. **ML truyền thống phụ thuộc nhiều vào feature engineering**

   Các mô hình truyền thống không thể tự động học biểu diễn ảnh. Hiệu năng của chúng phụ thuộc mạnh vào chất lượng đặc trưng.

2. **HOG và biểu đồ RGB bổ sung cho nhau**

   HOG ghi nhận thông tin hình dạng và gradient, trong khi biểu đồ RGB ghi nhận phân bố màu sắc. Kết hợp chúng giúp SVM có biểu diễn phong phú hơn.

3. **Simple CNN hoạt động tốt hơn nhiều baseline dùng đặc trưng thủ công**

   CNN có thể học đặc trưng không gian trực tiếp từ pixel ảnh, điều này làm cho chúng phù hợp hơn với phân loại ảnh.

4. **ResNet18 Fine-tuned là mô hình tốt nhất**

   Fine-tuning một ResNet18 pretrained cho phép mô hình thích nghi các đặc trưng ImageNet mạnh với bộ dữ liệu phân loại cảnh mục tiêu.

5. **Phân tích lỗi là cần thiết**

   Chỉ độ chính xác là chưa đủ. Ma trận nhầm lẫn, metrics theo từng lớp, ví dụ bị phân loại sai và Grad-CAM cung cấp hiểu biết sâu hơn về hành vi mô hình.

---

## Hạn chế

Mặc dù dự án đã chứa một pipeline phân loại và phân tích hoàn chỉnh, vẫn còn một số hạn chế:

- Các mô hình ML truyền thống vẫn phụ thuộc vào đặc trưng được thiết kế thủ công.
- HOG + biểu đồ RGB không thể biểu diễn đầy đủ thông tin ngữ nghĩa cấp cao.
- Simple CNN có thể cần nhiều dữ liệu hơn hoặc regularization mạnh hơn để cạnh tranh với các mô hình pretrained.
- Một số danh mục cảnh bị mơ hồ về mặt trực quan, đặc biệt khi ảnh chứa các đối tượng trộn lẫn hoặc nền tương tự.
- Phân tích Grad-CAM hiện tại chủ yếu tập trung vào các mô hình dựa trên CNN.

---

## Hướng phát triển trong tương lai

Các cải tiến có thể bao gồm:

1. **Thử nghiệm với các kiến trúc CNN mạnh hơn**
   - ResNet50
   - EfficientNet
   - MobileNetV3
   - ConvNeXt

2. **Cải thiện data augmentation**
   - Random crop
   - Color jitter
   - Random perspective
   - CutMix hoặc MixUp

3. **Thêm triển khai mô hình**
   - Web demo
   - Giao diện Streamlit hoặc Gradio
   - REST API cho dự đoán ảnh

4. **Cải thiện khả năng giải thích**
   - Grad-CAM++
   - Score-CAM
   - Integrated Gradients

5. **Cải thiện phân tích lỗi**
   - Phân tích lỗi theo độ sáng ảnh
   - Phân tích lỗi theo màu chủ đạo
   - Phân tích lỗi theo khoảng tin cậy
   - Nhóm các trường hợp lỗi tương tự về mặt trực quan

---

## Kết luận

Dự án này xây dựng một pipeline phân loại cảnh ảnh hoàn chỉnh và có thể tái lập trên bộ dữ liệu Intel Image Classification.

Dự án so sánh các mô hình học máy truyền thống sử dụng đặc trưng thủ công với các mô hình học sâu học biểu diễn trực quan trực tiếp từ ảnh. Các thí nghiệm học máy truyền thống cho thấy tầm quan trọng của feature engineering, đặc biệt thông qua HOG, biểu đồ RGB và sự kết hợp của chúng trong mô hình SVM. Các thí nghiệm học sâu cho thấy các mô hình dựa trên CNN mạnh hơn cho các tác vụ nhận dạng trực quan, với ResNet18 Fine-tuned đạt hiệu năng tốt nhất.

Ngoài độ chính xác, dự án còn bao gồm quy trình phân tích chi tiết sử dụng metrics theo từng lớp, ma trận nhầm lẫn, các cặp lớp bị nhầm lẫn nhiều nhất, ví dụ bị phân loại sai và trực quan hóa Grad-CAM. Các thành phần này làm cho dự án hoàn chỉnh hơn, dễ giải thích hơn và phù hợp để trình bày trong GitHub portfolio.
