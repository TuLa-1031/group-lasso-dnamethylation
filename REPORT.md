# Báo cáo: Xác định CpG Ảnh hưởng đến Gene expression.

## 1. Tổng quan
Mục tiêu của repo là xác định các vị trí Methyl hóa (CpG) có ảnh hưởng mạnh nhất đến mức độ biểu hiện của các Gen tương ứng. Phương pháp được sử dụng là **Logistic Group Lasso**, lựa chọn các feature (CpG) quan trọng trong khi loại bỏ các feature nhiễu (weight về 0).

## 2. Phương pháp (Methodology)

### 2.1. Chuẩn bị Dữ liệu
-   **Map Dữ liệu**: Dữ liệu Methylation (CpG) được ánh xạ với dữ liệu Gene expression (mRNA) qua Ensembl Gene ID.
-   **Xử lý Mục tiêu (Target)**: Dữ liệu biểu hiện Gen (liên tục) được chuyển đổi thành bài toán phân loại nhị phân (**High** / **Low**) dựa trên giá trị trung vị.
-   **Xử lý Đặc trưng (Features)**: Các giá trị Methylation thiếu (NaN) được điền bằng giá trị trung bình và chuẩn hóa.

### 2.2. Mô hình Thuật toán
-   **Mô hình**: Logistic Group Lasso.
-   **Tối ưu hóa**: Sử dụng thuật toán **Proximal Gradient Descent (FISTA)** để giải quyết bài toán tối ưu lồi với thành phần điều chuẩn L1 (Lasso).
-   **Cơ chế**: Thành phần điều chuẩn L1 ép các trọng số của các CpG ít quan trọng về đúng bằng 0, giúp mô hình có khả năng "tự chọn lọc" đặc trưng (Feature Selection).

## 3. Cấu trúc repo

Repo được tổ chức thành các module:

### 3.1. Source code
*   **`main.py`**:
    *   Thực hiện tải toàn bộ dữ liệu vào bộ nhớ, lặp qua từng Gen, huấn luyện mô hình Logistic Group Lasso cho từng Gene.
    *   Kết quả cuối cùng được lưu vào file `all_influential_cpgs.csv`.

*   **`models/logistic_group_lasso.py`**:
    *   Chứa class `LogisticGroupLasso`, cài đặt thuật toán Logistic Regression với điều chuẩn Group Lasso (áp dụng cho từng CpG riêng lẻ như một nhóm).
    *   Sử dụng hàm `_proximal_operator` để soft-thresholding -> sparsity.

*   **`data_loader.py`**:
    *   Chứa hàm `load_real_data` để đọc và xử lý sơ bộ các file dữ liệu lớn (`.tsv`, `.csv`).
    *   Xử lý việc lọc mẫu (samples) và gen (genes) chung giữa các tập dữ liệu.

### 3.2. Script
*   **`prepare_data.py`**:
    *   Script chạy một lần để chuẩn hóa Ensembl ID(loại bỏ hậu tố).
    *   Xác định danh sách các Gen và Mẫu bệnh phẩm (Samples) chung giữa các nguồn dữ liệu (CNV, mRNA, Methylation).

*   **`init.py`**:
    *   Script khởi tạo để tạo file mapping `cg_to_ensembl.csv` từ file manifest gốc, sử dụng thư viện `pyranges` để tìm vị trí overlap.

## 4. Kết quả
File kết quả `all_influential_cpgs.csv` chứa các cột:
*   `Gene_ID`: Mã Ensembl của Gen.
*   `IlmnID`: Mã Probe của CpG.
*   `Weight`: Trọng số ảnh hưởng (+:Tăng biểu hiện, -:Giảm biểu hiện).
*   `AbsWeight`: Giá trị tuyệt đối của Gene expression (Dùng để xếp hạng).