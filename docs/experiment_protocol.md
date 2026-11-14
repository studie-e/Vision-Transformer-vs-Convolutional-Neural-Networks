# Giao thức ViT–ResNet cho tuần 2

Giao thức dưới đây áp dụng cho sáu lượt chính P0. YAML chạy chính nằm trong `configs/week2/`. Hai YAML pilot vẫn dùng một epoch.

| Hạng mục | Điều kiện |
|---|---|
| Dữ liệu | CIFAR-10, 45.000 train và 5.000 validation phân tầng từ tập train gốc. Giữ riêng 10.000 test chính thức. |
| Tỷ lệ | 10%, 50%, 100% tương ứng 4.500, 22.500, 45.000 ảnh. Các tập con lồng nhau và dùng cùng danh sách chỉ số giữa hai mô hình trong từng cặp. |
| Seed | (split, train) = (2026, 42), (2027, 43), (2028, 44). Sáu lượt P0 dùng cặp đầu tiên. |
| Mô hình | ViT 274.282 tham số, ResNet 272.474 tham số. Chênh 0,66% theo mẫu số ResNet. |
| Tiền xử lý | Cùng loader và transform. Crop có padding và lật ngang chỉ trên train. Validation dùng ToTensor và Normalize. |
| Ngân sách | 100 epoch, batch size 128, cosine annealing theo epoch của trainer hiện tại. |
| ViT | AdamW, LR 0.0003, weight decay 0.05, dropout 0.1. |
| ResNet | SGD, LR 0.05, momentum 0.9, weight decay 0.0005. |
| Chọn checkpoint | Top-1 validation cao nhất, hòa điểm giữ epoch sớm hơn. Dùng best.pt để đánh giá. Resume từ last.pt của cùng run. |
| Tập test | Chỉ dùng sau khi khóa lựa chọn mô hình, recipe, seed và checkpoint. |

Ma trận 18 lượt và trạng thái bàn giao nằm ở [run_matrix.csv](../results/hien_week2/run_matrix.csv). Lan Anh bàn giao ViT, Minh bàn giao ResNet. Mỗi lượt cần thư mục có config.yaml, metadata.json, metrics.csv, best.pt và last.pt. Kiểm tra đúng cặp seed, tỷ lệ và manifest, đối chiếu tham số, số bước và accuracy từ checkpoint trước khi đưa vào bảng. Ghi đường dẫn run trong ma trận sau khi đối chiếu.

Giữ recipe của từng mô hình qua cả ba tỷ lệ. Kết quả so sánh phản ánh kiến trúc cùng recipe đã chọn. Ưu tiên sáu lượt P0 trên GPU, sau đó mới chạy P1. Dùng tên run riêng cho mỗi cấu hình và bàn giao toàn bộ thư mục run.

```powershell
python -X utf8 -m src.train --config configs/week2/vit_10.yaml --run-dir runs/week2_vit_10_seed42 --device cuda
```

Thay `vit_10` bằng `vit_50`, `vit_100`, `resnet_10`, `resnet_50` hoặc `resnet_100` ở cả hai đường dẫn để chạy các lượt còn lại.
