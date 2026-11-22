# Milestone tuần 3 — phần Minh

**Người phụ trách: Minh.**

## Tiến độ và kết quả validation

Hoàn tất sáu lượt ResNet bổ sung và giữ nguyên ba lượt seed đầu, tổng 9 lượt ResNet. Dữ liệu mỗi split được chia phân tầng và dùng chung theo từng cặp seed của hai mô hình. ResNet train từ đầu trên RTX 5080, với batch 128, SGD lr 0,05, momentum 0,9, weight decay 0,0005 và cosine annealing trong 100 epoch. Đánh giá lại `best.pt` trên CUDA khớp số dự đoán đúng trong cả sáu lượt mới.

| Tỷ lệ | Seed 42 | Seed 43 | Seed 44 | Trung bình ± SD (điểm %) | Bước mỗi lượt |
|---|---:|---:|---:|---:|---:|
| 10% | 74,24% | 74,14% | 74,64% | 74,34 ± 0,26 | 3.600 |
| 50% | 88,40% | 88,28% | 88,28% | 88,32 ± 0,07 | 17.600 |
| 100% | 91,80% | 91,82% | 91,82% | 91,81 ± 0,01 | 35.200 |

Cả ba mức dữ liệu có ba seed ResNet. Mức 100% đạt 91,81% trung bình trên validation, cao hơn 17,47 điểm phần trăm so với mức 10%. Mức 10% có độ lệch chuẩn mẫu lớn nhất. Do mỗi cặp seed thay cả cách chia ảnh và khởi tạo, độ lệch chuẩn mô tả biến thiên của toàn bộ giao thức. Dùng cùng số epoch dẫn đến ngân sách cập nhật khác nhau giữa các tỷ lệ. Thời gian đo được cho sáu lượt mới là 64,4 phút trong vòng train, hoặc 70,7 phút khi cộng validation.

## Phạm vi so sánh và phần tiếp theo

Ở cặp `(2026, 42)`, ViT tuần 2 đạt 52,16%, 67,66% và 75,90% ở ba tỷ lệ. Hiện có 3 lượt ViT đã xác minh và 9 lượt ResNet. Cần bổ sung các lượt ViT ở `(2027, 43)` và `(2028, 44)` để tính độ lệch chuẩn ViT và đối chiếu chênh lệch trên từng cặp dữ liệu. Các mô hình dùng optimizer khác nhau theo recipe đã chốt, vì vậy kết quả mô tả kiến trúc cùng thiết lập huấn luyện. Tập test dành cho đánh giá cuối.

Đã cấu hình Swin dạng thăm dò với 261.824 tham số và chạy thử trên CUDA một epoch ở mức 10%, đạt 1.097/5.000 ảnh validation đúng. Hai mức 10% và 100% đã có YAML dùng AdamW lr 0,0003, weight decay 0,05 và cosine 100 epoch. Các lượt Swin đầy đủ xếp sau khi đủ bảng chính. Cấu hình Swin dùng mã lớp `SwinTransformer` từ torchvision. [Bài báo Swin](https://arxiv.org/abs/2103.14030) · [Mã torchvision 0.26](https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/swin_transformer.html).

[Bằng chứng kết quả, thời gian và hướng dẫn chuyển gói](../results/week3/README.md).
