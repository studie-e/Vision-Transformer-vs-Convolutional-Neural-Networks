# Kết quả ResNet tuần 3

**Người phụ trách: Minh.**

Đã hoàn tất sáu lượt ResNet mới trên RTX 5080 với `(split_seed, seed)` là `(2027, 43)` và `(2028, 44)`. Giữ giao thức tuần 2 gồm 100 epoch, batch 128, SGD lr 0,05, momentum 0,9, weight decay 0,0005, cosine annealing và 272.474 tham số. Mỗi lượt chọn checkpoint trên 5.000 ảnh validation, hòa giữ epoch sớm hơn. Tập test dành cho đánh giá cuối.

| Tỷ lệ | Seed 42 | Seed 43 | Seed 44 | Trung bình ± SD (điểm %) | Bước mỗi lượt |
|---|---:|---:|---:|---:|---:|
| 10% | 74,24% | 74,14% | 74,64% | 74,34 ± 0,26 | 3.600 |
| 50% | 88,40% | 88,28% | 88,28% | 88,32 ± 0,07 | 17.600 |
| 100% | 91,80% | 91,82% | 91,82% | 91,81 ± 0,01 | 35.200 |

Sáu checkpoint mới cho đúng số ảnh dự đoán đúng như log khi đánh giá lại trên CUDA. TensorBoard đủ epoch 1–100 và các checkpoint khớp YAML, manifest cùng số bước cập nhật. Độ lệch chuẩn mẫu dùng `ddof=1`, phản ánh biến thiên cả do chia dữ liệu và khởi tạo. So sánh ở cùng số epoch nên lượng ảnh và số cập nhật cùng tăng.

| Run | Best epoch | Đúng / 5.000 | Giây train |
|---|---:|---:|---:|
| `resnet_10_seed43` | 79 | 3707 | 121,4 |
| `resnet_50_seed43` | 84 | 4414 | 499,9 |
| `resnet_100_seed43` | 93 | 4591 | 1128,6 |
| `resnet_10_seed44` | 87 | 3732 | 130,6 |
| `resnet_50_seed44` | 94 | 4414 | 663,0 |
| `resnet_100_seed44` | 94 | 4591 | 1318,4 |

Tổng thời gian vòng train của sáu lượt là **64,4 phút**, còn train cộng validation là **70,7 phút** theo log. Thời gian này được đo trước bước ghi checkpoint và log. Tỷ lệ 10% đạt độ biến thiên lớn nhất trong ba tỷ lệ ResNet đã đo. Mean tăng 13,98 điểm phần trăm khi đi từ 10% lên 50%, rồi tăng thêm 3,49 điểm khi dùng toàn bộ tập train.

Swin 261.824 tham số đã chạy thử một epoch CUDA ở 10% trên split 2026 và seed 42. Validator độc lập đếm **1.097/5.000** ảnh đúng, khớp log. Vòng train mất **13,6 giây** và có 36 cập nhật. Hai YAML Swin 100 epoch ở 10% và 100% đã sẵn sàng. Hàng đợi chính có 12/18 lượt đã xác minh, còn sáu lượt ViT của hai cặp seed mới trước loạt Swin đầy đủ.

- [Bảng từng lượt](resnet_runs.csv) · [Thống kê ba seed](resnet_summary.csv) · [Manifest ZIP và SHA-256](resnet_handoff_manifest.json).
- [Swin chạy thử trên CUDA](swin_smoke_runs.csv) · [Log một epoch](swin_smoke_metrics.csv).
- [Hướng dẫn thực thi](../../docs/minh-week3.md) · [Phần Minh viết cho milestone](../../reports/milestone_minh_week3.md).

Gói tổng sáu lượt ở `runs/handoff/week3/resnet_week3_all_runs.zip`, khoảng 27.9 MB, kèm checksum cạnh file. Mỗi ZIP chứa config, metadata, metrics, best.pt, last.pt, TensorBoard và dự đoán validation. Ma trận đánh dấu `ready_for_handoff` cho sáu lượt ResNet mới.
