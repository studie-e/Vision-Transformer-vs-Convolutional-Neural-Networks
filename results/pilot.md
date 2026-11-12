# Pilot tuần 1 của Minh — 12/11/2026

## Môi trường và phạm vi

- Windows, Python 3.13.7, PyTorch 2.11.0+cu128, CUDA runtime 12.8, RTX 5080, driver 617.42.
- ResNet CIFAR-style, width 16, 3 block/stage, projection shortcut; **272,474 tham số trainable**.
- Mã nguồn khi chạy: SHA-256 `683577fd1856792d93cec53166a3cce9a67a70296dcdd5cea1d0745eba2ea9f0`; HEAD `f9a17bef7ee8d4c244f8773287c0a59665d00f02` còn thay đổi chưa commit. Metadata đầy đủ nằm trong từng thư mục run.
- Lan Anh/Hiền chưa giao ViT/split chính. Các phép chạy dưới đây kiểm tra kỹ thuật của Minh, không phải bảng so sánh chính.

## Dữ liệu pilot

Dùng CIFAR-10 dạng PNG từ [kho fastai](https://docs.fast.ai/data.external.html), tải từ URL trong `pilot_manifest.json`. Nguồn Toronto ban đầu tải chậm nên đã dừng, không dùng file tải dở. Archive fastai có 135.107.811 byte, MD5 quan sát và cố định khi chạy lại: `7d7b892d003476893d94d030e4f7c0f0`.

Chỉ đọc 50.000 ảnh trong thư mục **train**, lấy ngẫu nhiên 4.500 train và 1.000 validation không giao nhau, split seed 2026, run seed 17. Chỉ số chính xác lưu ở [pilot_manifest.json](pilot_manifest.json). Chuẩn hóa (x−0,5)/0,5, không augmentation, batch 128, num_workers=0. **Chưa phải split phân tầng 45k/5k của Hiền**, không dùng test chính thức.

## Pilot train và resume

Config: `configs/train.yaml`; SGD lr 0,05, momentum 0,9, weight decay 0,0005; cosine schedule 3 epoch; FP32, deterministic algorithms. Run `runs/minh_week1_pilot`: chạy epoch 1, thoát tiến trình rồi gọi CLI resume để chạy epoch 2–3.

| Epoch | Train accuracy | Validation accuracy | Train loss | Val loss | Thời gian train (giây) |
|---|---:|---:|---:|---:|---:|
| 1 | 22.04% | 28.10% | 2.0647 | 1.9885 | 12.794 |
| 2 | 33.02% | 30.50% | 1.7443 | 1.9119 | 8.041 |
| 3 | 40.71% | 39.80% | 1.5715 | 1.5639 | 2.311 |

Kết quả cuối: train **40,71%**, validation **39,8%**, 108 bước cập nhật; checkpoint tốt nhất ở epoch 3. CSV gốc: [pilot_resnet_metrics.csv](pilot_resnet_metrics.csv). TensorBoard tại `runs/minh_week1_pilot/tensorboard/`; `best.pt` và `last.pt` ở cùng thư mục.

Thời gian epoch đầu và epoch đầu sau khi mở lại tiến trình cao hơn do khởi động/caching. Lượt chạy liền dùng để đối chiếu có thời gian train epoch 2–3 trung bình khoảng **2.00 giây/epoch** với đúng 4.500 ảnh pilot. Bộ nhớ CUDA cấp phát đỉnh khoảng **276,2 MiB**; đây không phải toàn bộ VRAM của tiến trình. Các phép đo gồm đọc batch/chuyển thiết bị/tính toán trong vòng train; chưa gồm tải/extract dữ liệu, khởi tạo tiến trình, ghi checkpoint và TensorBoard. Không ngoại suy thành thời gian chắc chắn cho 45.000 ảnh hoặc ViT.

## Kiểm tra học một batch thật

Run `runs/minh_week1_overfit`, config `configs/overfit.yaml`: 32 ảnh train cố định, 32 validation khác; AdamW lr 0,003, cosine 60 epoch, không augmentation.

- Train loss: **2.3834 → 0.002245**.
- Train accuracy cuối: **100%**; validation cuối **15,625%** trên 32 ảnh khác.
- Mục tiêu là chứng minh mô hình/vòng train có thể ghi nhớ batch nhỏ; không diễn giải đây là khả năng khái quát hóa tốt.
- Kết quả gốc: [overfit_resnet_metrics.csv](overfit_resnet_metrics.csv).

## Kiểm tra resume và mã nguồn

- `python -m pytest -q`: **6 passed**, gồm gradient/shape, học batch nhỏ, validation không đổi BatchNorm và tính loss đúng trọng số mẫu, resume với dropout/shuffle, chống ghi đè/config khác, split pilot.
- `python -m compileall -q src tests`: đạt; `git diff --check`: đạt.
- Đối chiếu GPU thật: `runs/minh_week1_uninterrupted` chạy liền 3 epoch, cùng config với run ngắt/resume. **Trọng số/buffer mô hình bitwise bằng nhau**, momentum optimizer bằng nhau, scheduler và loss/accuracy/lr từng epoch bằng nhau; cùng 108 bước. Thời gian chạy được phép khác.
- Đọc lại log TensorBoard bằng EventAccumulator: validation có đúng epoch **1, 2, 3**, không trùng bước sau resume.
- CSV đối chiếu: [pilot_resnet_uninterrupted_metrics.csv](pilot_resnet_uninterrupted_metrics.csv); tóm tắt máy đọc: [week1_summary.json](week1_summary.json).

## Phần còn chờ để nghiệm thu toàn bộ tuần 1

1. ViT của Lan Anh: chạy cùng trainer, kiểm tra đầu ra/attention và xác nhận số tham số; điều chỉnh width ResNet trước loạt chính.
2. Split/manifest chính của Hiền: thay adapter pilot và kiểm tra hai mô hình dùng chung dữ liệu.
3. Nơi sao lưu/chia sẻ log của nhóm; chưa tải log/checkpoint lên dịch vụ ngoài. Không cần mời GitHub theo giảng viên, như Minh xác nhận.
4. Minh đọc bài/giải thích code cùng nhóm; gửi [đoạn proposal](../reports/proposal_minh.md) cho Hiền và kiểm tra PDF/hạn nộp thực tế.

Mã/config đã được rà theo skill code-review-and-quality: không phát hiện lỗi chặn trong phạm vi pilot; giới hạn hiện tại là resume tại epoch boundary, num_workers=0, FP32 và dữ liệu chẩn đoán. Chưa kiểm tra cài sạch toàn bộ requirements trên máy khác. Checkpoint/dữ liệu nằm ngoài Git theo `.gitignore`; các file CSV/JSON nhỏ ở đây là bằng chứng có thể đưa vào repo.
