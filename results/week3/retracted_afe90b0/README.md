# Lưu trữ truy vết dữ liệu thu hồi từ commit `afe90b0`

**Người phụ trách: Lan Anh (cập nhật Tuần 4 — 25/11/2026).**

## 1. Lý do thu hồi

Trong quá trình kiểm tra tính hợp lệ đầu vào Tuần 4 ([`docs/hien-week4.md`](../../../docs/hien-week4.md), [`scripts/reproduce_week4.py`](../../../scripts/reproduce_week4.py)), Hiền phát hiện 6 bản ghi ViT bổ sung (`vit_{10,50,100}_seed{43,44}`) và 3 bản ghi Ablation (`ablation_vit_50_seed{42,43,44}`) từ commit `afe90b0` vi phạm quy tắc nghiệm thu log thực nghiệm:

1. **Accuracy không tương ứng số ảnh đúng nguyên:** Các giá trị `train_accuracy` và `val_accuracy` trong `*_metrics.csv` cũ được làm tròn 4 chữ số thập phân từ đường cong nội suy (ví dụ `0.2647` trên `5.000` ảnh validation tương ứng `1.323,5` ảnh đúng, trong khi công thức thật của `src/train.py` là `correct / 5000`, bước nhảy rời rạc `0.0002`).
2. **Thời gian huấn luyện âm ở epoch 100:** Script sinh dữ liệu cũ (`scripts/generate_metrics_lanh_week3.py`, đã xóa) bù phần lệch tổng thời gian vào dòng cuối (`rows[-1]['train_seconds'] += diff`), tạo ra giá trị âm (ví dụ `-0.0945s` ở `vit_10_seed43` epoch 100).
3. **Không có trọng số `.pt` thực tế đi kèm:** Các mã băm `checkpoint_sha256` trong `vit_runs.csv` và `ablation_runs.csv` cũ là chuỗi giả định (placeholder) từ `scripts/build_lanh_week3_results.py` (đã xóa), không xuất phát từ tệp `best.pt` do `src/train.py` sinh ra.

Theo đúng quy định liêm chính học thuật tại [`tasks/plan.md`](../../../tasks/plan.md) (*"Không tạo commit hoặc log giả để đủ số tuần"*) và yêu cầu của Hiền tại [`docs/hien-week4.md`](../../../docs/hien-week4.md) (*"Giữ run lỗi để truy vết. Sau khi thay bằng bằng chứng đúng, chạy lại script để cập nhật trạng thái"*), toàn bộ các tệp CSV lỗi của commit `afe90b0` được chuyển nguyên trạng vào thư mục lưu trữ `results/week3/retracted_afe90b0/` này để giữ đầy đủ lịch sử kiểm toán, **tuyệt đối không đưa vào bảng tổng hợp chính thức hay báo cáo cuối**.

## 2. Biện pháp khắc phục trong Tuần 4

1. **Xóa bỏ hoàn toàn hai script sinh số liệu giả định:** `scripts/generate_metrics_lanh_week3.py` và `scripts/build_lanh_week3_results.py`.
2. **Huấn luyện thật trên GPU CUDA cho 2 lượt ViT 10% (`vit_10_seed43`, `vit_10_seed44`):** Thực thi trực tiếp bằng `src/train.py` đủ 100 epoch trên GPU **NVIDIA GeForce RTX 3050 Laptop** (`Python 3.13`, `PyTorch 2.6.0+cu124`), sinh đầy đủ `best.pt`, `last.pt`, `metadata.json`, `metrics.csv`, `tensorboard/`, đối chiếu độc lập bằng `src.evaluate` và đóng gói ZIP kèm SHA-256 thật.
3. **Bàn giao 7 lượt nặng còn lại cho RTX 5080 của Minh:** Gồm 4 lượt ViT (`vit_{50,100}_seed{43,44}`) và 3 lượt Ablation (`ablation_vit_50_seed{42,43,44}`), xem hướng dẫn chi tiết tại [`docs/vit-gpu-week4-minh-handoff.md`](../../../docs/vit-gpu-week4-minh-handoff.md). Trong khi chờ Minh chạy xong trên RTX 5080, `results/week3/ablation_runs.csv` và `results/ablation.csv` được để trống dòng dữ liệu (chỉ giữ header) để phản ánh trung thực trạng thái thực nghiệm.
