# Minh vận hành GPU hỗ trợ Lan Anh — ViT tuần 2

## Tài liệu đã đọc và cấu hình ưu tiên

Đã đọc [báo cáo hỗ trợ Lan Anh](bao-cao-tuan2-lan-anh-ho-tro.md), [giao thức nhóm](experiment_protocol.md), `docs/lanh-week2.md` và các YAML trên `origin/hien`/`origin/lanh`. Mã được fetch và merge vào nhánh `Minh`, giữ lại bằng chứng ResNet đã hoàn thành. Xung đột checklist được xử lý theo quyết định nhóm mới nhất, không dùng kết quả 5 epoch để xác nhận đủ 100 epoch.

Giao thức thực thi đúng yêu cầu nhóm: `split_seed=2026`, `seed=42`, batch 128, 100 epoch, cosine annealing; ViT AdamW lr **0.0003**, weight decay 0.05, dropout 0.1. Ba tập train 4.500 / 22.500 / 45.000 ảnh và validation 5.000 ảnh lấy từ cùng manifest với ResNet. Chọn best.pt theo top-1 validation accuracy; hòa thì giữ epoch sớm hơn. Không đánh giá test chính thức.

Các file `configs/week2/vit_10.yaml`, `vit_50.yaml`, `vit_100.yaml` là cấu hình hiện hành. Những YAML `vit_d*_s42.yaml` hoặc `configs/vit_d*.yaml` thuộc các giai đoạn thử trước đó; không trộn vào bảng sáu lượt P0. Xem [phân biệt cấu hình](../configs/week2/README.md).

## Các lệnh thực thi

```powershell
python -X utf8 -m src.train --config configs/week2/vit_10.yaml --run-dir runs/week2_vit_10_seed42 --device cuda
python -X utf8 -m src.train --config configs/week2/vit_50.yaml --run-dir runs/week2_vit_50_seed42 --device cuda
python -X utf8 -m src.train --config configs/week2/vit_100.yaml --run-dir runs/week2_vit_100_seed42 --device cuda
```

Các lượt chạy tuần tự, dừng hàng đợi nếu trainer lỗi. Stdout từng lượt tại `runs/team_week2_logs/`. Khi bị ngắt, dùng đúng config và `--resume <run_dir>/last.pt`, không giảm tổng epoch hoặc đổi lr để tiếp tục một run cũ.

## ResNet dùng để đối chiếu

Đã so sánh dict YAML của ba thư mục `runs/week2/resnet_d{10,50,100}_s42/` với `configs/week2/resnet_{10,50,100}.yaml`: **khớp toàn bộ cấu hình**, đủ 100 epoch. Các kết quả 74,24% / 88,40% / 91,80% validation được dùng lại với đúng checkpoint/provenance gốc, không đổi tên rồi tính thành lượt chạy mới.

## Gói bàn giao nội bộ

Người dùng đã xác nhận đóng ZIP cục bộ để chuyển cho Hiền. Mỗi gói chứa **toàn bộ thư mục run**, tối thiểu:

- `config.yaml`, `metadata.json`, `metrics.csv`.
- `best.pt`, `last.pt`.
- `tensorboard/` và các dự đoán/tệp phụ nếu đã được xuất.

Script `scripts/package_team_week2.py` kiểm tra cấu hình nhóm, SHA-256 manifest, 100 epoch, số cập nhật, lịch sử CSV và quy tắc best checkpoint trước khi nén; kiểm tra CRC ZIP và xuất `.sha256.txt` sau khi nén. Đây là đóng gói nội bộ, không tự tải file lên dịch vụ hoặc gửi tin nhắn cho người khác.

Để đóng gói cả sáu run sau khi đã hoàn tất:

```powershell
python -X utf8 -m scripts.package_team_week2 --out runs/handoff/team_week2 --run runs/week2/resnet_d10_s42 --run runs/week2/resnet_d50_s42 --run runs/week2/resnet_d100_s42 --run runs/week2_vit_10_seed42 --run runs/week2_vit_50_seed42 --run runs/week2_vit_100_seed42
```

Thư mục đích có sáu ZIP, checksum và `handoff_manifest.json`. Chỉ mục nhỏ và bảng kết quả được lưu trong repo; checkpoint/ZIP lớn vẫn ở `runs/` theo `.gitignore`. Khi sao chép cho Hiền, đối chiếu SHA-256 với manifest để phát hiện file thiếu/hỏng.

## Trực quan hóa và trạng thái bàn giao

Sau khi đủ sáu run, dùng `src.visualize --label validation` trên đúng sáu thư mục để đối chiếu checkpoint/log, vẽ learning curve, confusion matrix và attention ViT. Công cụ chạy validation, không mở test. Script hiện tại **không tự cập nhật `run_matrix.csv`**; trạng thái ma trận phải cập nhật riêng theo việc đã kiểm tra và việc Hiền thực sự nhận file.

Checkpoint C chỉ được đánh dấu đủ sáu lượt khi cả ba ViT 100 epoch đã hoàn tất và qua đối chiếu. Trạng thái `ready_for_handoff` nghĩa là file đã sẵn sàng cục bộ, không có nghĩa Hiền đã tiếp nhận/đọc chéo. Các số liệu, hình và chỉ mục gói hoàn tất sẽ được ghi trong `results/team_week2/`.

## Ghi nhận đóng góp

Lan Anh phụ trách kiến trúc ViT, mã attention và phần viết phương pháp. Minh cung cấp RTX 5080 và vận hành các lượt GPU, kiểm tra/đóng gói. Hiền cung cấp dữ liệu/giao thức và tiếp nhận, phân tích kết quả. Việc vận hành trên máy Minh không thay đổi tác giả phần mô hình. Một seed không đủ để ước lượng std mẫu; test cuối và hai seed tiếp theo vẫn theo kế hoạch nhóm.

## Kết quả hoàn tất

Ba ViT đã đủ 100 epoch; best validation 52,16% / 67,66% / 75,90% ở 10/50/100%. Sáu run P0 đã đối chiếu bằng evaluator CUDA và có ZIP đầy đủ. [Báo cáo kết quả và chỉ mục gói](../results/team_week2/README.md). Gói gộp `runs/handoff/team_week2_all_runs.zip`, 33.833.022 byte; checksum ở file `.sha256.txt` cạnh gói. Chưa gửi cho Hiền.

Công cụ visual được bổ sung `--device`, mặc định theo metadata train; tránh ép GPU run đánh giá bằng CPU, vốn có thể khác dự đoán trên ảnh sát ranh giới do số học dấu phẩy động. Với ViT 10%, CPU đếm 2.607/5.000 và CUDA đếm 2.608/5.000; số liệu bàn giao dùng CUDA đúng thiết bị train. Đã kiểm thử hồi quy, toàn suite 39 test đạt. Không sửa checkpoint, không nới điều kiện khớp log.
