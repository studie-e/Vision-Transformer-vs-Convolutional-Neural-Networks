# So sánh Vision Transformer và CNN

Đề tài 3 môn Học sâu. Lan Anh: ViT; Minh: ResNet/trainer/môi trường; Hiền: dữ liệu/đánh giá/attention. [Kế hoạch](tasks/plan.md) · [Checklist](tasks/todo.md) · [Bàn giao tuần 1 của Minh](docs/minh-week1.md).

Phần dữ liệu chính, evaluator, bảng từng seed/mean/std và proposal: [bàn giao tuần 1 của Hiền](docs/hien-week1-handoff.md). Config chạy tích hợp: `configs/hien_smoke.yaml`; lệnh `python -X utf8 -m scripts.check_hien_week1 --device cpu` kiểm tra một epoch trên split chính và lưu bằng chứng pilot riêng.

## Giao thức nhóm mới nhất — ViT GPU hỗ trợ Lan Anh

Dùng [sáu YAML hiện hành](configs/week2/README.md) và [hướng dẫn nhóm](docs/bao-cao-tuan2-lan-anh-ho-tro.md): ViT **AdamW lr 0.0003, wd 0.05, dropout 0.1**; ResNet **SGD lr 0.05, momentum 0.9, wd 0.0005**. Cùng 100 epoch, batch 128, split seed 2026/train seed 42; giữ test cuối. Minh vận hành ba lượt ViT GPU hỗ trợ Lan Anh theo yêu cầu mới.

Phần tuning/khóa cũ bên dưới được giữ để truy vết; ưu tiên YAML `vit_10.yaml`, `vit_50.yaml`, `vit_100.yaml` khi chạy P0 hiện hành.

**Đã chạy đủ sáu lượt P0 100 epoch:** ViT validation 52,16% / 67,66% / 75,90%; ResNet 74,24% / 88,40% / 91,80%. [Bảng, hình và bằng chứng hiện hành](results/team_week2/README.md). Gói gửi Hiền nằm ở `runs/handoff/team_week2_all_runs.zip` (chưa gửi ra ngoài). `results/runs.csv` chỉ chứa sáu lượt này; kết quả CPU 5 epoch cũ được lưu riêng.

## Tuần 2 của Minh

[Bàn giao và recipe tuần 2](docs/minh-week2.md) · [Method–CNN](reports/method_minh_week2.md). ViT và split chính đã tích hợp; dùng các YAML trong `configs/week2/` cho thí nghiệm chính, giữ `configs/train.yaml` làm pilot tuần 1.

Đã hoàn tất ba lượt ResNet × 100 epoch: validation **74,24% / 88,40% / 91,80%** ở 10%/50%/100% train. [Kết quả và biểu đồ](results/week2/minh_week2.md) · [CSV metadata](results/week2/resnet_runs.csv). Một seed, std N/A; chưa mở test hoặc chạy các lượt ViT chính thay Lan Anh.

Đã khóa 100 epoch cho cả ba tỷ lệ, cặp seed `(split=2026, train=42)`. Sau ngân sách tuning ngang nhau, ResNet chọn SGD và ViT chọn AdamW. Cấu hình ViT chính được bàn giao Lan Anh; script dưới đây thực thi ba lượt ResNet của Minh:

```powershell
python -X utf8 -m src.data --verify --verify-labels
python -X utf8 -m scripts.run_minh_week2 --stage tune
python -X utf8 -m scripts.run_minh_week2 --stage train
python -X utf8 -m scripts.run_minh_week2 --stage collect
python -X utf8 -m scripts.plot_minh_week2
```

Nếu chưa có CIFAR-10 bản Python, chuẩn bị theo hướng dẫn dữ liệu của Hiền trước khi verify. Thêm `--resume` để tiếp tục một giai đoạn dở. `locked_protocol.json` bảo vệ recipe đã khóa; kết quả tuần 2 chỉ dùng validation, chưa mở test. CSV tuning và các lượt chính được tách tại `results/week2/`; checkpoint/log nằm trong `runs/week2/`.

## Môi trường

Đã kiểm tra trên Windows, Python 3.13.7, RTX 5080, PyTorch 2.11.0+cu128. Cài mới trong PowerShell từ thư mục repo:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

Các lệnh `python` bên dưới dùng interpreter đã cài requirements; với môi trường mới, thay bằng `.venv\Scripts\python`. Phiên kiểm tra hiện tại dùng các gói đã có trên máy, chưa phải một lượt cài sạch.

## ResNet và pilot tuần 1

```powershell
python -m src.inspect_model
python -m src.train --config configs/overfit.yaml --run-dir runs/resnet_overfit_new
python -m src.train --config configs/train.yaml --run-dir runs/resnet_pilot_new --stop-after-epoch 1
python -m src.train --config configs/train.yaml --run-dir runs/resnet_pilot_new --resume runs/resnet_pilot_new/last.pt
python -m tensorboard.main --logdir runs --host localhost
```

Mở địa chỉ TensorBoard do lệnh cuối in ra. Không dùng lại tên run cũ nếu muốn chạy từ đầu; trainer từ chối ghi đè. `--stop-after-epoch 1` dừng sau epoch 1 nhưng giữ nguyên lịch learning rate của tổng số epoch đã cấu hình. Resume chỉ từ `last.pt` của cùng thư mục run và giữ nguyên config.

Pilot tải CIFAR-10 dạng ảnh từ kho fastai, dùng riêng 4.500 ảnh train/1.000 validation lấy từ **tập train**; không dùng tập test chính thức. Overfit dùng 32 ảnh train và 32 ảnh validation khác nhau. Chuẩn hóa cố định về [-1, 1], không augmentation. Đây là kiểm tra kỹ thuật, **chưa phải split phân tầng 45k/5k hay kết quả 10/50/100% chính thức của Hiền**. [Nguồn kho CIFAR của fastai](https://docs.fast.ai/data.external.html).

## Sản phẩm của mỗi run

- `config.yaml`, `metadata.json`: cấu hình, seed, commit/trạng thái Git, hash mã nguồn, phần cứng và phiên bản.
- `metrics.csv`: epoch/step/lr, loss/accuracy train và validation, thời gian, bộ nhớ CUDA. Accuracy lưu theo thang 0–1.
- `tensorboard/`: log có thể mở bằng TensorBoard; chưa tự tải lên dịch vụ ngoài.
- `best.pt`: checkpoint có validation accuracy cao nhất (hòa thì giữ bản sớm hơn); `last.pt`: để resume.

Checkpoint chứa model, optimizer, scheduler, epoch, bước cập nhật, RNG Python/NumPy/PyTorch/CUDA và generator của DataLoader. Resume ở ranh giới epoch; phiên bản đầu yêu cầu `num_workers=0`. Không cam kết tái lập bitwise giữa các máy/phiên bản khác nhau. Chỉ tải checkpoint tin cậy; mã dùng `weights_only=True`.

`data/`, `runs/`, trọng số và dữ liệu lớn được loại khỏi Git. Sao lưu cả thư mục run ra nơi nhóm dùng chung trước khi bàn giao. Theo xác nhận mới của Minh, giảng viên **không yêu cầu gửi lời mời GitHub**; phần quyền chia sẻ log được xác nhận riêng theo yêu cầu môn.

## Tích hợp ViT và dữ liệu chính

Trainer nhận factory ở dạng `module:function` trong YAML. Lan Anh cung cấp `build_model(config) -> nn.Module`, đầu vào `[B,3,32,32]`, đầu ra logits `[B,10]`. Hiền cung cấp `build_loaders(config) -> (train_loader, val_loader)`, batch `(images, labels)`; split/seed/phần trăm phải nằm trong config hoặc manifest cố định. Không cần sửa vòng train để đổi mô hình.

Đếm tham số thật của ViT bằng config chứa factory của Lan Anh, rồi tìm chiều rộng ResNet phù hợp:

```powershell
python -m src.inspect_model --config configs/vit.yaml
python -m src.inspect_model --target 272474
```

Lan Anh đã giao ViT: lệnh thứ nhất đếm 274.282 tham số. Số `272474` ở lệnh thứ hai là ví dụ cú pháp, khi khớp ViT thay bằng `274282`; ResNet hiện tại 272.474 tham số, lệch 0,659%. `configs/resnet.yaml` mô tả riêng mô hình; `configs/train.yaml` là config pilot đầy đủ, còn `configs/week2/` chứa các cấu hình chính.

Hiền đã bàn giao evaluator/tổng hợp, Lan Anh đã bàn giao ViT. Các lệnh pilot tuần 1 chỉ train/validation; lệnh tuần 2 dùng dữ liệu chính và đối chiếu evaluator, vẫn chưa mở test. Attention/bảng so sánh đủ hai mô hình thuộc bước tiếp theo của nhóm.

## Nguồn và phần tự cài

ResNet cho ảnh nhỏ được viết trong repo, dựa trên [He et al., 2015](https://arxiv.org/abs/1512.03385), dùng projection 1×1 ở chỗ đổi kích thước; không sao chép nguyên thí nghiệm CIFAR của bài gốc. Trainer/checkpoint dùng API PyTorch; [nguồn và ghi chú](docs/minh-week1.md). Mã và tài liệu tuần 1 có hỗ trợ AI; Minh cần đọc, kiểm tra và khai báo đúng phạm vi khi viết báo cáo.
