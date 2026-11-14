# Bàn giao tuần 2 — Minh

## Phạm vi và quy trình

Ngày bắt đầu: **14/11/2026**, Asia/Saigon. Đã pull `main` đến commit `6211fe0`, sau đó fast-forward nhánh `Minh` để triển khai. Tiếp nối đúng phân công: Minh phụ trách ResNet/trainer, chuẩn bị Swin; Lan Anh phụ trách các lượt ViT chính; Hiền phân tích và tổng hợp dữ liệu.

Minh giao quyền chốt recipe cho trợ lý trong phiên này. Quy trình thực hiện:

1. Kiểm tra dữ liệu/split của Hiền với nhãn thật, kiểm tra khớp tham số với ViT của Lan Anh.
2. Thử hai recipe cho **mỗi** kiến trúc trên cùng 50% train: SGD và AdamW, mỗi lượt 10 epoch, cùng seed `(split=2026, train=42)`.
3. Chọn recipe riêng theo best validation accuracy; hòa giữ SGD; khóa trước các lượt chính. Test không tham gia tuning/chọn checkpoint.
4. ResNet chạy **100 epoch ở cả 10%, 50%, 100%**, từ đầu, một seed trong tuần 2. Các YAML ViT tương ứng được chuẩn bị để Lan Anh chạy, không tuyên bố đã có kết quả ViT chính.
5. Đối chiếu evaluator của Hiền với trainer, lưu log/checkpoint/config/CSV và chuẩn bị bản viết Method–CNN.

Hai recipe có cùng số lượt, số ảnh và số epoch tuning; không đồng nghĩa thời gian GPU bằng nhau hoặc bảo đảm tìm được cấu hình tối ưu toàn cục. Recipe riêng là một yếu tố khác biệt cần khai báo khi phân tích kiến trúc.

## Recipe đã khóa sau tuning thực tế

| Mô hình | SGD (10 epoch) | AdamW (10 epoch) | Recipe chọn |
|---|---:|---:|---|
| ResNet | 77,74% validation | 72,20% validation | SGD, lr 0,05, momentum 0,9, weight decay 0,0005 |
| ViT | 50,24% validation | 55,02% validation | AdamW, lr 0,001, weight decay 0,05 |

Tất cả ứng viên dùng 22.500 train / 5.000 validation, batch 128, split seed 2026, train seed 42, cosine schedule trong 10 epoch. Mỗi mô hình được thử hai recipe độc lập; kết quả là best validation accuracy trong lượt thử. Đây là số **tuning**, không phải kết quả chính hay kết quả test. Các lượt chính khởi tạo lại từ đầu, dùng cosine schedule 100 epoch, không fine-tune từ checkpoint tuning. Bằng chứng: `results/week2/tuning.csv` và `configs/week2/locked_protocol.json`.

## Dữ liệu và kiến trúc

- Dữ liệu CIFAR-10 archive bản Python, giữ nguyên thứ tự gốc mà torchvision và manifest của Hiền sử dụng. Archive tải qua mirror Hugging Face vì máy chủ Toronto chậm; MD5 **`c58f30108f718f92721af3b95e74349a`** trùng giá trị torchvision yêu cầu, rồi qua kiểm tra checksum từng batch. Không dùng ảnh PNG pilot tuần 1 cho các lượt tuần 2.
- Nguồn mirror: `https://huggingface.co/datasets/VerisimilitudeX/cifar10/resolve/main/cifar-10-python.tar.gz`; đối chiếu checksum với [mã dataset chính thức](https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py).
- Manifest: `results/splits/split_seed2026_manifest.json`, train 45.000 / validation 5.000; ba tập train 4.500 / 22.500 / 45.000 ảnh, phân tầng và lồng nhau.
- Pipeline giữ nguyên của Hiền: train crop/flip, val chỉ tensor/chuẩn hóa. SHA-256 manifest được lưu trong metadata/checkpoint.
- ResNet: width 16, 3 block/stage, projection shortcut, **272.474 tham số**. ViT tự cài hiện tại **274.282 tham số**, lệch **0,659%** theo công thức chia số lớn hơn. Không cần đổi width.
- Seed mới dùng theo bàn giao thực tế của Hiền/proposal: `(2026,42)`; hai cặp sau là `(2027,43)`, `(2028,44)`. Các run ID `s17` trong lộ trình ban đầu chỉ là đề xuất trước khi nhóm có bàn giao dữ liệu; run thực tế tuần 2 dùng `s42` và lưu riêng split seed.

## Cấu hình và lệnh chạy

```powershell
python -X utf8 -m src.data --verify --verify-labels
python -m pytest -q
python -X utf8 -m scripts.run_minh_week2 --stage tune
python -X utf8 -m scripts.run_minh_week2 --stage train
python -X utf8 -m scripts.run_minh_week2 --stage collect
python -X utf8 -m scripts.plot_minh_week2
```

Để tiếp tục giai đoạn bị ngắt, thêm `--resume`. Script kiểm tra config/manifest và số epoch; run đã đủ epoch được kiểm tra rồi dùng lại, không tính thành một seed mới. Muốn xem một run, dùng CLI trainer như tuần 1 với đúng config lưu trong run và `last.pt`.

- `configs/week2/tuning/`: bốn cấu hình thử có ngân sách ngang nhau.
- `configs/week2/locked_protocol.json`: recipe được chọn, số epoch, seed, source hash và bản chụp đầy đủ các cấu hình chính. Không sửa file này để hợp thức hóa một run đã dùng recipe khác.
- `configs/week2/resnet_d{10,50,100}_s42.yaml`: ba cấu hình Minh thực thi.
- `configs/week2/vit_d{10,50,100}_s42.yaml`: cấu hình bàn giao Lan Anh, không sửa model factory/kiến trúc ViT.
- `runs/week2/tuning/`: checkpoint/log các lượt chọn recipe, tách khỏi bảng chính.
- `runs/week2/resnet_d{10,50,100}_s42/`: checkpoint/log các lượt chính; stdout ở `runs/week2/logs/`.
- `results/week2/`: CSV tuning và các kết quả ResNet chính khi thực sự hoàn tất; hình vẽ ở `results/week2/figures/`.

Khác biệt thời gian train phải được báo cáo bằng đo đạc thực tế. `train_seconds` không gồm validation; `epoch_seconds` gồm train+val tới trước lúc ghi checkpoint/log. Bộ nhớ đỉnh trong CSV là MiB tensor CUDA được cấp phát, không phải toàn VRAM của tiến trình.

## Prototype Swin cho tuần 3

`src/models/swin.py` dùng lớp `SwinTransformer` của **torchvision 0.26**, không tự nhận là nhóm tự cài Swin. Cấu hình ảnh nhỏ: patch 2×2, embed dim 26, depth `[2,2,2]`, heads `[2,4,8]`, window 4×4, MLP ratio 2, 10 lớp. Có **261.824 tham số**, khởi tạo từ đầu; chỉ kiểm tra logits/gradient, **chưa huấn luyện Swin**.

Nguồn: [API Swin](https://docs.pytorch.org/vision/0.26/models/swin_transformer.html), [implementation chính thức](https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/swin_transformer.html). Config mô hình: `configs/week2/swin_prototype.yaml`. Huấn luyện phần mở rộng vẫn sau các lượt chính theo kế hoạch.

## Đọc chéo ViT và phần Minh cần nắm

- `PatchEmbedding` dùng conv stride bằng patch size: ảnh 32×32, patch 4×4 thành 64 token, mỗi token 96 chiều.
- Thêm CLS token thành chuỗi `[B,65,96]`, cộng positional embedding và qua 4 block pre-LN.
- Mỗi block: LayerNorm → attention → cộng residual; LayerNorm → MLP → cộng residual. Bốn head, mỗi head 24 chiều; QK được scale theo `1/sqrt(24)`.
- ViT trả logits `[B,10]` giống ResNet, nên hai mô hình dùng cùng trainer. Attention tách riêng bằng `get_attention_weights()` để Hiền trực quan hóa.
- Ablation tắt positional embedding chuyển nó thành buffer toàn 0; số tham số giảm, cần ghi đúng khi phân tích. Minh không sửa logic ViT của Lan Anh trong tuần 2.

Các ghi chú này là hướng dẫn đọc code. Việc Minh tự hiểu và giải thích phải được xác nhận qua buổi đọc chéo/vấn đáp, không tự đánh dấu thay người học.

## Trạng thái nghiệm thu

Theo dõi mục T05/T06/T09/T11 và sản phẩm tuần 2 trong `tasks/todo.md`. Đã hoàn tất ba run ResNet 100 epoch, evaluator và trainer khớp chính xác; [báo cáo kết quả cuối tuần 2](../results/week2/minh_week2.md), [bảng từng run](../results/week2/resnet_runs.csv), [bộ kiểm tra](../results/week2/minh_week2_checks.json). Validation lần lượt 74,24% / 88,40% / 91,80%; 36 test đạt, không test bị skip. ViT chính/attention chung và test cuối chưa được đánh dấu hoàn thành.

Tuần 2 mới một seed: std mẫu phải là N/A, chưa kết luận ổn định và chưa thể khẳng định CNN thắng/thua ViT từ pilot. Tiếp tục hai seed ResNet còn lại ở tuần 3; giữ test chính thức đến khi khóa toàn bộ phương pháp/checkpoint cuối.
