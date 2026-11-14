# Cấu hình nhóm hiện hành — tuần 2

Giao thức được nhóm xác nhận: [experiment_protocol.md](../../docs/experiment_protocol.md), [bàn giao GPU ViT](../../docs/bao-cao-tuan2-lan-anh-ho-tro.md).

| Run | YAML hiện hành | Recipe |
|---|---|---|
| ViT 10/50/100% | `vit_10.yaml`, `vit_50.yaml`, `vit_100.yaml` | AdamW, lr **0.0003**, wd 0.05, dropout 0.1 |
| ResNet 10/50/100% | `resnet_10.yaml`, `resnet_50.yaml`, `resnet_100.yaml` | SGD, lr 0.05, momentum 0.9, wd 0.0005 |

Cùng split seed 2026, train seed 42, batch 128, 100 epoch và cosine annealing. Các tỷ lệ tính trên 45.000 train; chọn best.pt theo accuracy 5.000 validation, hòa giữ epoch sớm hơn; test giữ đến cuối.

Các file `*_d10_s42.yaml`, `*_d50_s42.yaml`, `*_d100_s42.yaml` và `locked_protocol.json` là bản khóa từ tuning trước quyết định mới của nhóm. Giữ nguyên chúng để truy vết lịch sử. Phần ResNet tương đương YAML hiện hành; phần ViT cũ dùng lr 0.001 nên **không dùng làm cấu hình P0 hiện hành**. Script `scripts.run_minh_week2 --stage tune` thuộc quy trình cũ; không gọi lại để thay giao thức nhóm đã chốt.

Các file `configs/vit_d{10,50,100}.yaml` và kết quả ViT CPU 5 epoch là lượt ngắn trước giao thức 100 epoch. Không trộn chúng vào bảng sáu lượt P0.
