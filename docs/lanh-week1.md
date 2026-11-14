# Bàn giao tuần 1 — Lan Anh

## Phạm vi và trạng thái

Ngày thực hiện: **14/11/2026**, trên nhánh `lanh`. Đã merge code tuần 1 của Minh (trainer, ResNet, checkpoint) và Hiền (data pipeline, evaluator, proposal) từ branch `hien`.

Đã tự cài Vision Transformer (ViT) bằng PyTorch, tích hợp với trainer chung, kiểm tra forward/backward/overfit/attention/ablation, và khớp tham số với ResNet. Bằng chứng kiểm tra ghi trong phần dưới; pilot chạy dữ liệu thật ghi riêng khi hoàn thành.

## ViT tự cài

### Nguồn kiến trúc
Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).

### Các thành phần đã tự cài (`src/models/vit.py`)

| Thành phần | Mô tả | Tự cài? |
|---|---|---|
| **PatchEmbedding** | Conv2d(3, embed_dim, 4, 4) — chia ảnh 32×32 thành 64 patch 4×4 | ✅ Tự cài |
| **CLS Token** | `nn.Parameter(1, 1, embed_dim)` — token phân loại learnable | ✅ Tự cài |
| **Positional Embedding** | `nn.Parameter(1, 65, embed_dim)` — learnable, có ablation mode | ✅ Tự cài |
| **MultiHeadSelfAttention** | Q/K/V projection → scaled dot-product → concat → output | ✅ Tự cài |
| **MLP Block** | Linear → GELU → Dropout → Linear → Dropout | ✅ Tự cài |
| **TransformerBlock** | Pre-LN: LN → MHSA → residual → LN → MLP → residual | ✅ Tự cài |
| **Classification Head** | LayerNorm(CLS token) → Linear(embed_dim, 10) | ✅ Tự cài |

Các thành phần PyTorch cơ bản (`nn.Linear`, `nn.LayerNorm`, `nn.Conv2d`, `nn.GELU`, `nn.Dropout`) được dùng làm building block. Không dùng mô hình ViT hoàn chỉnh có sẵn từ thư viện nào.

### Cấu hình đã chốt (khớp ResNet)

| Tham số | Giá trị |
|---|---|
| image_size | 32 |
| patch_size | 4 (→ 8×8 = 64 patch) |
| embed_dim | 96 |
| depth | 4 blocks |
| num_heads | 4 |
| mlp_ratio | 1.5 (mlp_dim = 144) |
| dropout | 0.1 |
| num_classes | 10 |

### So sánh tham số

| Mô hình | Tham số trainable | Chênh lệch |
|---|---|---|
| ViT (cấu hình trên) | **274,282** | — |
| ResNet (width=16, 3 block/stage) | **272,474** | — |
| Chênh lệch | | **0.7%** ✅ |
| ViT ablation (bỏ pos. embed) | **268,042** | −6,240 |

## Kết quả kiểm tra tuần 1

| Kiểm tra | Kết quả |
|---|---|
| Forward pass: logits shape [B, 10] | ✅ Đạt |
| Backward pass: tất cả tham số có gradient | ✅ Đạt (56 tham số) |
| Đếm tham số: chênh lệch ≤ 10% ResNet | ✅ Đạt (0.66%) |
| Overfit 1 batch nhỏ (50 steps) | ✅ Loss 2.33 → 0.001 |
| Attention weights: shape đúng | ✅ [B, 4, 65, 65] |
| Ablation mode: forward không lỗi | ✅ Đạt |
| Ablation: pos_embed là buffer, không Parameter | ✅ Đạt |
| Suite kiểm thử tự động pytest (`tests/test_lanh.py`) | ✅ 9/9 test đạt |
| Pilot dữ liệu thật 4.500 ảnh CIFAR-10 (1 epoch) | ✅ 20,34% val acc (1.017/5.000) |
| Evaluator và Trainer đồng thuận | ✅ Khớp 100% |
| Mẫu Attention weights trên ảnh thật | ✅ Đã lưu tại `runs/lanh_week1_smoke/sample_attention.pt` |

## Pilot trên dữ liệu thật CIFAR-10

- **Cấu hình:** 1 epoch, batch size 128, tỷ lệ 10% (4.500 ảnh train, 5.000 val), split seed 2026, train seed 42, SGD lr=0.05, momentum=0.9.
- **Kết quả huấn luyện:**
  - 36 steps cập nhật
  - Train loss: 2.2947, Train accuracy: 17,40%
  - Val loss: 2.1755, Val accuracy: **20,34%** (1.017 / 5.000 ảnh)
  - Thời gian: 16,57s trên CPU
- **Đối chiếu:** `src.evaluate` nạp `best.pt` trả đúng 0.2034, khớp tuyệt đối với `best_val_accuracy` được lưu trong checkpoint bởi Trainer.
- **Bằng chứng kỹ thuật:**
  - `results/lanh_pilot_runs.csv`: metadata chi tiết của lượt chạy pilot.
  - `results/lanh_pilot_summary.csv`: bảng tóm tắt 1 seed.
  - `results/lanh_pilot_metrics.csv`: log từng epoch.
  - `results/lanh_week1_checks.json`: xác nhận kỹ thuật tự động.
  - `runs/lanh_week1_smoke/sample_attention.pt`: mẫu tensor 4 layer attention weights trên batch validation cho Hiền.

## Cách lấy attention weights (cho Hiền)

```python
model.eval()
with torch.no_grad():
    logits = model(images)
attn_weights = model.get_attention_weights()
# attn_weights: list[Tensor], mỗi phần tử [B, num_heads, N, N]
# N = 65 (1 CLS + 64 patch tokens)
# len(attn_weights) = depth = 4
```

## Tích hợp với trainer Minh

Config YAML:
```yaml
model:
  factory: src.models.vit:build_model
  kwargs:
    image_size: 32
    patch_size: 4
    embed_dim: 96
    depth: 4
    num_heads: 4
    mlp_ratio: 1.5
    num_classes: 10
    dropout: 0.1
    use_pos_embed: true
```

Lệnh train (tương tự ResNet):
```powershell
python -X utf8 -m src.train --config configs/vit.yaml --run-dir runs/vit_pilot --device cuda
```

## Lệnh tái lập kiểm tra

```powershell
python -X utf8 -m scripts.check_lanh_week1
```

## Bản đồ file để Lan Anh vấn đáp

| File | Lan Anh cần giải thích được |
|---|---|
| `src/models/vit.py` | Patch embedding, CLS token, positional embedding, multi-head self-attention, MLP, encoder block, classification head |
| `configs/vit.yaml` | Vì sao embed_dim=96, depth=4, heads=4, mlp_ratio=1.5; cách khớp tham số |
| `src/train.py` (Minh) | Cách factory hoạt động; seed; CrossEntropy nhận logits; train khác eval |
| `src/data.py` (Hiền) | Cách split/manifest hoạt động; augmentation chỉ trên train |

## Câu hỏi tự kiểm tra khi vấn đáp

1. Vì sao ViT chia ảnh thành patch thay vì xử lý từng pixel?
2. CLS token dùng để làm gì? Vì sao không dùng average pooling tất cả token?
3. Positional embedding cần thiết vì sao? Attention có tự biết vị trí token không?
4. Scaled dot-product attention chia cho √d_k để làm gì?
5. Pre-LN khác Post-LN như thế nào? Ưu điểm của Pre-LN?
6. Vì sao embed_dim=96 mà không phải 128? Liên quan gì đến ResNet?
7. Bỏ positional embedding thì ViT mất khả năng gì? Dự đoán accuracy sẽ thay đổi thế nào?
8. GELU khác ReLU ở điểm nào? Vì sao Transformer hay dùng GELU?

## Lan Anh cần tự thực hiện sau khi nhận bàn giao

- Đọc bài ViT và giải thích code cho Minh/Hiền; AI chưa thể xác nhận thay Lan Anh đã hiểu bài.
- Kiểm tra pilot ViT trên GPU khi dữ liệu CIFAR-10 đã tải xong.
- Viết đoạn proposal về ViT gửi Hiền.
- Khi viết báo cáo, khai báo AI đã hỗ trợ cài ViT; Lan Anh kiểm tra, điều chỉnh và chịu trách nhiệm giải thích.
