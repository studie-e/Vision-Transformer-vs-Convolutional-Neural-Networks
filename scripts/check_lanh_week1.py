"""Kiểm tra tuần 1 Lan Anh — ViT tự cài.

Chạy: python -X utf8 -m scripts.check_lanh_week1

Kiểm tra:
1. Forward pass: logits đúng shape [B, 10]
2. Backward pass: tất cả tham số có gradient
3. Đếm tham số: so sánh chênh lệch với ResNet 272.474
4. Overfit 1 batch nhỏ: loss giảm rõ rệt
5. Lấy attention weights: đúng shape
6. Ablation mode: bỏ positional embedding vẫn forward được
"""

import sys
import torch
from torch import nn

from src.models.vit import VisionTransformer, count_parameters, build_model
from src.models.resnet import CifarResNet, count_parameters as resnet_count


def main():
    device = 'cpu'
    print("=" * 60)
    print("KIỂM TRA TUẦN 1 — LAN ANH — ViT tự cài")
    print("=" * 60)
    errors = []

    # --- 1. Forward pass ---
    print("\n[1/6] Forward pass...")
    config = {
        'model': {
            'kwargs': {
                'image_size': 32, 'patch_size': 4, 'embed_dim': 96,
                'depth': 4, 'num_heads': 4, 'mlp_ratio': 1.5,
                'num_classes': 10, 'dropout': 0.1, 'use_pos_embed': True,
            }
        }
    }
    model = build_model(config).to(device)
    batch = torch.randn(4, 3, 32, 32, device=device)
    logits = model(batch)
    expected = (4, 10)
    if logits.shape == expected:
        print(f"  OK: logits shape = {logits.shape}")
    else:
        msg = f"  FAIL: logits shape = {logits.shape}, expected {expected}"
        print(msg)
        errors.append(msg)

    # --- 2. Backward pass ---
    print("\n[2/6] Backward pass...")
    loss = nn.functional.cross_entropy(logits, torch.randint(0, 10, (4,), device=device))
    loss.backward()
    no_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if not no_grad:
        print(f"  OK: tất cả {sum(1 for _ in model.parameters())} tham số có gradient")
    else:
        msg = f"  FAIL: {len(no_grad)} tham số không có gradient: {no_grad[:5]}..."
        print(msg)
        errors.append(msg)

    # --- 3. Đếm tham số ---
    print("\n[3/6] Đếm tham số và so sánh ResNet...")
    vit_params = count_parameters(model)
    resnet = CifarResNet(width=16, blocks_per_stage=3, num_classes=10)
    resnet_params = resnet_count(resnet)
    diff_pct = abs(vit_params - resnet_params) / resnet_params * 100
    print(f"  ViT:    {vit_params:>10,} tham số")
    print(f"  ResNet: {resnet_params:>10,} tham số")
    print(f"  Chênh lệch: {diff_pct:.1f}%")
    if diff_pct <= 10:
        print(f"  OK: chênh lệch ≤ 10%")
    else:
        msg = f"  CẢNH BÁO: chênh lệch {diff_pct:.1f}% > 10% — cần điều chỉnh embed_dim/depth"
        print(msg)
        errors.append(msg)

    # --- 4. Overfit 1 batch nhỏ ---
    print("\n[4/6] Overfit 1 batch nhỏ (50 steps)...")
    model.zero_grad()
    model.train()
    small_batch = torch.randn(8, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (8,), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    first_loss = None
    last_loss = None
    for step in range(50):
        optimizer.zero_grad()
        out = model(small_batch)
        l = nn.functional.cross_entropy(out, labels)
        l.backward()
        optimizer.step()
        if step == 0:
            first_loss = l.item()
        if step == 49:
            last_loss = l.item()

    print(f"  Loss đầu: {first_loss:.4f}")
    print(f"  Loss cuối: {last_loss:.4f}")
    if last_loss < first_loss * 0.5:
        print(f"  OK: loss giảm > 50%")
    else:
        msg = f"  CẢNH BÁO: loss không giảm đủ (đầu={first_loss:.4f}, cuối={last_loss:.4f})"
        print(msg)
        errors.append(msg)

    # --- 5. Attention weights ---
    print("\n[5/6] Lấy attention weights...")
    model.eval()
    with torch.no_grad():
        _ = model(batch)
    attn_weights = model.get_attention_weights()
    num_blocks = 4
    if len(attn_weights) == num_blocks:
        print(f"  OK: {len(attn_weights)} blocks")
        shape = attn_weights[0].shape
        # Expect [B, heads, N, N] with N = 1 + 64 = 65
        expected_n = 1 + (32 // 4) ** 2  # 65
        if shape == (4, 4, expected_n, expected_n):
            print(f"  OK: attention shape = {shape}")
        else:
            msg = f"  FAIL: attention shape = {shape}, expected (4, 4, {expected_n}, {expected_n})"
            print(msg)
            errors.append(msg)
    else:
        msg = f"  FAIL: {len(attn_weights)} blocks, expected {num_blocks}"
        print(msg)
        errors.append(msg)

    # --- 6. Ablation mode ---
    print("\n[6/6] Ablation mode (no positional embedding)...")
    config_abl = {
        'model': {
            'kwargs': {
                'image_size': 32, 'patch_size': 4, 'embed_dim': 96,
                'depth': 4, 'num_heads': 4, 'mlp_ratio': 1.5,
                'num_classes': 10, 'dropout': 0.1, 'use_pos_embed': False,
            }
        }
    }
    model_abl = build_model(config_abl).to(device)
    logits_abl = model_abl(batch)
    if logits_abl.shape == (4, 10):
        print(f"  OK: ablation forward shape = {logits_abl.shape}")
    else:
        msg = f"  FAIL: ablation shape = {logits_abl.shape}"
        print(msg)
        errors.append(msg)

    # Kiểm tra pos_embed không phải Parameter trong ablation.
    has_pos_param = any(name == 'pos_embed' for name, _ in model_abl.named_parameters())
    if not has_pos_param:
        print(f"  OK: pos_embed không phải Parameter (là buffer zero)")
    else:
        msg = f"  FAIL: pos_embed vẫn là Parameter trong ablation mode"
        print(msg)
        errors.append(msg)

    abl_params = count_parameters(model_abl)
    pos_params = vit_params - abl_params
    print(f"  Params bình thường: {vit_params:,}")
    print(f"  Params ablation:    {abl_params:,}")
    print(f"  Positional embed:   {pos_params:,} tham số bị bỏ")

    # --- Tổng kết ---
    print("\n" + "=" * 60)
    if not errors:
        print("✅ TẤT CẢ KIỂM TRA ĐẠT — ViT tuần 1 sẵn sàng")
    else:
        print(f"❌ {len(errors)} VẤN ĐỀ CẦN XỬ LÝ:")
        for e in errors:
            print(f"   • {e}")
    print("=" * 60)
    return len(errors)


if __name__ == '__main__':
    sys.exit(main())
