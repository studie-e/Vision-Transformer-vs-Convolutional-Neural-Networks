"""Kiểm thử tự động cho các thành phần ViT của Lan Anh (tuần 1):
- src.models.vit: PatchEmbedding, MultiHeadSelfAttention, MLP, TransformerBlock, VisionTransformer
- Khớp tham số so với ResNet: chênh lệch ≤10% (T04, T05)
- Cơ chế xuất attention weights cho Hiền trực quan hóa
- Chế độ ablation: bỏ positional embedding
- Khả năng tạo mô hình qua build_model factory cho trainer của Minh
"""

import pytest
import torch
from torch import nn

from src.models.resnet import CifarResNet, count_parameters as count_resnet_params
from src.models.vit import (
    MLP,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
    VisionTransformer,
    build_model,
    count_parameters,
)


@pytest.fixture
def default_config():
    return {
        'model': {
            'factory': 'src.models.vit:build_model',
            'kwargs': {
                'image_size': 32,
                'patch_size': 4,
                'embed_dim': 96,
                'depth': 4,
                'num_heads': 4,
                'mlp_ratio': 1.5,
                'num_classes': 10,
                'dropout': 0.1,
                'use_pos_embed': True,
            },
        }
    }


def test_patch_embedding_shape():
    pe = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=96)
    assert pe.num_patches == 64
    x = torch.randn(2, 3, 32, 32)
    out = pe(x)
    assert out.shape == (2, 64, 96)

    # Lỗi khi kích thước không chia hết
    with pytest.raises(ValueError):
        PatchEmbedding(image_size=32, patch_size=5)


def test_multi_head_attention_and_weights():
    mha = MultiHeadSelfAttention(embed_dim=96, num_heads=4, dropout=0.0)
    x = torch.randn(2, 65, 96)
    out = mha(x)
    assert out.shape == (2, 65, 96)
    assert mha._attn_weights is not None
    assert mha._attn_weights.shape == (2, 4, 65, 65)

    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=96, num_heads=5)


def test_mlp_block():
    mlp = MLP(embed_dim=96, mlp_dim=144, dropout=0.0)
    x = torch.randn(2, 65, 96)
    out = mlp(x)
    assert out.shape == (2, 65, 96)


def test_transformer_block():
    block = TransformerBlock(embed_dim=96, num_heads=4, mlp_ratio=1.5, dropout=0.0)
    x = torch.randn(2, 65, 96)
    out = block(x)
    assert out.shape == (2, 65, 96)


def test_vit_forward_and_backward(default_config):
    model = build_model(default_config)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)

    # Test backward pass
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"


def test_vit_parameter_budget(default_config):
    model = build_model(default_config)
    vit_params = count_parameters(model)
    resnet = CifarResNet(width=16, blocks_per_stage=3, num_classes=10)
    resnet_params = count_resnet_params(resnet)

    diff_pct = abs(vit_params - resnet_params) / resnet_params * 100
    assert diff_pct <= 10.0, f"Chênh lệch {diff_pct:.2f}% vượt ngưỡng 10%"
    assert vit_params == 274282
    assert resnet_params == 272474


def test_vit_attention_extraction(default_config):
    model = build_model(default_config)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        _ = model(x)
    weights = model.get_attention_weights()
    assert len(weights) == 4
    for w in weights:
        assert w.shape == (2, 4, 65, 65)
        # Kiểm tra attention weights đã được normalize qua softmax
        sums = w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_vit_ablation_mode():
    config = {
        'model': {
            'kwargs': {
                'image_size': 32,
                'patch_size': 4,
                'embed_dim': 96,
                'depth': 4,
                'num_heads': 4,
                'mlp_ratio': 1.5,
                'num_classes': 10,
                'dropout': 0.1,
                'use_pos_embed': False,
            }
        }
    }
    model = build_model(config)
    # pos_embed không nằm trong parameters khi ablation
    param_names = [n for n, _ in model.named_parameters()]
    assert 'pos_embed' not in param_names
    # Nhưng vẫn nằm trong buffers
    buffer_names = [n for n, _ in model.named_buffers()]
    assert 'pos_embed' in buffer_names

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    assert count_parameters(model) == 268042


def test_vit_overfit_single_batch(default_config):
    torch.manual_seed(42)
    # Tắt dropout khi kiểm tra overfit batch nhỏ
    cfg = dict(default_config)
    cfg['model'] = dict(default_config['model'])
    cfg['model']['kwargs'] = dict(default_config['model']['kwargs'])
    cfg['model']['kwargs']['dropout'] = 0.0

    model = build_model(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    initial_loss = None
    final_loss = None
    for step in range(50):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < 0.05, f"Loss không giảm đủ: {initial_loss:.4f} -> {final_loss:.4f}"
