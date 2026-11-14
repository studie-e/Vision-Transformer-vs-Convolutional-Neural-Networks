"""Vision Transformer (ViT) tự cài cho ảnh nhỏ CIFAR-10 32×32.

Nguồn kiến trúc: Dosovitskiy et al., "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale", https://arxiv.org/abs/2010.11929

Tự cài toàn bộ: patch embedding, positional embedding, multi-head self-attention,
transformer encoder block và classification head. Không dùng mô hình ViT hoàn chỉnh
có sẵn từ thư viện. Các thành phần PyTorch cơ bản (nn.Linear, nn.LayerNorm, v.v.)
được dùng làm building block theo yêu cầu đề tài.

Cách lấy attention weights cho Hiền trực quan hoá:
    model.eval()
    logits = model(images)             # forward bình thường
    attn_weights = model.get_attention_weights()  # list[Tensor], mỗi phần tử [B, heads, N, N]
"""

import math

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# 1. Patch Embedding
# ---------------------------------------------------------------------------
# PatchEmbedding: chia ảnh thành patch rồi chiếu tuyến tính sang embed_dim.
# Ảnh 32×32 patch 4×4 → lưới 8×8 = 64 patch, mỗi patch là vector embed_dim chiều.
# Dùng Conv2d với kernel=stride=patch_size, tương đương reshape + linear nhưng gọn hơn.
class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        # Kiểm tra ảnh chia hết cho patch.
        if image_size % patch_size != 0:
            raise ValueError(f'image_size {image_size} không chia hết cho patch_size {patch_size}')
        # Số patch = (image_size / patch_size)² — đây là chiều dài chuỗi token.
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        # Conv2d kernel=stride=patch_size: mỗi vùng patch_size×patch_size thành 1 pixel embed_dim kênh.
        # Tương đương: reshape (B,3,H,W) → (B, num_patches, 3*P*P) rồi Linear → embed_dim.
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                                    stride=patch_size)

    # Forward: [B, 3, 32, 32] → [B, 64, embed_dim] cho patch 4×4.
    def forward(self, x: Tensor) -> Tensor:
        # Conv2d: [B, 3, 32, 32] → [B, embed_dim, 8, 8]
        x = self.projection(x)
        # Flatten spatial: [B, embed_dim, 8, 8] → [B, embed_dim, 64]
        x = x.flatten(2)
        # Chuyển vị: [B, embed_dim, 64] → [B, 64, embed_dim] — chuỗi token.
        x = x.transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# 2. Multi-Head Self-Attention
# ---------------------------------------------------------------------------
# MultiHeadSelfAttention: tính attention giữa tất cả token trong chuỗi.
# Mỗi token query tất cả token khác qua Q·K^T/√d rồi lấy tổng trọng số V.
# Chia thành num_heads head song song, mỗi head xử lý embed_dim/num_heads chiều.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # Kiểm tra embed_dim chia hết cho num_heads.
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim {embed_dim} không chia hết cho num_heads {num_heads}')
        self.num_heads = num_heads
        # Chiều mỗi head: d_k = embed_dim / num_heads.
        self.head_dim = embed_dim // num_heads
        # Hệ số scale trong scaled dot-product attention: 1/√d_k.
        self.scale = self.head_dim ** -0.5

        # Projection Q, K, V: từ embed_dim → embed_dim, nhưng chia thành num_heads × head_dim.
        # Gộp thành 1 Linear cho hiệu quả, sau đó reshape.
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # Output projection: gộp các head lại → embed_dim.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Dropout trên attention weights.
        self.attn_dropout = nn.Dropout(dropout)
        # Dropout trên output.
        self.proj_dropout = nn.Dropout(dropout)

        # Lưu attention weights cho trực quan hoá (Hiền dùng).
        self._attn_weights: Tensor | None = None

    # Forward: [B, N, D] → [B, N, D] với N = num_tokens, D = embed_dim.
    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape

        # Tính Q, K, V cùng lúc: [B, N, 3*D] → reshape → [B, N, 3, heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # Permute → [3, B, heads, N, head_dim] → tách Q, K, V.
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # mỗi cái [B, heads, N, head_dim]

        # Scaled dot-product attention: attention = softmax(Q·K^T / √d_k).
        # [B, heads, N, head_dim] × [B, heads, head_dim, N] → [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Lưu attention weights (detach khỏi graph để không ảnh hưởng training).
        self._attn_weights = attn.detach()

        attn = self.attn_dropout(attn)

        # Nhân attention với V: [B, heads, N, N] × [B, heads, N, head_dim] → [B, heads, N, head_dim]
        x = attn @ v
        # Gộp heads: [B, heads, N, head_dim] → [B, N, heads*head_dim] = [B, N, D]
        x = x.transpose(1, 2).reshape(B, N, D)
        # Output projection.
        x = self.out_proj(x)
        x = self.proj_dropout(x)
        return x


# ---------------------------------------------------------------------------
# 3. MLP Block (Feed-Forward Network)
# ---------------------------------------------------------------------------
# MLP: Linear → GELU → Dropout → Linear → Dropout.
# Mở rộng chiều lên mlp_dim rồi thu về embed_dim. GELU là hàm kích hoạt mượt hơn ReLU.
class MLP(nn.Module):
    def __init__(self, embed_dim: int = 128, mlp_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            # Lớp ẩn: embed_dim → mlp_dim.
            nn.Linear(embed_dim, mlp_dim),
            # GELU: xấp xỉ x·Φ(x), mượt hơn ReLU, phổ biến trong Transformer.
            nn.GELU(),
            nn.Dropout(dropout),
            # Lớp ra: mlp_dim → embed_dim.
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. Transformer Encoder Block
# ---------------------------------------------------------------------------
# TransformerBlock: Pre-LN architecture (LayerNorm trước, không phải sau).
# Cấu trúc: x + MHSA(LN(x)) rồi x + MLP(LN(x)). Residual connection giúp
# gradient chảy trực tiếp, tương tự ý tưởng của ResNet.
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        # LayerNorm chuẩn hoá theo chiều embed_dim (khác BatchNorm theo batch).
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        # mlp_dim = embed_dim × mlp_ratio, ví dụ 128 × 2 = 256.
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-LN: norm trước attention/MLP, sau đó cộng residual.
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 5. Vision Transformer
# ---------------------------------------------------------------------------
# VisionTransformer: ghép patch embedding → CLS token + positional embedding
# → N transformer blocks → LayerNorm → classification head.
class VisionTransformer(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 10,
                 embed_dim: int = 128, depth: int = 4, num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1,
                 use_pos_embed: bool = True):
        """
        Args:
            image_size: kích thước ảnh vuông (32 cho CIFAR).
            patch_size: kích thước mỗi patch vuông (4 → 8×8 = 64 patch).
            in_channels: số kênh ảnh (3 cho RGB).
            num_classes: số lớp phân loại (10 cho CIFAR-10).
            embed_dim: chiều embedding mỗi token.
            depth: số transformer encoder blocks.
            num_heads: số attention heads mỗi block.
            mlp_ratio: tỷ lệ mở rộng MLP = mlp_dim / embed_dim.
            dropout: tỷ lệ dropout.
            use_pos_embed: True dùng positional embedding; False bỏ (cho ablation).
        """
        super().__init__()
        self.use_pos_embed = use_pos_embed

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 64 cho 32×32, patch 4×4

        # --- CLS Token ---
        # Token đặc biệt gắn vào đầu chuỗi, dùng để phân loại.
        # Learnable, khởi tạo gần 0 bằng truncated normal.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- Positional Embedding ---
        # Learnable, 1 vector cho mỗi vị trí (CLS + num_patches).
        # Ablation: khi use_pos_embed=False, không cộng positional embedding.
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        else:
            # Không tạo parameter, đăng ký buffer zero để forward thống nhất.
            self.register_buffer('pos_embed', torch.zeros(1, 1 + num_patches, embed_dim))

        self.pos_dropout = nn.Dropout(dropout)

        # --- Transformer Encoder ---
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # --- Classification Head ---
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Khởi tạo trọng số.
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số theo cách của ViT gốc."""
        # CLS token: truncated normal std=0.02
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Positional embedding: truncated normal std=0.02
        if self.use_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Khởi tạo Linear và LayerNorm.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform cho weights của Linear (phù hợp Transformer).
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: ảnh [B, 3, H, W]
        Returns:
            logits [B, num_classes] — chưa softmax, dùng cho CrossEntropyLoss.
        """
        B = x.shape[0]

        # Patch embedding: [B, 3, 32, 32] → [B, 64, embed_dim]
        x = self.patch_embed(x)

        # Nối CLS token vào đầu: [B, 1, embed_dim] expand thành B bản → nối → [B, 65, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Cộng positional embedding (hoặc cộng 0 nếu ablation).
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer encoder blocks.
        x = self.blocks(x)

        # Lấy CLS token (vị trí 0) → LayerNorm → Linear → logits.
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits

    def get_attention_weights(self) -> list[Tensor]:
        """Trả attention weights của tất cả encoder blocks sau forward gần nhất.

        Returns:
            List[Tensor], mỗi phần tử có shape [B, num_heads, N, N]
            với N = 1 + num_patches (CLS + patch tokens).
            Dùng cho trực quan hoá; không tham gia backward.
        """
        weights = []
        for block in self.blocks:
            w = block.attn._attn_weights
            if w is not None:
                weights.append(w)
        return weights


# ---------------------------------------------------------------------------
# 6. count_parameters và build_model (factory cho trainer chung)
# ---------------------------------------------------------------------------
# Import count_parameters từ resnet để dùng chung; cũng export lại ở đây.
from src.models.resnet import count_parameters  # noqa: E402


def build_model(config: dict) -> nn.Module:
    """Factory function cho trainer — tạo ViT từ config YAML.

    Config mẫu:
        model:
          factory: src.models.vit:build_model
          kwargs:
            image_size: 32
            patch_size: 4
            embed_dim: 128
            depth: 4
            num_heads: 4
            mlp_ratio: 2.0
            num_classes: 10
            dropout: 0.1
            use_pos_embed: true
    """
    return VisionTransformer(**config['model']['kwargs'])
