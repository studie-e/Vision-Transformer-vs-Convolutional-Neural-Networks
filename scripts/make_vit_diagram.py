# Người phụ trách: Lan Anh (sinh sơ đồ kiến trúc ViT cho báo cáo ACL và slide tuần 4).
"""Vẽ sơ đồ kiến trúc Vision Transformer tự cài (274.282 tham số) và biến thể Ablation."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, x, y, w, h, title, subtitle='', facecolor='#EBF3FA', edgecolor='#173752', lw=1.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.04,rounding_size=0.08',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.68, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#102538')
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha='center', va='center',
                fontsize=8.8, color='#2C3E50', linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='#102538')


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color='#173752', lw=1.8, shrinkA=2, shrinkB=2)
    )


def main():
    out_dir = Path('reports/figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.2, 4.5))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    # Hàng trên: Luồng tensor chính của ViT
    draw_box(ax, 0.20, 2.55, 2.25, 1.75,
             'Ảnh CIFAR-10', 'RGB [B, 3, 32, 32]\nLưới 8×8 patch 4×4',
             facecolor='#FDF2E9', edgecolor='#BA4A00')
    draw_arrow(ax, 2.52, 3.42, 2.98, 3.42)

    draw_box(ax, 3.05, 2.55, 2.65, 1.75,
             'Patch Embedding', 'Conv2d(3→96, k=4, s=4)\nChuỗi [B, 64, 96]',
             facecolor='#EBF5FB', edgecolor='#1B4F72')
    draw_arrow(ax, 5.77, 3.42, 6.23, 3.42)

    draw_box(ax, 6.30, 2.55, 3.15, 1.75,
             'Ghép CLS + PosEmbed', 'CLS [1,1,96] + E_pos [1,65,96]\nTensor [B, 65, 96] (Ablation: 0)',
             facecolor='#E8F8F5', edgecolor='#117864')
    draw_arrow(ax, 9.52, 3.42, 9.98, 3.42)

    draw_box(ax, 10.05, 2.55, 2.65, 1.75,
             'Encoder × 4', 'Pre-LN · 4 Heads (d_k=24)\nMLP 96→144→96 (GELU)',
             facecolor='#F4ECF7', edgecolor='#6C3483')
    draw_arrow(ax, 12.77, 3.42, 13.23, 3.42)

    draw_box(ax, 13.30, 2.55, 1.70, 1.75,
             'CLS Head', 'LayerNorm\nLinear → [B, 10]',
             facecolor='#EAFAF1', edgecolor='#1E8449')

    # Hàng dưới: Chi tiết bên trong 1 Pre-LN Transformer Block + Xuất Attention
    draw_box(ax, 0.50, 0.35, 9.20, 1.55,
             'Chi tiết 1 khối Pre-LN Transformer Block (lặp L = 4 lần, tổng 262.080 tham số)',
             "z'_l = z_{l-1} + MHSA(LN(z_{l-1}))    →    z_l = z'_l + MLP(LN(z'_l))    |    Dropout = 0.1",
             facecolor='#F8F9F9', edgecolor='#566573', lw=1.3)

    draw_box(ax, 10.25, 0.35, 4.65, 1.55,
             'Trích xuất Attention (Hiền)',
             'get_attention_weights()\n4 tầng × [B, 4, 65, 65] (CLS → 64 patch)',
             facecolor='#FEF9E7', edgecolor='#B7950B', lw=1.3)
    draw_arrow(ax, 11.38, 2.48, 12.10, 1.98)

    ax.text(7.6, 4.82,
            'Vision Transformer tự cài (274.282 tham số; Ablation bỏ E_pos: 268.042 tham số)',
            ha='center', va='center', fontsize=12.0, fontweight='bold', color='#173752')

    fig.tight_layout(pad=0.15)
    fig.savefig(out_dir / 'vit_architecture_lanh.pdf', bbox_inches='tight')
    fig.savefig(out_dir / 'vit_architecture_lanh.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('Saved clean reports/figures/vit_architecture_lanh.{pdf,png}')


if __name__ == '__main__':
    main()
