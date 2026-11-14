"""Vẽ learning curve và data scaling ResNet từ CSV thật của tuần 2."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Hàm read_csv: đọc file CSV thành list dict để vẽ, không tính lại hay sửa số liệu thí nghiệm.
# Nhận Path; trả danh sách dòng, các giá trị số được chuyển khi sử dụng ở main.
def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


# Hàm main: vẽ ba cặp đường loss/accuracy và validation accuracy theo tỷ lệ dữ liệu.
# Chỉ đọc kết quả ResNet một seed; không tạo error bar hay kết luận so sánh ViT khi chưa có dữ liệu.
def main():
    output = Path('results/week2/figures')
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_csv('results/week2/resnet_runs.csv'), key=lambda r: float(r['ratio']))
    if len(rows) != 3 or {r['split'] for r in rows} != {'val'}:
        raise ValueError('Expected three validation rows, one per training ratio')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for column, row in enumerate(rows):
        history = read_csv(Path(row['checkpoint_path']).parent / 'metrics.csv')
        epochs = [int(r['epoch']) for r in history]
        for series, label, color in (('train', 'Train', 'tab:blue'), ('val', 'Validation', 'tab:orange')):
            axes[0, column].plot(epochs, [float(r[f'{series}_loss']) for r in history], label=label, color=color)
            axes[1, column].plot(epochs, [100 * float(r[f'{series}_accuracy']) for r in history], label=label, color=color)
        axes[0, column].set_title(f'ResNet — {100 * float(row["ratio"]):.0f}% train data')
        axes[0, column].set_ylabel('Cross-entropy loss')
        axes[1, column].set_ylabel('Accuracy (%)')
        axes[1, column].set_xlabel('Epoch')
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
            axis.legend()
    fig.savefig(output / 'resnet_learning_curves.png', dpi=180)
    fig.savefig(output / 'resnet_learning_curves.pdf')
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ratios = [100 * float(r['ratio']) for r in rows]
    scores = [100 * float(r['accuracy']) for r in rows]
    axis.plot(ratios, scores, marker='o', label='ResNet — split 2026, train seed 42')
    for x, y in zip(ratios, scores, strict=True):
        axis.annotate(f'{y:.2f}%', (x, y), xytext=(0, 8), textcoords='offset points', ha='center')
    axis.set(xlabel='Fraction of 45,000 training images (%)', ylabel='Best-checkpoint validation accuracy (%)',
             title='Week 2: ResNet data scaling — one seed', xticks=ratios)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(output / 'resnet_data_scaling.png', dpi=180)
    fig.savefig(output / 'resnet_data_scaling.pdf')
    plt.close(fig)
    print(f'Saved figures to {output}')


if __name__ == '__main__':
    main()
