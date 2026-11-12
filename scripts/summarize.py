"""Script tổng hợp kết quả — Hiền.

Đọc results/runs.csv (do src.evaluate tạo ra), tính mean ± std theo từng
(model, ratio, split, protocol) trên các seed, sinh bảng Markdown và CSV tổng hợp.

Dùng: python scripts/summarize.py
      python scripts/summarize.py --csv results/runs.csv --split val --out results/summary.csv

Ghi chú:
  - Chỉ tổng hợp các dòng có cùng split (val mặc định).
  - Độ lệch chuẩn mẫu (ddof=1); không đủ seed thì ghi NaN.
  - Không tự chọn seed tốt nhất; mọi seed hợp lệ đều được tính vào.
  - Bảng Markdown in ra terminal để kiểm tra nhanh; CSV lưu lại để Hiền dùng trong báo cáo.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Đọc runs.csv
# ---------------------------------------------------------------------------

def load_runs(csv_path, split_filter=None):
    """Nạp results/runs.csv, lọc theo split nếu cần.

    Đầu vào:
        csv_path:     đường dẫn file CSV.
        split_filter: 'val', 'test' hoặc None (nạp tất cả).
    Trả về:
        list of dict, mỗi dict là một dòng trong CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Không tìm thấy {path}\n'
            f'Chạy: python -m src.evaluate --config <cfg> --checkpoint <ckpt> để tạo kết quả.')
    rows = []
    with path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if split_filter is None or row.get('split') == split_filter:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Tổng hợp mean ± std
# ---------------------------------------------------------------------------

def _mean_std(values):
    """Tính mean và độ lệch chuẩn mẫu (ddof=1) của danh sách số thực.

    Trả về (mean, std). Nếu ít hơn 2 phần tử thì std = NaN.
    """
    n = len(values)
    if n == 0:
        return float('nan'), float('nan')
    mean = sum(values) / n
    if n == 1:
        return mean, float('nan')
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


def summarize(rows):
    """Nhóm theo (model, ratio, split, protocol), từ chối seed trùng.

    Đầu vào:
        rows: list of dict từ load_runs().
    Trả về:
        list of dict, mỗi dict là một nhóm (model, ratio) với:
            model, ratio, n_seeds, mean_accuracy, std_accuracy,
            accuracies (list float), split_seeds (list str), train_seeds (list str),
            params (str), global_steps (list int).
    """
    # Nhóm dòng theo (model, ratio).
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        required = ('model', 'ratio', 'split', 'protocol_id', 'train_seed', 'split_seed', 'params')
        if any(str(row.get(k, '')).strip() == '' for k in required):
            raise ValueError('Missing required experiment metadata')
        if row['split'] not in ('val', 'test'):
            raise ValueError('Unknown evaluation split')
        accuracy = float(row['accuracy'])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError('Accuracy must be finite and between zero and one')
        ratio = f"{float(row['ratio']):.2f}"
        key = (row['model'], ratio, row['split'], row['protocol_id'])
        identity = (*key, int(row['train_seed']))
        if identity in seen:
            raise ValueError('Duplicate training seed in an experiment group')
        seen.add(identity)
        groups[key].append(row)

    summary = []
    for (model, ratio, split, protocol), group_rows in sorted(groups.items()):
        group_rows.sort(key=lambda r: (int(r['train_seed']), int(r['split_seed'])))
        if len({str(r['params']) for r in group_rows}) != 1:
            raise ValueError('Inconsistent parameter counts within one protocol')
        accs = []
        split_seeds, train_seeds, global_steps = [], [], []
        params = ''
        for r in group_rows:
            try:
                accs.append(float(r['accuracy']))
            except (ValueError, KeyError):
                pass
            split_seeds.append(r.get('split_seed', ''))
            train_seeds.append(r.get('train_seed', ''))
            if r.get('global_step'):
                try:
                    global_steps.append(int(r['global_step']))
                except ValueError:
                    pass
            if r.get('params'):
                params = r['params']

        mean, std = _mean_std(accs)
        summary.append({
            'model':          model,
            'split':          split,
            'protocol_id':    protocol,
            'ratio':          ratio,
            'n_seeds':        len(accs),
            'mean_accuracy':  mean,
            'std_accuracy':   std,
            'accuracies':     accs,
            'split_seeds':    split_seeds,
            'train_seeds':    train_seeds,
            'params':         params,
            'global_steps':   global_steps,
        })

    return summary


# ---------------------------------------------------------------------------
# Xuất bảng
# ---------------------------------------------------------------------------

def _fmt(val, digits=4):
    """Định dạng số thực; NaN → 'N/A'."""
    if val != val:   # NaN
        return 'N/A'
    return f'{val:.{digits}f}'


def print_markdown_table(summary, split='val'):
    """In bảng Markdown ra stdout."""
    print(f'\n## Bảng kết quả tổng hợp ({split})\n')
    print('| Model | Ratio | Split | Protocol | N seeds | Mean Acc | Std Acc | Accuracies per seed |')
    print('|---|---|---|---|---:|---:|---:|---|')
    for s in summary:
        accs_str = ', '.join(f'{split_seed}/{train_seed}: {_fmt(a)}'
                            for split_seed, train_seed, a in zip(s['split_seeds'], s['train_seeds'], s['accuracies']))
        print(f"| {s['model']} | {s['ratio']} | {s['split']} | {s['protocol_id'][:12]} | {s['n_seeds']} "
              f"| {_fmt(s['mean_accuracy'])} | {_fmt(s['std_accuracy'])} | {accs_str} |")


def save_summary_csv(summary, out_path='results/summary.csv'):
    """Lưu bảng tổng hợp ra CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'model', 'ratio', 'split', 'protocol_id', 'n_seeds', 'mean_accuracy', 'std_accuracy',
        'accuracies', 'split_seeds', 'train_seeds', 'params', 'global_steps',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary:
            writer.writerow({
                **s,
                'accuracies':   str(s['accuracies']),
                'split_seeds':  str(s['split_seeds']),
                'train_seeds':  str(s['train_seeds']),
                'global_steps': str(s['global_steps']),
            })
    return path


# ---------------------------------------------------------------------------
# Kiểm tra bằng ví dụ biết đáp án
# ---------------------------------------------------------------------------

def _verify():
    """Kiểm tra _mean_std bằng ví dụ tính tay."""
    # Ví dụ 1: 3 seed, accuracy 0.70/0.72/0.74 → mean=0.72, std=0.02
    vals = [0.70, 0.72, 0.74]
    mean, std = _mean_std(vals)
    assert abs(mean - 0.72) < 1e-10, f'mean sai: {mean}'
    # std mẫu = sqrt(((0.02)^2 + 0 + (0.02)^2) / 2) = sqrt(0.0004) = 0.02
    expected_std = math.sqrt(((0.70 - 0.72)**2 + 0 + (0.74 - 0.72)**2) / 2)
    assert abs(std - expected_std) < 1e-10, f'std sai: {std}'

    # Ví dụ 2: 1 seed → std = NaN
    mean2, std2 = _mean_std([0.85])
    assert mean2 == 0.85
    assert math.isnan(std2), 'std 1 seed phải là NaN'

    # Ví dụ 3: danh sách rỗng → mean và std đều NaN
    mean3, std3 = _mean_std([])
    assert math.isnan(mean3) and math.isnan(std3)

    print('[OK] _mean_std kiểm tra bằng ví dụ tính tay — tất cả đạt.')
    return True


# ---------------------------------------------------------------------------
# Điểm vào dòng lệnh
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',    default='results/runs.csv',
                        help='File runs.csv đầu vào (mặc định: results/runs.csv).')
    parser.add_argument('--split',  default='val', choices=('val', 'test', 'all'),
                        help='Lọc theo split (mặc định: val).')
    parser.add_argument('--out',    default='results/summary.csv',
                        help='File CSV đầu ra (mặc định: results/summary.csv).')
    parser.add_argument('--verify', action='store_true',
                        help='Chỉ kiểm tra _mean_std bằng ví dụ biết đáp án.')
    args = parser.parse_args()

    if args.verify:
        _verify()
        sys.exit(0)

    split_filter = None if args.split == 'all' else args.split
    rows = load_runs(args.csv, split_filter=split_filter)

    if not rows:
        print(f'Không có dòng nào trong {args.csv} với split={args.split!r}.')
        print('Chạy src.evaluate để tạo kết quả trước.')
        sys.exit(1)

    summary = summarize(rows)
    print_markdown_table(summary, split=args.split)

    out_path = save_summary_csv(summary, out_path=args.out)
    print(f'\nLưu tổng hợp vào: {out_path}')

    # Danh sách điểm cần kiểm tra hoặc bổ sung.
    print('\n## Điểm cần kiểm tra:')
    for s in summary:
        if s['n_seeds'] < 3:
            print(f"  [THIẾU SEED] {s['model']} ratio={s['ratio']}: "
                  f"có {s['n_seeds']} seed, cần 3")
        if math.isnan(s['std_accuracy']) and s['n_seeds'] > 1:
            print(f"  [LỖI STD] {s['model']} ratio={s['ratio']}: "
                  f"không tính được std dù có {s['n_seeds']} seed")
        if not s['params']:
            print(f"  [THIẾU PARAMS] {s['model']} ratio={s['ratio']}: "
                  f"metadata.json chưa có số tham số")

