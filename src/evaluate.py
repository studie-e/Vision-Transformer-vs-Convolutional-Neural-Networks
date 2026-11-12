"""Script đánh giá — Hiền.

Đánh giá một checkpoint đã lưu trên tập validation hoặc test; sinh nhãn dự đoán và phân phối
xác suất. Tạo metadata đầy đủ cho mỗi lượt chạy để lưu vào results/runs.csv.

Dùng: python -m src.evaluate --config <cfg.yaml> --checkpoint <best.pt> --split val|test
      python -m src.evaluate --verify  # kiểm tra metric bằng ví dụ biết đáp án

Lưu ý:
  - Test chỉ được đánh giá SAU khi toàn bộ phương pháp và checkpoint đã được khóa.
  - Validation dùng để theo dõi và chọn checkpoint trong quá trình train.
  - Mọi kết quả đánh giá được ghi vào results/runs.csv với metadata đầy đủ.
"""

import argparse
import csv
import json
import hashlib
import copy
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch import nn
from torchvision import datasets

from src.data import build_test_loader, load_manifest, make_eval_transform, validate_manifest, manifest_digest, CLASSES


def checked_config(state, config):
    if state.get('config') != config:
        raise ValueError('Evaluation config differs from checkpoint config')
    if config['data']['factory'] != 'src.data:build_loaders':
        raise ValueError('Evaluator requires the main CIFAR-10 data factory')
    manifest = load_manifest(config['data']['split_seed'], config['data'].get('splits_dir', 'results/splits'))
    if state.get('manifest_sha256') != manifest_digest(manifest):
        raise ValueError('Checkpoint manifest fingerprint is missing or changed')
    return state['config']


def protocol_id(config):
    """Group identical recipes while allowing train/split seeds to vary."""
    recipe = copy.deepcopy(config)
    recipe.pop('seed', None)
    for key in ('split_seed', 'root', 'download', 'splits_dir', 'ratio'):
        recipe['data'].pop(key, None)
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Hàm tính metric
# ---------------------------------------------------------------------------

def compute_accuracy(predictions, labels):
    """Tính top-1 accuracy từ danh sách nhãn dự đoán và nhãn thật.

    Đầu vào:
        predictions: list/tensor nhãn dự đoán (int 0–9).
        labels:      list/tensor nhãn thật (int 0–9).
    Trả về:
        dict gồm accuracy (0–1), correct, total.

    Dùng ví dụ biết đáp án để kiểm tra trước khi dùng trên dữ liệu thật.
    """
    preds = torch.as_tensor(predictions)
    lbls  = torch.as_tensor(labels)
    if preds.shape != lbls.shape:
        raise ValueError(f'Kích thước không khớp: predictions={preds.shape}, labels={lbls.shape}')
    if preds.ndim != 1:
        raise ValueError('Expected one-dimensional class labels, not logits/probabilities')
    correct = int((preds == lbls).sum().item())
    total   = int(lbls.numel())
    if total == 0:
        raise ValueError('Danh sách rỗng, không tính được accuracy')
    return {'accuracy': correct / total, 'correct': correct, 'total': total}


def evaluate_checkpoint(checkpoint_path, config, split='val', device='cpu'):
    """Đánh giá một checkpoint trên tập val hoặc test.

    Đầu vào:
        checkpoint_path: đường dẫn đến best.pt hoặc last.pt.
        config: dict YAML đã nạp.
        split: 'val' hoặc 'test'.
        device: 'cpu' hoặc 'cuda'.
    Trả về:
        dict gồm accuracy, predictions (list int), probabilities (list list float),
              labels (list int), class_names.

    Quy trình:
        1. Nạp checkpoint và trích xuất trọng số.
        2. Tạo model theo factory trong config.
        3. Tạo loader theo split.
        4. Chạy suy luận, thu thập nhãn/xác suất.
    """
    import importlib

    # Nạp checkpoint; weights_only=True tránh thực thi code tùy ý khi nạp.
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checked_config(state, config)

    # Tạo model và nạp trọng số từ checkpoint.
    module_path, func_name = config['model']['factory'].split(':')
    build_fn = getattr(importlib.import_module(module_path), func_name)
    model = build_fn(config)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()

    # Tạo loader.
    if split == 'val':
        manifest = load_manifest(
            config['data']['split_seed'],
            splits_dir=config['data'].get('splits_dir', 'results/splits'))
        val_indices = manifest['val_indices']
        base = datasets.CIFAR10(root=config['data'].get('root', 'data'),
                                train=True, transform=make_eval_transform(),
                                download=config['data'].get('download', False))
        validate_manifest(manifest, base.targets, config['data']['split_seed'])
        from torch.utils.data import DataLoader, Subset
        loader = DataLoader(Subset(base, val_indices),
                            batch_size=config['data'].get('batch_size', 128),
                            shuffle=False, num_workers=0)
    elif split == 'test':
        loader = build_test_loader(config)
    else:
        raise ValueError(f"split phải là 'val' hoặc 'test', nhận được: {split!r}")

    # Chạy suy luận.
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)                        # [B, C]
            probs  = nn.functional.softmax(logits, dim=1) # [B, C]
            preds  = logits.argmax(dim=1)                 # [B]
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    metrics = compute_accuracy(all_preds, all_labels)
    return {
        **metrics,
        'predictions':  all_preds,
        'probabilities': all_probs,
        'labels':       all_labels,
        'class_names':  list(CLASSES),
    }


# ---------------------------------------------------------------------------
# Ghi kết quả vào runs.csv
# ---------------------------------------------------------------------------

RUNS_CSV_FIELDNAMES = [
    'run_id', 'model', 'ratio', 'split_seed', 'train_seed', 'protocol_id',
    'source_sha256', 'git_status', 'manifest_sha256', 'checkpoint_sha256',
    'params', 'config_path', 'commit', 'checkpoint_path',
    'split', 'accuracy', 'correct', 'total',
    'best_val_accuracy', 'best_epoch', 'global_step', 'checkpoint_step', 'checkpoint_epoch',
    'train_seconds_total',
    'train_seconds_per_epoch',  # thời gian train TB mỗi epoch (giây)
    'notes', 'evaluated_utc',
]


def append_run(result_row, csv_path='results/runs.csv'):
    """Ghi thêm một dòng vào results/runs.csv.

    Đầu vào:
        result_row: dict với các key từ RUNS_CSV_FIELDNAMES.
        csv_path: đường dẫn file CSV.
    Trả về:
        Path của file đã ghi.

    File được tạo mới với header nếu chưa tồn tại; các dòng mới được thêm vào cuối.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != RUNS_CSV_FIELDNAMES:
                raise ValueError('CSV schema mismatch; use a new output file')
            identity = ('model', 'ratio', 'split_seed', 'train_seed', 'protocol_id', 'split')
            for existing in reader:
                if all(str(existing[k]) == str(result_row[k]) for k in identity):
                    raise ValueError('Duplicate seed evaluation; use a separate CSV for reruns')

    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_FIELDNAMES, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        # Điền giá trị mặc định cho các cột thiếu.
        row = {k: result_row.get(k, '') for k in RUNS_CSV_FIELDNAMES}
        writer.writerow(row)

    return path


def build_run_row(config_path, checkpoint_path, eval_result, split, notes=''):
    """Tổng hợp một dòng metadata cho runs.csv từ checkpoint và kết quả đánh giá.

    Đầu vào:
        config_path:     đường dẫn file YAML.
        checkpoint_path: đường dẫn best.pt/last.pt.
        eval_result:     dict từ evaluate_checkpoint().
        split:           'val' hoặc 'test'.
        notes:           ghi chú tùy ý (ví dụ 'seed2', 'rerun').
    Trả về:
        dict tương thích với RUNS_CSV_FIELDNAMES.
    """
    cfg_path  = Path(config_path)
    ckpt_path = Path(checkpoint_path)
    config    = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

    # Đọc metadata từ run_dir (thư mục chứa checkpoint).
    run_dir  = ckpt_path.parent
    meta_path = run_dir / 'metadata.json'
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))

    # Đọc lịch sử từ checkpoint để lấy best val, epoch và thời gian train.
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checked_config(state, config)
    if not meta or meta.get('config') != config:
        raise ValueError('Missing or inconsistent training metadata.json')
    # Training cost must include epochs after the best checkpoint, if available.
    last_path = run_dir / 'last.pt'
    final_state = torch.load(last_path, map_location='cpu', weights_only=True) if last_path.exists() else state
    checked_config(final_state, config)
    if final_state['global_step'] < state['global_step']:
        raise ValueError('last.pt is older than the evaluated checkpoint')
    history = final_state.get('history', [])
    avg_train_sec = (sum(r['train_seconds'] for r in history) / len(history)
                     if history else '')

    # Lấy commit từ metadata hoặc git.
    commit = meta['commit']

    return {
        'run_id':          run_dir.name,
        'model':           config.get('model', {}).get('factory', ''),
        'ratio':           f"{float(config['data']['ratio']):.2f}",
        'split_seed':      config.get('data', {}).get('split_seed', ''),
        'train_seed':      config.get('seed', ''),
        'protocol_id':     hashlib.sha256((protocol_id(config) + meta['source_sha256']).encode()).hexdigest(),
        'source_sha256':   meta['source_sha256'],
        'git_status':      meta['git_status'],
        'manifest_sha256': state['manifest_sha256'],
        'checkpoint_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        'params':          meta.get('parameters', ''),
        'config_path':     str(cfg_path),
        'commit':          commit,
        'checkpoint_path': str(ckpt_path),
        'split':           split,
        'accuracy':        eval_result['accuracy'],
        'correct':         eval_result['correct'],
        'total':           eval_result['total'],
        'best_val_accuracy': state.get('best_val_accuracy', ''),
        'best_epoch':      state.get('best_epoch', ''),
        'global_step':     final_state.get('global_step', ''),
        'checkpoint_step': state.get('global_step', ''),
        'checkpoint_epoch': state.get('epoch', state.get('best_epoch', '')),
        'train_seconds_total': sum(r['train_seconds'] for r in history),
        'train_seconds_per_epoch': avg_train_sec,
        'notes':           notes,
        'evaluated_utc':   datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Kiểm tra metric bằng ví dụ biết đáp án
# ---------------------------------------------------------------------------

def _verify_metric():
    """Kiểm tra compute_accuracy bằng ví dụ biết đáp án trước khi dùng trên dữ liệu thật.

    Chạy: python -m src.evaluate --verify

    Nếu hàm này đạt thì metric được tính đúng theo định nghĩa top-1 accuracy.
    """
    # Ví dụ 1: 3/5 đúng → accuracy = 0.6
    preds  = [0, 1, 2, 3, 4]
    labels = [0, 1, 2, 9, 9]   # vị trí 3,4 sai
    result = compute_accuracy(preds, labels)
    assert result['correct'] == 3,   f"Kỳ vọng correct=3, nhận {result['correct']}"
    assert result['total']   == 5,   f"Kỳ vọng total=5, nhận {result['total']}"
    assert abs(result['accuracy'] - 0.6) < 1e-9, f"Kỳ vọng accuracy=0.6, nhận {result['accuracy']}"

    # Ví dụ 2: tất cả đúng → accuracy = 1.0
    result2 = compute_accuracy([7, 3, 5], [7, 3, 5])
    assert result2['accuracy'] == 1.0, f"Kỳ vọng 1.0, nhận {result2['accuracy']}"

    # Ví dụ 3: tất cả sai → accuracy = 0.0
    result3 = compute_accuracy([0, 0, 0], [1, 2, 3])
    assert result3['accuracy'] == 0.0, f"Kỳ vọng 0.0, nhận {result3['accuracy']}"

    # Ví dụ 4: 1 mẫu
    result4 = compute_accuracy([5], [5])
    assert result4['accuracy'] == 1.0 and result4['total'] == 1

    # Ví dụ 5: tensor thay vì list (tương thích với đầu ra của mô hình)
    result5 = compute_accuracy(torch.tensor([1, 2]), torch.tensor([1, 0]))
    assert result5['correct'] == 1

    print('[OK] compute_accuracy kiểm tra bằng 5 ví dụ biết đáp án — tất cả đạt.')
    return True


# ---------------------------------------------------------------------------
# Điểm vào dòng lệnh
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',     help='Đường dẫn file YAML cấu hình.')
    parser.add_argument('--checkpoint', help='Đường dẫn checkpoint (best.pt/last.pt).')
    parser.add_argument('--split',      choices=('val', 'test'), default='val',
                        help='Tập đánh giá (mặc định: val).')
    parser.add_argument('--device',     choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--out-csv',    default='results/runs.csv',
                        help='File CSV lưu kết quả.')
    parser.add_argument('--notes',      default='',
                        help='Ghi chú tùy ý cho dòng kết quả.')
    parser.add_argument('--predictions', help='JSON output for labels and probabilities')
    parser.add_argument('--verify',     action='store_true',
                        help='Chỉ chạy kiểm tra metric bằng ví dụ biết đáp án, không đánh giá.')
    args = parser.parse_args()

    if args.verify:
        _verify_metric()
    elif args.config and args.checkpoint:
        config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))

        print(f'Đánh giá {args.checkpoint} trên {args.split} ...')
        result = evaluate_checkpoint(args.checkpoint, config, split=args.split, device=args.device)
        print(f'  Accuracy: {result["accuracy"]:.4f} ({result["correct"]}/{result["total"]})')

        row = build_run_row(args.config, args.checkpoint, result, args.split, args.notes)
        path = append_run(row, csv_path=args.out_csv)
        prediction_path = Path(args.predictions) if args.predictions else Path(args.checkpoint).parent / f'predictions_{args.split}.json'
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps(result), encoding='utf-8')
        print(f'  Ghi vào: {path}')
    else:
        parser.print_help()

