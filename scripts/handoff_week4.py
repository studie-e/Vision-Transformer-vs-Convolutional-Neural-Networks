# Minh phụ trách kiểm tra thư mục run, đóng gói checkpoint và kiểm tra khả năng tái lập.
"""Đối chiếu các run thật trên validation và đóng gói theo đường dẫn mà registry sử dụng."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import torch
import yaml
from torchvision.datasets import CIFAR10

from scripts.reproduce_week4 import check_history, load_runs, write_csv
from src.checkpoint import load_checkpoint
from src.data import load_manifest
from src.evaluate import RUNS_CSV_FIELDNAMES, evaluate_checkpoint


# Minh kiểm tra file checkpoint, lịch sử đầy đủ và số ảnh đúng khi nạp lại trên GPU.
def verify(row, device):
    checkpoint = Path(row['checkpoint_path'].replace('\\', '/'))
    folder = checkpoint.parent
    if checkpoint.name != 'best.pt' or folder.name != row['run_id']:
        raise ValueError(f'Unexpected run path: {checkpoint}')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    best = load_checkpoint(checkpoint)
    last = load_checkpoint(folder / 'last.pt')
    history = load_runs(folder / 'metrics.csv')
    errors = check_history(history, row, int(config['training']['epochs']))
    for saved, logged in zip(last['history'], history, strict=True):
        if (int(logged['epoch']) != int(saved['epoch'])
                or int(logged['global_step']) != int(saved['global_step'])
                or any(not math.isclose(float(logged[key]), float(value), rel_tol=0, abs_tol=1e-12)
                       for key, value in saved.items() if key not in ('epoch', 'global_step'))):
            errors.append('metrics.csv differs from last.pt')
            break
    if (digest != row['checkpoint_sha256'] or meta['config'] != config or last['config'] != config
            or best['config'] != config or meta['manifest_sha256'] != row['manifest_sha256']
            or best['manifest_sha256'] != row['manifest_sha256'] or meta['commit'] != row['commit']
            or last['global_step'] != int(row['global_step']) or last['best_epoch'] != int(row['best_epoch'])
            or best['epoch'] != int(row['best_epoch']) or best['best_epoch'] != int(row['best_epoch'])):
        errors.append('source, configuration or checkpoint identity differs')
    if errors:
        raise ValueError(f'{folder}: {errors}')
    result = evaluate_checkpoint(checkpoint, config, split='val', device=device)
    if result['total'] != 5000 or result['correct'] != int(row['correct']):
        raise ValueError(f'{folder}: validation counts differ')
    return dict(run_id=folder.name, checkpoint=checkpoint.as_posix(), sha256=digest,
                correct=result['correct'], total=result['total'], best_epoch=last['best_epoch'],
                manifest_sha256=row['manifest_sha256'], commit=meta['commit'])


# Minh đối chiếu trọng số và chỉ số học khi tái lập lượt ResNet đại diện từ seed gốc.
def compare_repeat(reference, repeat):
    for file in ('best.pt', 'last.pt'):
        original = load_checkpoint(reference / file)
        new = load_checkpoint(repeat / file)
        for key in ('config', 'manifest_sha256', 'best_epoch', 'epoch', 'global_step', 'best_val_accuracy'):
            if original[key] != new[key]:
                raise ValueError(f'{file}: repeat differs at {key}')
        for name, weight in original['model'].items():
            if not torch.equal(weight, new['model'][name]):
                raise ValueError(f'{file}: different tensor {name}')
        for a, b in zip(original['history'], new['history'], strict=True):
            if any(a[k] != b[k] for k in ('epoch','global_step','lr','train_loss','val_loss','train_accuracy','val_accuracy')):
                raise ValueError(f'{file}: different learning history at epoch {a["epoch"]}')
    return dict(reference=reference.as_posix(), repeat=repeat.as_posix(), epochs=new['epoch'],
                best_epoch=new['best_epoch'], best_validation=new['best_val_accuracy'],
                model_tensors_equal=True, epoch_metrics_equal=True)


# Minh lấy ảnh validation đầu tiên theo manifest để người khác tái tạo demo từ cùng đầu vào.
def export_demo_image(output, data_root=Path('data'), splits_dir=Path('results/splits')):
    indices = load_manifest(2026, splits_dir=splits_dir)['val_indices']
    image, label = CIFAR10(str(data_root), train=True, download=False)[indices[0]]
    image.save(output / 'demo_input.png')
    return dict(split='validation', split_seed=2026, index=indices[0], true_label=label)


# Minh xuất bảng 18 lượt baseline đã xác minh để công cụ tuần 3 của Hiền kiểm tra đủ 18/18 run.
def export_baseline_18_csv(rows, destination):
    baseline = [
        dict(r) for r in rows
        if r['model'] in ('src.models.resnet:build_model', 'src.models.vit:build_model')
        and not r['run_id'].startswith(('ablation_', 'swin_'))
    ]
    if len(baseline) == 18:
        p0_protocols = {r['model']: r['protocol_id'] for r in baseline if int(r['train_seed']) == 42}
        for r in baseline:
            r['protocol_id'] = p0_protocols[r['model']]
            r['checkpoint_path'] = r['checkpoint_path'].replace('\\', '/')
            r['config_path'] = r['config_path'].replace('\\', '/')
        write_csv(destination, baseline, RUNS_CSV_FIELDNAMES)
        return destination.as_posix()
    return None


# Minh tạo ZIP nguyên cây thư mục run sau khi tất cả checkpoint của đợt bàn giao qua kiểm tra.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, nargs='+', action='extend', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--repeat', type=Path)
    parser.add_argument('--reference', type=Path, default=Path('runs/week2/resnet_d10_s42'))
    parser.add_argument('--archive', type=Path, default=None)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--splits-dir', type=Path, default=Path('results/splits'))
    args = parser.parse_args()
    rows = [r for file in args.records for r in load_runs(file)]
    if not rows or len(rows) != len({r['run_id'] for r in rows}):
        raise ValueError('Missing or duplicated runs')
    args.out.mkdir(parents=True, exist_ok=True)
    records = [verify(row, args.device) for row in rows]
    write_csv(args.out / 'verified.csv', records, list(records[0]))
    baseline_csv = export_baseline_18_csv(rows, args.out / 'baseline_18_val_runs.csv')
    summary = dict(validation_runs=len(records), device=args.device,
                   demo=export_demo_image(args.out, args.data_root, args.splits_dir))
    if baseline_csv:
        summary['baseline_18_csv'] = baseline_csv
    if args.repeat:
        summary['reproduction'] = compare_repeat(args.reference, args.repeat)
    archive = args.archive or (Path('runs/handoff/week4') / (args.out.name + '.zip'))
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for record in records:
            folder = Path(record['checkpoint']).parent
            for p in sorted(folder.rglob('*')):
                if p.is_file():
                    bundle.write(p, p.as_posix())
        bundle.write(args.out / 'verified.csv', (args.out / 'verified.csv').as_posix())
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip():
            raise ValueError('Corrupt handoff archive')
    temporary.replace(archive)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    checksum_file = archive.with_suffix('.sha256.txt')
    checksum_file.write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    summary['archive'] = dict(path=archive.as_posix(), bytes=archive.stat().st_size,
                              sha256=digest, checksum_file=checksum_file.as_posix())
    (args.out / 'handoff.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
