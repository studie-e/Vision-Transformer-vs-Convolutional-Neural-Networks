# Người phụ trách: Hiền (kiểm tra bằng chứng validation, xuất bảng và biểu đồ tuần 4); Minh cung cấp kết quả ResNet/Swin.
"""Audit saved evidence and regenerate week-four validation tables and figures."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

def load_runs(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return (statistics.mean(values) if values else float('nan'),
            statistics.stdev(values) if len(values) > 1 else float('nan'))

OUTPUT = Path('results/hien_week4')


def expected_recipe_path(scope, architecture, pct, seed):
    if scope == 'main' and seed == 42:
        return Path(f'configs/week2/{architecture}_{pct}.yaml')
    if scope == 'main':
        return Path(f'configs/week3/{architecture}_{pct}_seed{seed}.yaml')
    if scope == 'ablation':
        return Path(f'configs/week3/ablation_{architecture}_{pct}_seed{seed}.yaml')
    return Path(f'configs/week3/{architecture}_{pct}_seed{seed}.yaml')


def check_history(history, row, epochs=100):
    errors = []
    n = round(45000 * float(row['ratio']))
    steps = math.ceil(n / 128)
    if [int(h['epoch']) for h in history] != list(range(1, epochs + 1)):
        errors.append('epoch_sequence')
    for h in history:
        epoch = int(h['epoch'])
        for field in ('train_accuracy', 'val_accuracy'):
            accuracy = float(h[field])
            count = n if field == 'train_accuracy' else 5000
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                errors.append(f'epoch_{epoch}:{field}_range')
            elif abs(accuracy * count - round(accuracy * count)) > 1e-6:
                errors.append(f'epoch_{epoch}:{field}_noninteger_correct')
        for field in ('train_loss', 'val_loss', 'train_seconds', 'epoch_seconds', 'lr', 'peak_cuda_mb'):
            value = float(h[field])
            if not math.isfinite(value) or value < 0:
                errors.append(f'epoch_{epoch}:{field}_invalid')
        if float(h['epoch_seconds']) < float(h['train_seconds']):
            errors.append(f'epoch_{epoch}:epoch_seconds_less_than_train')
        if int(h['global_step']) != epoch * steps:
            errors.append(f'epoch_{epoch}:global_step')
    best = max(history, key=lambda h: float(h['val_accuracy']))
    if (int(best['epoch']) != int(row['best_epoch'])
            or abs(float(best['val_accuracy']) - float(row['accuracy'])) > 1e-12):
        errors.append('best_checkpoint_record')
    if (int(row['total']) != 5000 or float(row['accuracy']) != int(row['correct']) / 5000
            or row['split'] != 'val'):
        errors.append('evaluation_counts_or_split')
    if int(row['global_step']) != epochs * steps:
        errors.append('total_steps')
    seconds = sum(float(h['train_seconds']) for h in history)
    if not math.isclose(seconds, float(row['train_seconds_total']), rel_tol=1e-6):
        errors.append('total_time')
    return errors


def collect(root=Path('.'), require_swin=False):
    sources = [
        ('results/team_week2/p0_runs.csv', 'main', 'p0'),
        ('results/week3/resnet_runs.csv', 'main', 'resnet'),
        ('results/week3/vit_runs.csv', 'main', 'vit'),
        ('results/week3/ablation_runs.csv', 'ablation', 'ablation'),
        ('results/week3/swin_smoke_runs.csv', 'smoke', 'swin'),
    ]
    swin_csv = root / 'results/week3/swin_runs.csv'
    if require_swin and not swin_csv.is_file():
        raise FileNotFoundError(f'Missing Swin validation records: {swin_csv}')
    if swin_csv.is_file():
        sources.append(('results/week3/swin_runs.csv', 'extension', 'swin_full'))
    records, issues, registry = [], [], []
    seen = set()
    for filename, scope, group in sources:
        for row in load_runs(root / filename):
            architecture = row['model'].split(':')[0].split('.')[-1]
            pct = round(float(row['ratio']) * 100)
            seed = int(row['train_seed'])
            split_seed = int(row['split_seed'])
            identity = scope, architecture, pct, seed
            if identity in seen:
                raise ValueError(f'Duplicate run {identity}')
            seen.add(identity)
            if group == 'p0':
                rel_log = Path(f'results/team_week2/{architecture}_{pct}_metrics.csv')
            elif group == 'swin':
                rel_log = Path('results/week3/swin_smoke_metrics.csv')
            else:
                rel_log = Path(f'results/week3/{row["run_id"]}_metrics.csv')
            log = root / rel_log
            history = load_runs(log)
            errors = check_history(history, row, epochs=1 if scope == 'smoke' else 100)
            if (split_seed, seed) not in {(2026, 42), (2027, 43), (2028, 44)}:
                errors.append('seed_pair')
            if (row.get('checkpoint_epoch', '') != '' and int(row['checkpoint_epoch']) != int(row['best_epoch'])
                    or row.get('best_val_accuracy', '') != ''
                    and abs(float(row['best_val_accuracy']) - float(row['accuracy'])) > 1e-12):
                errors.append('checkpoint_metadata')
            manifest = json.loads((root / f"results/splits/split_seed{row['split_seed']}_manifest.json").read_text(encoding='utf-8'))
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            if row['manifest_sha256'] != digest:
                errors.append('manifest_hash')
            params = 268042 if scope == 'ablation' else {'resnet': 272474, 'vit': 274282, 'swin': 261824}[architecture]
            if int(row['params']) != params:
                errors.append('parameter_count')
            ckpt_rel = Path(row['checkpoint_path'].replace('\\', '/'))
            cfg_rel = Path(row['config_path'].replace('\\', '/'))
            ckpt_path = root / ckpt_rel
            cfg_path = root / cfg_rel
            ckpt_present = ckpt_path.is_file()
            cfg_present = cfg_path.is_file()
            ckpt_sha_verified = False
            if ckpt_present:
                if hashlib.sha256(ckpt_path.read_bytes()).hexdigest() != row['checkpoint_sha256']:
                    errors.append('checkpoint_sha256')
                else:
                    ckpt_sha_verified = True
            recipe_path = root / expected_recipe_path(scope, architecture, pct, seed)
            if cfg_present and recipe_path.is_file():
                if yaml.safe_load(cfg_path.read_text(encoding='utf-8')) != yaml.safe_load(recipe_path.read_text(encoding='utf-8')):
                    errors.append('config_recipe')
            for error in errors:
                issues.append(dict(run_id=row['run_id'], evidence=rel_log.as_posix(), issue=error))
            status = 'invalid_saved_log' if errors else 'saved_log_consistent'
            record = dict(row, architecture=architecture, scope=scope, audit_status=status,
                          metrics_path=rel_log.as_posix(),
                          metrics_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
            records.append(record)
            registry.append(dict(run_id=row['run_id'], scope=scope, architecture=architecture,
                ratio=row['ratio'], split_seed=row['split_seed'], train_seed=row['train_seed'],
                manifest_sha256=row['manifest_sha256'], best_epoch=row['best_epoch'], val_accuracy=row['accuracy'],
                checkpoint_path=ckpt_rel.as_posix(), checkpoint_sha256=row['checkpoint_sha256'],
                checkpoint_sha256_verified=ckpt_sha_verified,
                config_path=cfg_rel.as_posix(), audit_status=status,
                checkpoint_present=ckpt_present,
                config_present=cfg_present,
                selected_by='best validation accuracy; earliest tied epoch'))
    return records, issues, registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true',
                        help='Require full 18 main, 3 ablation, 6 Swin runs with zero issues and verified local checkpoints.')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, issues, registry = collect(require_swin=args.strict)
    write_csv(OUTPUT / 'per_seed.csv', records, list(records[0]))
    write_csv(OUTPUT / 'audit_issues.csv', issues, ['run_id', 'evidence', 'issue'])
    write_csv(OUTPUT / 'checkpoint_registry.csv', registry, list(registry[0]))
    summaries = []
    for scope, architecture, ratio in sorted({(r['scope'], r['architecture'], r['ratio']) for r in records}):
        group = sorted((r for r in records if (r['scope'], r['architecture'], r['ratio']) ==
                        (scope, architecture, ratio)), key=lambda r: int(r['train_seed']))
        good = [r for r in group if r['audit_status'] == 'saved_log_consistent']
        reported_mean, reported_std = mean_std([float(r['accuracy']) for r in group])
        mean, std = mean_std([float(r['accuracy']) for r in good])
        summaries.append(dict(scope=scope, architecture=architecture, ratio=ratio,
            reported_seeds=len(group), consistent_seeds=len(good),
            consistent_mean=mean, consistent_std=std,
            reported_mean=reported_mean, reported_std=reported_std,
            params=group[0]['params'], train_seconds=sum(float(r['train_seconds_total']) for r in good)))
    write_csv(OUTPUT / 'summary.csv', summaries, list(summaries[0]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ('vit', 'resnet'):
        group = [s for s in summaries if s['scope'] == 'main' and s['architecture'] == model]
        ax.errorbar([float(s['ratio']) * 100 for s in group],
                    [s['consistent_mean'] * 100 for s in group],
                    yerr=[s['consistent_std'] * 100 if s['consistent_seeds'] > 1 else 0 for s in group],
                    marker='o', capsize=4, label=f'{model}, n=' + '/'.join(str(s['consistent_seeds']) for s in group))
    ax.set(xlabel='Training data (%)', ylabel='Validation accuracy (%)', xticks=[10, 50, 100])
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'validation_scaling.png', dpi=160)
    plt.close(fig)
    plan = {
        'scope': 'validation evidence audit',
        'main_records': sum(r['scope'] == 'main' for r in records),
        'ablation_records': sum(r['scope'] == 'ablation' for r in records),
        'extension_records': sum(r['scope'] == 'extension' for r in records),
        'smoke_records': sum(r['scope'] == 'smoke' for r in records),
        'total_records': len(records),
        'consistent_records': sum(r['audit_status'] == 'saved_log_consistent' for r in records),
        'invalid_run_ids': sorted({r['run_id'] for r in records if r['audit_status'] != 'saved_log_consistent'}),
        'local_main_checkpoints': sum(r['scope'] == 'main' and r['checkpoint_present'] for r in registry),
        'local_all_checkpoints': sum(bool(r['checkpoint_present']) for r in registry),
        'verified_checkpoint_hashes': sum(bool(r['checkpoint_sha256_verified']) for r in registry),
        'test_evaluated': Path('results/final_test/summary.csv').is_file(),
        'sources': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(
            {'results/team_week2/p0_runs.csv', 'results/week3/resnet_runs.csv',
             'results/week3/vit_runs.csv', 'results/week3/ablation_runs.csv', 'results/week3/swin_smoke_runs.csv'})},
    }
    extension = Path('results/week3/swin_runs.csv')
    if extension.is_file():
        plan['sources'][extension.as_posix()] = hashlib.sha256(extension.read_bytes()).hexdigest()
    (OUTPUT / 'audit.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2))
    if args.strict:
        if (plan['main_records'] != 18 or plan['ablation_records'] != 3 or plan['extension_records'] != 6
                or plan['invalid_run_ids'] or plan['verified_checkpoint_hashes'] != plan['total_records']):
            raise ValueError(f'Strict week-4 audit failed: {json.dumps(plan)}')


if __name__ == '__main__':
    main()
