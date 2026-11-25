# Hiền phụ trách mã, danh sách checkpoint và kết quả test cuối. Minh hỗ trợ vận hành GPU.
"""Preflight the selected checkpoint registry, then evaluate test with --execute."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

from scripts.reproduce_week4 import OUTPUT, load_runs, write_csv, mean_std


# Hiền đối chiếu tất cả lựa chọn với YAML đã chốt, manifest, best epoch và SHA-256 trên validation.
def preflight(selected, root=Path('.')):
    import yaml
    from src.checkpoint import load_checkpoint
    from src.evaluate import checked_config

    expected = {(arch, ratio, seed + 1984, seed) for arch in ('vit', 'resnet')
                for ratio in ('0.10', '0.50', '1.00') for seed in (42, 43, 44)}
    identities = [(r['architecture'], f"{float(r['ratio']):.2f}", int(r['split_seed']), int(r['train_seed']))
                  for r in selected]
    errors = []
    if len(identities) != len(expected) or set(identities) != expected:
        errors.append('Expected exactly the 18 locked architecture/ratio/seed combinations')
    for row, identity in zip(selected, identities, strict=True):
        if row['audit_status'] != 'saved_log_consistent' or identity not in expected:
            errors.append(f"{row['run_id']}: invalid validation evidence or identity")
            continue
        arch, ratio, _, seed = identity
        pct = round(float(ratio) * 100)
        recipe = root / (Path(f'configs/week2/{arch}_{pct}.yaml') if seed == 42 else
                         Path(f'configs/week3/{arch}_{pct}_seed{seed}.yaml'))
        checkpoint = root / Path(row['checkpoint_path'].replace('\\', '/'))
        config_path = root / Path(row['config_path'].replace('\\', '/'))
        last_path = checkpoint.with_name('last.pt')
        meta_path = checkpoint.with_name('metadata.json')
        if not all(p.is_file() for p in (checkpoint, config_path, last_path, meta_path, recipe)):
            errors.append(f"{row['run_id']}: checkpoint/config/metadata missing")
            continue
        if (checkpoint.name != 'best.pt' or config_path != checkpoint.with_name('config.yaml')
                or checkpoint.parent.name != row['run_id']
                or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row['checkpoint_sha256']):
            errors.append(f"{row['run_id']}: checkpoint identity or SHA-256 differs")
            continue
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if config != yaml.safe_load(recipe.read_text(encoding='utf-8')) or meta.get('config') != config:
            errors.append(f"{row['run_id']}: config differs from locked recipe")
            continue
        state = load_checkpoint(checkpoint)
        final = load_checkpoint(last_path)
        try:
            checked_config(state, config)
            checked_config(final, config)
        except ValueError as exc:
            errors.append(f"{row['run_id']}: {exc}")
            continue
        if (row.get('manifest_sha256') and state.get('manifest_sha256') != row['manifest_sha256']
                or meta.get('manifest_sha256') != state.get('manifest_sha256')):
            errors.append(f"{row['run_id']}: manifest SHA-256 differs from locked registry")
            continue
        if (final['epoch'] != config['training']['epochs'] or
                [h['epoch'] for h in final['history']] != list(range(1, final['epoch'] + 1))):
            errors.append(f"{row['run_id']}: incomplete training history")
            continue
        winner = max(final['history'], key=lambda h: h['val_accuracy'])
        if (state['epoch'] != state['best_epoch'] or winner['epoch'] != state['epoch']
                or winner['val_accuracy'] != state['best_val_accuracy']
                or (row.get('best_epoch') and state['best_epoch'] != int(row['best_epoch']))
                or (row.get('val_accuracy') and not math.isclose(
                    state['best_val_accuracy'], float(row['val_accuracy']), rel_tol=0, abs_tol=1e-12))):
            errors.append(f"{row['run_id']}: checkpoint does not match validation selection")
    return errors


# Hiền phụ trách kiểm tra danh sách đã khóa, đánh giá test và tổng hợp.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=OUTPUT / 'checkpoint_registry.csv')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--out', type=Path, default=Path('results/final_test'))
    args = parser.parse_args()
    selected = [r for r in load_runs(args.registry) if r['scope'] == 'main']
    errors = preflight(selected)
    print(json.dumps({'selected': len(selected), 'errors': errors}, indent=2))
    if not args.execute:
        return
    if errors:
        raise ValueError('Resolve preflight errors before evaluating the selected checkpoints')
    if args.out.exists():
        raise FileExistsError('Choose a new output directory to retain previous test evidence')
    import yaml
    from src.evaluate import evaluate_checkpoint, build_run_row, RUNS_CSV_FIELDNAMES
    results = []
    for row in selected:
        config_path = Path(row['config_path'].replace('\\', '/'))
        checkpoint = Path(row['checkpoint_path'].replace('\\', '/'))
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        result = evaluate_checkpoint(checkpoint, config, split='test', device=args.device)
        if result['total'] != 10000:
            raise ValueError(f"Expected 10000 test images: {row['run_id']}")
        results.append(build_run_row(config_path, checkpoint, result, 'test'))
    summaries = []
    for model in sorted({r['model'] for r in results}):
        for ratio in ('0.10', '0.50', '1.00'):
            group = [r for r in results if r['model'] == model and r['ratio'] == ratio]
            if len(group) != 3:
                raise ValueError(f'Expected 3 seeds for {model} at ratio {ratio}, got {len(group)}')
            mean, std = mean_std([r['accuracy'] for r in group])
            summaries.append(dict(model=model, ratio=ratio, split='test', n_seeds=len(group),
                                  mean_accuracy=mean, std_accuracy=std))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = args.out.parent / f'.{args.out.name}.tmp'
    shutil.rmtree(stage, ignore_errors=True)
    try:
        stage.mkdir()
        write_csv(stage / 'runs.csv', results, RUNS_CSV_FIELDNAMES)
        write_csv(stage / 'summary.csv', summaries, list(summaries[0]))
        write_csv(stage / 'selected_registry.csv', selected, list(selected[0]))
        stage.replace(args.out)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
