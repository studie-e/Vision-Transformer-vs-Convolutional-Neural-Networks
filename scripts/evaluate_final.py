"""Preflight the selected checkpoint registry, then evaluate test with --execute."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.reproduce_week4 import OUTPUT, load_runs, write_csv, mean_std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=OUTPUT / 'checkpoint_registry.csv')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--out', type=Path, default=Path('results/final_test'))
    args = parser.parse_args()
    selected = [r for r in load_runs(args.registry) if r['scope'] == 'main']
    errors = []
    identities = [(r['architecture'], r['ratio'], r['split_seed'], r['train_seed']) for r in selected]
    if len(selected) != 18 or len(set(identities)) != 18:
        errors.append('Expected 18 distinct main runs')
    for row in selected:
        if row['audit_status'] != 'saved_log_consistent':
            errors.append(f"{row['run_id']}: saved evidence needs correction")
        checkpoint, config = Path(row['checkpoint_path']), Path(row['config_path'])
        if not checkpoint.is_file() or not config.is_file():
            errors.append(f"{row['run_id']}: checkpoint/config missing")
        elif hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row['checkpoint_sha256']:
            errors.append(f"{row['run_id']}: checkpoint SHA-256 differs")
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
        config = yaml.safe_load(Path(row['config_path']).read_text(encoding='utf-8'))
        result = evaluate_checkpoint(row['checkpoint_path'], config, split='test', device=args.device)
        results.append(build_run_row(row['config_path'], row['checkpoint_path'], result, 'test'))
    args.out.mkdir(parents=True)
    write_csv(args.out / 'runs.csv', results, RUNS_CSV_FIELDNAMES)
    summaries = []
    for model in sorted({r['model'] for r in results}):
        for ratio in ('0.10', '0.50', '1.00'):
            group = [r for r in results if r['model'] == model and r['ratio'] == ratio]
            mean, std = mean_std([r['accuracy'] for r in group])
            summaries.append(dict(model=model, ratio=ratio, split='test', n_seeds=len(group),
                                  mean_accuracy=mean, std_accuracy=std))
    write_csv(args.out / 'summary.csv', summaries, list(summaries[0]))
    (args.out / 'selected_registry.csv').write_bytes(args.registry.read_bytes())


if __name__ == '__main__':
    main()
