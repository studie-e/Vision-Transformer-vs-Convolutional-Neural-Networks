# Người phụ trách: Minh (các lượt ResNet và Swin tuần 3).
# Chạy thêm seed ResNet hoặc thử Swin, kiểm tra khả năng đối chiếu các lượt chạy, rồi xuất CSV và gói bàn giao.
"""Chạy và kiểm tra các lượt ResNet/Swin của Minh trong tuần 3."""

import argparse
import ast
import csv
import json
import subprocess
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.package_team_week2 import package_run
from scripts.run_minh_week2 import completed_state, run_one
from scripts.summarize import save_summary_csv, summarize
from src.evaluate import append_run, build_run_row, evaluate_checkpoint, protocol_id


CONFIG_DIR = Path('configs/week3')
RUN_DIR = Path('runs/week3')
RESULT_DIR = Path('results/week3')


# Hàm config_paths trả danh sách YAML theo thứ tự seed rồi đến tỷ lệ.
# ResNet dùng hai cặp seed bổ sung, còn Swin tuần 3 dùng cặp đầu tiên ở 10% và 100%.
# Minh phụ trách chọn YAML ResNet hoặc Swin theo seed và tỷ lệ dữ liệu.
def config_paths(model):
    if model == 'resnet':
        return [CONFIG_DIR / f'resnet_{ratio}_seed{seed}.yaml'
                for seed in (43, 44) for ratio in (10, 50, 100)]
    return [CONFIG_DIR / f'swin_{ratio}_seed42.yaml' for ratio in (10, 100)]


# Hàm verify_training_source kiểm tra thay đổi mã giữa P0 và P1 trước khi gộp thống kê.
# Git xác nhận mã train sạch lúc chạy. Hai bản Python phải có cùng cây cú pháp.
# Công cụ vẽ hình và lời ghi người phụ trách ở đầu requirements không ảnh hưởng lượt train.
# Minh phụ trách đối chiếu phiên bản mã trước khi gộp kết quả.
def verify_training_source(baseline, repeat):
    for row in (baseline, repeat):
        if any(line[3:].startswith(('src/', 'requirements.txt')) for line in row['git_status'].splitlines()):
            raise ValueError(f"Uncommitted training source for {row['run_id']}")
    changed = subprocess.check_output(
        ['git', 'diff', '--no-renames', '--name-only', baseline['commit'], repeat['commit'],
         '--', 'src', 'requirements.txt'],
        text=True,
    ).splitlines()
    for path in changed:
        if path == 'src/visualize.py':
            continue
        try:
            before, after = (subprocess.check_output(['git', 'show', f"{row['commit']}:{path}"],
                                                    stderr=subprocess.DEVNULL)
                             for row in (baseline, repeat))
        except subprocess.CalledProcessError as error:
            # File chỉ có ở một commit là thay đổi mã, kể cả khi file trống.
            raise ValueError(f'Different training sources: {path}') from error
        if path.endswith('.py'):
            # AST giữ biểu thức và câu lệnh, bỏ chú thích cùng vị trí dòng.
            if ast.dump(ast.parse(before)) == ast.dump(ast.parse(after)):
                continue
        elif path == 'requirements.txt':
            label = '# Người phụ trách: Minh'.encode('utf-8')
            before_lines, after_lines = before.splitlines(), after.splitlines()
            if before_lines and before_lines[0].startswith(label):
                before_lines = before_lines[1:]
            if after_lines and after_lines[0].startswith(label):
                after_lines = after_lines[1:]
            if before_lines == after_lines:
                continue
        raise ValueError(f'Different training sources: {path}')


# Hàm collect_validation nạp lại best.pt bằng CUDA, đối chiếu trainer và đóng ZIP.
# CSV từng lượt giữ provenance gốc. Chỉ bảng thống kê ResNet gộp P0/P1 sau kiểm tra cấu hình và mã.
# Minh phụ trách kiểm tra, tổng hợp và đóng gói kết quả validation tuần 3.
def collect_validation(model, device):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    paths = config_paths(model)
    states = [completed_state(path, RUN_DIR / path.stem) for path in paths]
    destination = RESULT_DIR / f'{model}_runs.csv'
    temporary = destination.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    package_dir = Path('runs/handoff/week3') / model
    rows = []
    packages = []
    for path, state in zip(paths, states, strict=True):
        folder = RUN_DIR / path.stem
        config = state['config']
        result = evaluate_checkpoint(folder / 'best.pt', config, split='val', device=device)
        if result['total'] != 5000 or result['accuracy'] != state['best_val_accuracy']:
            raise ValueError(f'Validation does not match the training log: {folder}')
        events = EventAccumulator(str(folder / 'tensorboard'), size_guidance={'scalars': 0}).Reload()
        if [item.step for item in events.Scalars('val_accuracy')] != list(range(1, 101)):
            raise ValueError(f'TensorBoard does not contain epochs 1–100: {folder}')
        (folder / 'predictions_val.json').write_text(json.dumps(result), encoding='utf-8')
        row = build_run_row(folder / 'config.yaml', folder / 'best.pt', result, 'val',
                            'Week 3 ResNet P1' if model == 'resnet' else 'Week 3 exploratory Swin, one seed')
        rows.append(row)
        append_run(row, temporary)
        (RESULT_DIR / f'{folder.name}_metrics.csv').write_bytes((folder / 'metrics.csv').read_bytes())
        packages.append(package_run(folder, package_dir, expected_config_path=path))
        print(f"[VAL] {folder.name}: {result['correct']}/{result['total']}", flush=True)

    summary_rows = rows
    if model == 'resnet':
        # Dùng số liệu P0 đã xác minh. Kiểm tra cùng recipe và mã trước khi thống kê qua ba seed.
        with Path('results/team_week2/p0_runs.csv').open(encoding='utf-8', newline='') as stream:
            baselines = {row['ratio']: row for row in csv.DictReader(stream)
                         if row['model'] == 'src.models.resnet:build_model'}
        summary_rows = list(baselines.values())
        for row, state in zip(rows, states, strict=True):
            baseline = baselines[row['ratio']]
            base_path = Path(f"configs/week2/resnet_{int(float(row['ratio']) * 100)}.yaml")
            base_config = yaml.safe_load(base_path.read_text(encoding='utf-8'))
            if protocol_id(base_config) != protocol_id(state['config']):
                raise ValueError(f"Recipe changed since P0: {row['run_id']}")
            verify_training_source(baseline, row)
            # Chỉ quy về cùng ID trên bản sao dùng tính thống kê. CSV bằng chứng giữ ID gốc.
            summary_rows.append({**row, 'protocol_id': baseline['protocol_id'],
                                 'split_seed': str(row['split_seed']), 'train_seed': str(row['train_seed'])})

    summary = summarize(summary_rows)
    seeds = {int(row['n_seeds']) for row in summary}
    if seeds != ({3} if model == 'resnet' else {1}):
        raise ValueError(f'Unexpected number of seeds for {model}: {seeds}')
    save_summary_csv(summary, RESULT_DIR / f'{model}_summary.csv')
    temporary.replace(destination)
    manifest = json.dumps(packages, indent=2)
    (package_dir / 'handoff_manifest.json').write_text(manifest, encoding='utf-8')
    (RESULT_DIR / f'{model}_handoff_manifest.json').write_text(manifest, encoding='utf-8')


# Hàm main nhận lựa chọn train/collect. run_one kiểm tra và tiếp tục last.pt khi run bị ngắt.
# Minh phụ trách lệnh chạy thí nghiệm hoặc thu kết quả tuần 3.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'collect'), required=True)
    parser.add_argument('--model', choices=('resnet', 'swin'), default='resnet')
    args = parser.parse_args()
    if args.stage == 'train':
        for path in config_paths(args.model):
            run_one(path, RUN_DIR / path.stem, 'cuda', resume=True)
    else:
        collect_validation(args.model, 'cuda')


if __name__ == '__main__':
    main()
