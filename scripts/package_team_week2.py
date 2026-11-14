"""Đóng gói run P0 đã chạy đủ 100 epoch để bàn giao nội bộ cho Hiền."""

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import torch
import yaml

from src.data import load_manifest, manifest_digest


# Hàm package_run: kiểm tra bằng chứng của một run rồi nén toàn bộ thư mục thành ZIP.
# Nhận thư mục run và nơi xuất; trả metadata gói. Không gửi file ra dịch vụ bên ngoài.
def package_run(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    required = ('config.yaml', 'metadata.json', 'metrics.csv', 'best.pt', 'last.pt')
    # Đủ file chưa đủ để nghiệm thu: các bước dưới còn kiểm tra config, epoch và manifest.
    if any(not (run_dir / name).is_file() for name in required):
        raise FileNotFoundError(f'Missing handoff files: {run_dir}')
    config = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))
    metadata = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
    architecture = config['model']['factory'].split(':')[0].split('.')[-1]
    level = {'0.10': '10', '0.50': '50', '1.00': '100'}[f"{float(config['data']['ratio']):.2f}"]
    expected_path = Path(f'configs/week2/{architecture}_{level}.yaml')
    expected = yaml.safe_load(expected_path.read_text(encoding='utf-8'))
    if config != expected or metadata['config'] != config:
        raise ValueError(f'Run differs from the current team P0 config: {run_dir}')
    last = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=True)
    best = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    for state in (last, best):
        if state['config'] != config:
            raise ValueError('Checkpoint config differs from the run config')
        if state['manifest_sha256'] != manifest_digest(load_manifest(config['data']['split_seed'])):
            raise ValueError('Checkpoint manifest differs from the shared split')
    # batch cuối không bị bỏ nên số bước là ceil(số ảnh / batch_size) × 100.
    samples = {'10': 4500, '50': 22500, '100': 45000}[level]
    batch_size = config['data']['batch_size']
    steps = ((samples + batch_size - 1) // batch_size) * 100
    if last['epoch'] != 100 or len(last['history']) != 100 or last['global_step'] != steps:
        raise ValueError('Run has not completed the agreed 100-epoch budget')
    with (run_dir / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        history = list(csv.DictReader(stream))
    if [int(row['epoch']) for row in history] != list(range(1, 101)):
        raise ValueError('Metrics CSV is missing or duplicating epochs')
    # max giữ epoch sớm hơn khi hòa; đối chiếu best.pt với cả CSV và trạng thái cuối.
    winner = max(history, key=lambda row: float(row['val_accuracy']))
    if best['epoch'] != int(winner['epoch']) or best['epoch'] != last['best_epoch']:
        raise ValueError('Best checkpoint does not follow the agreed tie rule')
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'{run_dir.name}.zip'
    temporary = target.with_suffix('.zip.tmp')
    # Giữ tên thư mục run bên trong ZIP, kèm TensorBoard và các file phụ đã có.
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run_dir.rglob('*')):
            if path.is_file():
                archive.write(path, Path(run_dir.name) / path.relative_to(run_dir))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        if any(f'{run_dir.name}/{name}' not in archive.namelist() for name in required):
            raise ValueError('ZIP is missing a required handoff file')
    temporary.replace(target)
    # SHA-256 để người nhận kiểm tra file nén sau khi chép qua máy khác.
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.sha256.txt').write_text(f'{checksum}  {target.name}\n', encoding='utf-8')
    result = {'run_id': run_dir.name, 'architecture': architecture, 'ratio': config['data']['ratio'],
              'run_dir': run_dir.as_posix(), 'zip_path': target.as_posix(),
              'zip_bytes': target.stat().st_size, 'zip_sha256': checksum,
              'epochs': last['epoch'], 'best_epoch': best['epoch'],
              'best_val_accuracy': last['best_val_accuracy'], 'global_steps': steps}
    return result


# Hàm main: nhận nhiều --run, kiểm tra/nén lần lượt rồi lưu danh sách gói cho Hiền.
# Không tự thay đổi trạng thái run_matrix hoặc giả định Hiền đã tiếp nhận các file.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('runs/handoff/team_week2'))
    args = parser.parse_args()
    records = []
    for run in args.run:
        result = package_run(run, args.out)
        records.append(result)
        print(json.dumps(result), flush=True)
    (args.out / 'handoff_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
