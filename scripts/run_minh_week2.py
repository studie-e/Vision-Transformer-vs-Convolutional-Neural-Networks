"""Tuning ngang ngân sách, khóa recipe và thực hiện ResNet tuần 2 của Minh."""

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import yaml


# Hai recipe cho mỗi mô hình; cùng số ảnh/epoch/seed để so ngân sách thử cấu hình.
# Nguồn optimizer: https://docs.pytorch.org/docs/2.11/optim.html
RECIPES = {
    'sgd': {'epochs': 10, 'optimizer': 'sgd', 'lr': 0.05,
            'momentum': 0.9, 'weight_decay': 0.0005},
    'adamw': {'epochs': 10, 'optimizer': 'adamw', 'lr': 0.001, 'weight_decay': 0.05},
}
RATIOS = {'d10': '0.10', 'd50': '0.50', 'd100': '1.00'}


# Hàm make_tuning_configs: tạo bốn YAML thử cấu hình, hai cho ResNet và hai cho ViT.
# Nhận thư mục đích; trả list Path; từ chối ghi đè nếu file sẵn có khác cấu hình dự kiến.
def make_tuning_configs(out_dir=Path('configs/week2/tuning')):
    # Đọc kiến trúc ViT đã được Lan Anh bàn giao, không thay mã/kiến trúc của bạn.
    vit = yaml.safe_load(Path('configs/vit.yaml').read_text(encoding='utf-8'))['model']
    resnet = {'factory': 'src.models.resnet:build_model',
              'kwargs': {'width': 16, 'blocks_per_stage': 3, 'num_classes': 10}}
    # Dùng đúng pipeline/manifest chính, không dùng adapter pilot tuần 1.
    data = yaml.safe_load(Path('configs/data.yaml').read_text(encoding='utf-8'))['data']
    data.update({'split_seed': 2026, 'ratio': '0.50', 'batch_size': 128, 'download': False})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in (('resnet', resnet), ('vit', vit)):
        for recipe, training in RECIPES.items():
            config = {'seed': 42, 'model': copy.deepcopy(model),
                      'data': copy.deepcopy(data), 'training': copy.deepcopy(training)}
            path = out_dir / f'{name}_{recipe}.yaml'
            # Không thay config đang dùng bởi một run cũ khi gọi lại script.
            if path.exists() and yaml.safe_load(path.read_text(encoding='utf-8')) != config:
                raise FileExistsError(f'Config differs: {path}')
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            paths.append(path)
    return paths


# Hàm choose_recipe: chọn ứng viên có best validation accuracy cao nhất trong hai lượt đã đủ ngân sách.
# Nhận list metadata, trả dòng được chọn; hòa thì giữ recipe đứng trước (SGD theo thứ tự công bố).
def choose_recipe(rows):
    if len(rows) != 2 or {r['recipe'] for r in rows} != set(RECIPES):
        raise ValueError('Expected exactly one SGD and one AdamW candidate')
    # Không so run mới 2 epoch với run đã đủ 10 epoch rồi gọi đó là tuning công bằng.
    if any(r['completed_epochs'] != 10 for r in rows):
        raise ValueError('Each candidate must complete exactly 10 epochs')
    if any(not math.isfinite(r['best_val_accuracy']) or not 0 <= r['best_val_accuracy'] <= 1 for r in rows):
        raise ValueError('Invalid validation accuracy')
    # max giữ phần tử đầu tiên nếu key bằng nhau; không sử dụng test accuracy ở đây.
    return max(rows, key=lambda row: row['best_val_accuracy'])


# Hàm completed_state: kiểm tra run đã đủ epoch, cấu hình/manifest còn khớp trước khi nhận kết quả.
# Nhận đường dẫn config/run; trả checkpoint cuối, không chạy lại huấn luyện.
def completed_state(config_path, run_dir):
    # Tái sử dụng kiểm tra của Hiền: config trong checkpoint và hash manifest phải khớp.
    from src.evaluate import checked_config

    # YAML là cấu hình dự kiến; last.pt là trạng thái thực tế mà trainer đã ghi.
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    state = torch.load(Path(run_dir) / 'last.pt', map_location='cpu', weights_only=True)
    checked_config(state, config)
    # Phải đủ cả số epoch và từng dòng lịch sử; không gọi một run dở là kết quả hoàn tất.
    if state['epoch'] != config['training']['epochs'] or len(state['history']) != state['epoch']:
        raise ValueError(f'Run has incomplete epoch history: {run_dir}')
    return state


# Hàm run_one: thực thi một cấu hình qua CLI trainer trong tiến trình riêng, ghi stdout ra log.
# Đã hoàn tất thì kiểm tra và dùng lại; run dở chỉ tiếp tục khi truyền --resume rõ ràng.
def run_one(config_path, run_dir, device, resume=False):
    # Chuẩn hóa đường dẫn và xác định checkpoint dùng để tiếp tục, không dùng best.pt cho resume.
    run_dir = Path(run_dir)
    checkpoint = run_dir / 'last.pt'
    if checkpoint.exists():
        # Kiểm tra epoch trước khi quyết định bỏ qua hoặc tiếp tục run đã có.
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        if state['epoch'] == config['training']['epochs']:
            # Đủ epoch vẫn phải qua kiểm tra config/manifest/lịch sử rồi mới dùng lại.
            completed_state(config_path, run_dir)
            print(f'[COMPLETE] {run_dir}', flush=True)
            return
        if not resume:
            # Không tự nối một run đang dở nếu người chạy chưa truyền tùy chọn tiếp tục.
            raise FileExistsError(f'Incomplete run; pass --resume: {run_dir}')
    # Để log ngoài run_dir: trainer từ chối bắt đầu mới nếu thư mục run đã chứa file.
    log_dir = run_dir.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{run_dir.name}.log'
    # Dùng chính interpreter hiện tại; -X utf8 bảo đảm comment/log tiếng Việt trên Windows.
    # subprocess nhận list đối số, không ghép chuỗi lệnh qua shell.
    command = [sys.executable, '-X', 'utf8', '-m', 'src.train', '--config', str(config_path),
               '--run-dir', str(run_dir), '--device', device]
    if checkpoint.exists():
        # Trainer sẽ kiểm tra và phục hồi model/optimizer/scheduler/RNG từ checkpoint này.
        command += ['--resume', str(checkpoint)]
    print(f'[TRAIN] {run_dir}; stdout={log_path}', flush=True)
    with log_path.open('a' if checkpoint.exists() else 'w', encoding='utf-8') as log:
        # Chạy tuần tự trên cùng GPU; check=True dừng cả giai đoạn nếu một run bị lỗi.
        # stderr đi chung stdout để lỗi Python cũng nằm trong file log đã công bố.
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Mã thoát 0 chưa đủ: cần xác nhận đã thật sự train tới tổng epoch khóa trong YAML.
    completed_state(config_path, run_dir)
    print(f'[DONE] {run_dir}', flush=True)


# Hàm tune_and_lock: chạy bốn pilot, chọn recipe riêng theo validation và xuất giao thức cố định.
# Mỗi mô hình có cùng hai recipe × 10 epoch; các lượt chính sau đó dùng 100 epoch/tỷ lệ.
def tune_and_lock(device, resume=False):
    # Bốn file được tạo từ hai mô hình × hai recipe, dùng cùng split/seed/tỷ lệ.
    paths = make_tuning_configs()
    rows = []
    source_hashes = set()
    for path in paths:
        # Tên file resnet_sgd.yaml chứa tên mô hình và recipe, dùng đặt run ID tương ứng.
        model, recipe = path.stem.split('_')
        run_dir = Path('runs/week2/tuning') / path.stem
        run_one(path, run_dir, device, resume)
        # Số đo chọn từ checkpoint/log thật, không ghi sẵn accuracy dự kiến.
        state = completed_state(path, run_dir)
        metadata = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
        source_hashes.add(metadata['source_sha256'])
        # best_val_accuracy dùng chọn recipe; completed_epochs chứng minh ngân sách đã chạy đủ.
        # train_seconds cộng cả các epoch, kể cả epoch sau thời điểm best checkpoint.
        rows.append({'model': model, 'recipe': recipe,
                     'best_val_accuracy': state['best_val_accuracy'], 'best_epoch': state['best_epoch'],
                     'completed_epochs': state['epoch'], 'global_steps': state['global_step'],
                     'train_seconds': sum(r['train_seconds'] for r in state['history']),
                     'source_sha256': metadata['source_sha256'], 'run_dir': str(run_dir),
                     'config_path': path.as_posix()})
    if len(source_hashes) != 1:
        # Thay mã giữa các ứng viên sẽ khiến so sánh recipe có thêm biến không kiểm soát.
        raise ValueError('Tuning source versions differ; do not mix these trials')
    result_dir = Path('results/week2')
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / 'tuning.csv').open('w', newline='', encoding='utf-8') as stream:
        # Bảng này tách khỏi bảng các lượt chính: 10 epoch tuning khác 100 epoch chính.
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Ghi cả nguyên tắc chọn lẫn bản chụp config để phát hiện thay đổi sau khi khóa.
    protocol = {'date': '2026-11-14', 'main_epochs': 100, 'train_seed': 42, 'split_seed': 2026,
                'tuning_epochs_per_candidate': 10, 'tuning_candidates_per_model': 2,
                'selection': 'best validation accuracy; ties prefer SGD; test unused',
                'source_sha256': next(iter(source_hashes)), 'selected': {}, 'configs': {}}
    for model in ('resnet', 'vit'):
        # Chọn riêng trong hai ứng viên của đúng kiến trúc; không dùng điểm CNN để chọn recipe ViT.
        selected = choose_recipe([r for r in rows if r['model'] == model])
        protocol['selected'][model] = selected
        config = yaml.safe_load(Path(selected['config_path']).read_text(encoding='utf-8'))
        # Run chính tạo mạng mới; chỉ chuyển recipe, không lấy trọng số của run tuning.
        # Đổi tổng epoch từ 10 sang 100 cũng đổi độ dài cosine schedule tương ứng.
        config['training']['epochs'] = protocol['main_epochs']
        for label, ratio in RATIOS.items():
            # Giữ nguyên mô hình/recipe/seed, chỉ thay tỷ lệ dữ liệu ở ba run chính.
            final = copy.deepcopy(config)
            final['data']['ratio'] = ratio
            name = f'{model}_{label}_s42.yaml'
            path = Path('configs/week2') / name
            if path.exists() and yaml.safe_load(path.read_text(encoding='utf-8')) != final:
                # Không ghi đè một cấu hình khóa đã khác, tránh sửa lịch sử thí nghiệm.
                raise FileExistsError(f'Locked config differs: {path}')
            path.write_text(yaml.safe_dump(final, sort_keys=False), encoding='utf-8')
            protocol['configs'][name] = final
    # JSON khóa cả sáu cấu hình; phần ViT để Lan Anh chạy chính, script này chỉ train ResNet.
    path = Path('configs/week2/locked_protocol.json')
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise FileExistsError('Existing locked protocol differs; preserve the previous experiment')
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('[LOCKED] configs/week2/locked_protocol.json', flush=True)


# Hàm locked_resnet_configs: lấy đúng ba YAML ResNet đã khóa, phát hiện việc sửa recipe sau tuning.
# Trả list Path; không tự khóa lại bằng các số liệu từ tập test.
def locked_resnet_configs():
    # Đọc giao thức đã chọn trước loạt chính; thiếu file khóa thì dừng thay vì tự đoán recipe.
    protocol = json.loads(Path('configs/week2/locked_protocol.json').read_text(encoding='utf-8'))
    paths = [Path('configs/week2') / f'resnet_{label}_s42.yaml' for label in RATIOS]
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        # So toàn bộ dict: sửa epoch, lr, seed, tỷ lệ hoặc model đều bị phát hiện.
        if config != protocol['configs'][path.name]:
            raise ValueError(f'Config was changed after locking: {path}')
    return paths


# Hàm collect_validation: đối chiếu evaluator của Hiền với best.pt, xuất bảng để bàn giao.
# Chỉ dùng validation ở tuần 2; lưu dự đoán cạnh checkpoint trong runs, không mở tập test.
def collect_validation(device):
    # Dùng evaluator/summary của Hiền để cùng cách tính và cùng schema CSV với các lượt ViT sau này.
    from scripts.summarize import save_summary_csv, summarize
    from src.evaluate import append_run, build_run_row, evaluate_checkpoint

    output = Path('results/week2/resnet_runs.csv')
    # File tạm giúp kết quả bàn giao không trở thành bảng thiếu dòng nếu đánh giá bị ngắt.
    temporary = output.with_suffix('.tmp')
    if temporary.exists():
        raise FileExistsError(f'Previous collection is incomplete: {temporary}')
    for config_path in locked_resnet_configs():
        run_dir = Path('runs/week2') / config_path.stem
        final = completed_state(config_path, run_dir)
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        # best.pt đã được chọn bằng validation; không thay bằng last.pt chỉ vì nó mới hơn.
        result = evaluate_checkpoint(run_dir / 'best.pt', config, split='val', device=device)
        # Đếm số đúng trên đúng 5.000 ảnh phải khớp trainer; không dung thứ sai lệch lớn do nhầm split.
        if abs(result['accuracy'] - final['best_val_accuracy']) > 1e-12:
            raise ValueError(f'Evaluator/trainer accuracy mismatch: {run_dir}')
        (run_dir / 'predictions_val.json').write_text(json.dumps(result), encoding='utf-8')
        # build_run_row còn đọc last.pt để báo cáo đủ chi phí/bước cập nhật sau epoch tốt nhất.
        row = build_run_row(run_dir / 'config.yaml', run_dir / 'best.pt', result, 'val',
                            'Week 2 main ResNet; 100 epochs; one seed; test not evaluated')
        append_run(row, temporary)
        print(f'[VAL] {run_dir.name}: {result["correct"]}/{result["total"]}', flush=True)
    # Chỉ thay bảng cũ sau khi cả ba lượt đối chiếu đã đạt; không ghép pilot/tuning vào bảng chính.
    temporary.replace(output)
    # Một seed sẽ cho std mẫu N/A; summarize không tạo độ ổn định giả từ một checkpoint.
    with output.open(encoding='utf-8', newline='') as stream:
        summary = summarize(list(csv.DictReader(stream)))
    save_summary_csv(summary, 'results/week2/resnet_summary.csv')
    print('[SAVED] results/week2/resnet_runs.csv and resnet_summary.csv', flush=True)


# Hàm main: giao diện điều phối tuần 2, tách tuning/khóa cấu hình, train chính và đánh giá validation.
# Có thể khôi phục một giai đoạn dở bằng --resume; không tự huấn luyện lượt ViT chính.
def main():
    # tune chạy pilot và khóa YAML; train thực hiện ba ResNet; collect chỉ đánh giá validation.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('tune', 'train', 'collect'), required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    # parse_args lấy tùy chọn terminal; không có --stage hợp lệ thì argparse in lỗi/cách dùng.
    args = parser.parse_args()
    if args.stage == 'tune':
        tune_and_lock(args.device, args.resume)
    elif args.stage == 'train':
        # Chạy tuần tự theo thứ tự 10%, 50%, 100%, không tranh cùng GPU giữa hai tiến trình.
        for path in locked_resnet_configs():
            run_one(path, Path('runs/week2') / path.stem, args.device, args.resume)
    else:
        collect_validation(args.device)


if __name__ == '__main__':
    main()
