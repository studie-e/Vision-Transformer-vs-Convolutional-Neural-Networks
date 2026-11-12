"""Minimal model-independent trainer. See docs/minh-week1.md for the contract."""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.checkpoint import load_checkpoint, random_state, restore_random_state, save_checkpoint
from src.models.resnet import count_parameters


# Hàm seed_everything: đặt trạng thái ngẫu nhiên ban đầu cho một lượt chạy.
# Đầu vào: seed là số nguyên trong cấu hình. Hàm không trả dữ liệu, mà đổi thiết lập toàn cục.
# Mục đích: cùng seed/cấu hình/môi trường thì dễ kiểm tra và tái lập hơn;
# không bảo đảm kết quả bitwise giống nhau giữa các máy hoặc phiên bản PyTorch khác nhau.
def seed_everything(seed):
    # Chỉ đặt nếu biến chưa tồn tại; cuBLAS dùng workspace này cho một số phép toán CUDA xác định.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Seed cho thư viện random của Python; độc lập với NumPy và PyTorch.
    random.seed(seed)
    # Seed cho bộ sinh số ngẫu nhiên NumPy, dùng nếu pipeline dữ liệu gọi np.random.
    np.random.seed(seed)
    # Seed cho PyTorch trên CPU và các thiết bị CUDA; ảnh hưởng khởi tạo/dropout, v.v.
    torch.manual_seed(seed)
    # Không tự benchmark để chọn thuật toán convolution theo thời gian chạy từng lần.
    torch.backends.cudnn.benchmark = False
    # Yêu cầu cuDNN chọn các thuật toán convolution có tính xác định khi có hỗ trợ.
    torch.backends.cudnn.deterministic = True
    # Yêu cầu thuật toán xác định; báo lỗi nếu gặp phép toán không có cách thực hiện phù hợp.
    torch.use_deterministic_algorithms(True)


# Hàm sync: chờ GPU xử lý xong công việc để phép đo thời gian không bị kết thúc quá sớm.
# Đầu vào: torch.device. Không trả dữ liệu; với CPU thì không cần chờ thêm.
def sync(device):
    # Lệnh CUDA thường chạy bất đồng bộ: Python có thể đi tiếp khi GPU còn đang tính.
    if device.type == 'cuda':
        # Chờ các kernel trên thiết bị này hoàn tất trước khi đọc đồng hồ.
        torch.cuda.synchronize(device)


# Hàm run_epoch: chạy một lượt qua toàn bộ DataLoader, dùng chung cho train và validation.
# Đầu vào: model trả logits [B,C]; loader trả ảnh và nhãn long [B]; device là CPU/GPU.
# Có optimizer => cập nhật trọng số; optimizer=None => chỉ đánh giá.
# Trả về dict gồm loss trung bình theo ảnh, accuracy 0–1, số ảnh và số batch đã xử lý.
def run_epoch(model, loader, device, optimizer=None):
    # Không kiểm tra tên tập dữ liệu; việc có optimizer quyết định đây có phải bước học hay không.
    training = optimizer is not None
    # True bật chế độ train của BatchNorm/Dropout; False chuyển các lớp này sang eval.
    model.train(training)
    # Bộ đếm được đặt lại mỗi epoch: tổng loss theo ảnh, ảnh đúng, tổng ảnh, tổng batch.
    loss_sum, correct, samples, steps = 0.0, 0, 0, 0
    # CrossEntropyLoss consumes logits, not softmax probabilities.
    # Chỉ tạo đồ thị gradient khi train; validation không cần giữ đồ thị nên tiết kiệm bộ nhớ.
    with torch.set_grad_enabled(training):
        # Mỗi lần lặp nhận một batch ảnh [B,3,H,W] và nhãn lớp [B].
        for images, labels in loader:
            # Dữ liệu phải nằm trên cùng CPU/GPU với mô hình mới tính toán được.
            images, labels = images.to(device), labels.to(device)
            if training:
                # Xóa gradient của batch trước vì PyTorch mặc định cộng dồn gradient.
                # set_to_none=True bỏ tensor gradient cũ thay vì ghi toàn số 0.
                optimizer.zero_grad(set_to_none=True)
            # Forward: ảnh đi qua mạng, trả điểm thô từng lớp; chưa cần softmax.
            logits = model(images)
            # So sánh logits với chỉ số lớp thật; cross_entropy tự xử lý log-softmax.
            # Mặc định loss là trung bình của batch, không phải tổng loss cả epoch.
            loss = nn.functional.cross_entropy(logits, labels)
            # NaN/Inf thường báo hiệu dữ liệu hoặc cấu hình có vấn đề; dừng để kiểm tra.
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; inspect data/config before restarting')
            if training:
                # Backward dùng chain rule tính đạo hàm loss theo từng tham số cần học.
                loss.backward()
                # Optimizer dùng gradient vừa có để cập nhật trọng số theo SGD/AdamW.
                optimizer.step()
            # B là số ảnh thực tế; batch cuối có thể nhỏ hơn batch_size cấu hình.
            size = labels.size(0)
            # Đổi loss trung bình batch thành tổng theo ảnh, tránh batch cuối được tính quá nặng.
            # item() lấy giá trị số Python ra khỏi tensor để ghi thống kê.
            loss_sum += loss.item() * size
            # argmax(1) lấy lớp có logit lớn nhất; so nhãn, cộng số đúng trong batch.
            correct += (logits.argmax(1) == labels).sum().item()
            # Tích lũy đúng số ảnh đã xử lý, kể cả batch cuối nhỏ.
            samples += size
            # Khi train, mỗi batch ứng với một lần optimizer.step; khi val chỉ là số batch.
            steps += 1
    # Không thể tính loss/accuracy khi không có ảnh, đồng thời tránh chia cho 0.
    if samples == 0:
        raise ValueError('Empty DataLoader')
    # Accuracy là tỉ lệ 0–1; nhân 100 khi cần trình bày phần trăm trong báo cáo.
    return {'loss': loss_sum / samples, 'accuracy': correct / samples,
            'samples': samples, 'steps': steps}


# Hàm environment: ghi dấu môi trường và phiên bản mã để truy lại một thí nghiệm.
# Không có đối số; trả dict chứa Python/PyTorch/CUDA/GPU, commit, trạng thái Git và hash src.
# Đây là metadata, không phải số liệu accuracy hay thiết lập optimizer.
def environment():
    # Hàm con git: nhận các đối số lệnh Git, trả stdout dạng chuỗi đã bỏ khoảng trắng hai đầu.
    def git(*args):
        # Danh sách đối số không đi qua shell; capture_output giữ kết quả để đưa vào metadata.
        # check=False: Git trả mã lỗi không tự gây CalledProcessError; stdout có thể rỗng khi lỗi.
        return subprocess.run(['git', *args], capture_output=True, text=True, check=False).stdout.strip()

    # SHA-256 giúp nhận diện cả thay đổi chưa commit; đây không phải hash trọng số mô hình.
    digest = hashlib.sha256()
    # Sắp xếp để thứ tự duyệt file không làm hash thay đổi giữa hai lần đọc cùng mã.
    for file in sorted(Path('src').rglob('*.py')):
        # Đưa cả đường dẫn vào hash để đổi tên/chuyển file cũng được ghi nhận.
        digest.update(file.as_posix().encode())
        # Đọc toàn bộ byte, gồm cả comment: thêm chú thích cũng làm source_sha256 đổi.
        digest.update(file.read_bytes())
    # Lưu phiên bản phần mềm, GPU nếu có; HEAD chưa phản ánh phần sửa dở nên lưu thêm git_status/hash.
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda_runtime': torch.version.cuda, 'platform': platform.platform(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'source_sha256': digest.hexdigest()}


# Hàm write_metrics: ghi toàn bộ lịch sử epoch ra CSV để Hiền tổng hợp kết quả.
# Đầu vào: path là Path, history là danh sách dict có cùng các cột. Không trả dữ liệu.
# Ghi lại từ history trong checkpoint giúp tiếp tục run mà không nhân đôi các dòng epoch cũ.
def write_metrics(path, history):
    # Chưa có epoch nào thì không biết bộ cột cần ghi, nên chưa tạo CSV.
    if history:
        # 'w' ghi lại CSV; newline='' để csv xử lý xuống dòng đúng, kể cả trên Windows.
        with path.open('w', newline='', encoding='utf-8') as stream:
            # Dùng các key của epoch đầu làm tên và thứ tự cột.
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            # Dòng đầu là tên cột: epoch, lr, loss, accuracy, thời gian, bộ nhớ, v.v.
            writer.writeheader()
            # Các dòng sau là giá trị từng epoch đã lưu trong history.
            writer.writerows(history)


# Hàm fit: điều phối toàn bộ thí nghiệm, từ train/validation đến checkpoint, log và resume.
# Đầu vào: mô hình đã tạo; hai loader; config; thư mục run; các tùy chọn sau * phải truyền tên.
# device chọn CPU/GPU; resume là last.pt; stop_after_epoch dừng sớm nhưng giữ tổng epoch của scheduler.
# Trả về history của các epoch đã hoàn thành, đồng thời lưu kết quả vào run_dir.
# Khi bắt đầu run mới, bên gọi cần seed trước khi khởi tạo model (main đã làm việc này).
def fit(model, train_loader, val_loader, config, run_dir, *, device='cpu',
        resume=None, stop_after_epoch=None):
    """Train to the configured horizon; stop_after_epoch preserves the LR schedule.

    Resume is from last.pt in the same run, at an epoch boundary, with identical
    config and data factories. num_workers=0 is required for this first version.
    """
    # Chuẩn hóa chuỗi thiết bị/đường dẫn thành đối tượng có API kiểm tra và thao tác tương ứng.
    device, run_dir = torch.device(device), Path(run_dir)
    # Nhóm cấu hình train chứa tổng epoch, optimizer, lr, momentum, weight decay.
    settings = config['training']
    # Đây là tổng độ dài lịch học, không đổi nó thành 1 khi chỉ muốn chạy thử một epoch.
    epochs = settings['epochs']
    # Kiểm tra cả tổng epoch và mốc dừng: phải nằm trong [1, epochs].
    if epochs < 1 or (stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs):
        raise ValueError('Epoch bounds must be positive and within the configured horizon')
    # Bản đầu chỉ hỗ trợ resume xác định với dữ liệu đọc trong tiến trình chính.
    # Nhiều worker có RNG/trạng thái riêng chưa được checkpoint này quản lý.
    if any(loader.num_workers != 0 for loader in (train_loader, val_loader)):
        raise ValueError('Week-1 deterministic resume requires DataLoader num_workers=0')
    # Yêu cầu CUDA rõ ràng thì báo lỗi nếu không có, tránh vô tình đo tốc độ CPU thay GPU.
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for a smoke test')
    # Run mới không được ghi đè thư mục đã có file, để giữ lại kết quả cũ.
    if resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run already exists: {run_dir}; resume or choose another --run-dir')
    # Chỉ tiếp tục từ trạng thái cuối của đúng run này; không dùng best.pt để nối lịch sử train.
    if resume is not None and Path(resume).resolve() != (run_dir / 'last.pt').resolve():
        raise ValueError('Resume must use last.pt from this run directory')
    # Đặt trạng thái ngẫu nhiên trước train; nếu resume, RNG đã lưu sẽ được khôi phục ở dưới.
    # Lệnh này không khởi tạo lại trọng số của model đã tạo trước khi gọi fit.
    seed_everything(config['seed'])
    # Đưa tham số/buffer mô hình lên thiết bị trước khi optimizer tham chiếu chúng.
    model.to(device)
    # Nếu YAML không chỉ định optimizer thì dùng SGD.
    optimizer_name = settings.get('optimizer', 'sgd')
    if optimizer_name == 'sgd':
        # lr là tốc độ học; momentum tích lũy hướng cập nhật; weight_decay thêm regularization.
        # get(..., mặc_định) cho phép cấu hình bỏ momentum/weight_decay mà vẫn có giá trị rõ ràng.
        optimizer = torch.optim.SGD(model.parameters(), lr=settings['lr'],
            momentum=settings.get('momentum', 0.9), weight_decay=settings.get('weight_decay', 0))
    elif optimizer_name == 'adamw':
        # AdamW dùng các thống kê gradient thích nghi và weight decay tách khỏi cập nhật gradient.
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'],
                                     weight_decay=settings.get('weight_decay', 0))
    else:
        # Không tự đoán khi cấu hình ghi sai tên optimizer.
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    # Learning rate giảm theo đường cosine qua tổng epochs, mặc định về eta_min=0.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # start: epoch đã xong; global_step: số cập nhật; best: val tốt nhất; best_epoch: epoch tương ứng.
    # best=-vô_cùng để kết quả validation đầu tiên luôn được coi là tốt hơn ban đầu.
    start, global_step, best, best_epoch, history = 0, 0, -math.inf, 0, []
    if resume:
        # Tải các tensor và trạng thái huấn luyện từ last.pt lên CPU trước.
        state = load_checkpoint(resume)
        # Giữ nguyên toàn bộ config, kể cả seed/epoch, để không trộn hai thiết lập trong một run.
        if state['config'] != config:
            raise ValueError('Resume config differs; use the original config/horizon')
        # Khôi phục trọng số và buffer, ví dụ running_mean/running_var của BatchNorm.
        model.load_state_dict(state['model'])
        # Khôi phục momentum của SGD hoặc các thống kê moment của AdamW, không chỉ mỗi lr.
        optimizer.load_state_dict(state['optimizer'])
        # Khôi phục vị trí trên lịch lr; nếu bỏ bước này thì epoch tiếp theo có thể dùng lr sai.
        scheduler.load_state_dict(state['scheduler'])
        # Bắt đầu epoch sau epoch đã lưu, và tiếp tục đếm số bước từ trước.
        start, global_step = state['epoch'], state['global_step']
        # Lấy lại best/history để không quên checkpoint tốt nhất hoặc mất các dòng CSV cũ.
        best, best_epoch, history = state['best_val_accuracy'], state['best_epoch'], state['history']
        # Tiếp tục đúng trạng thái dropout/shuffle, thay vì quay về seed đầu run.
        restore_random_state(state['random'], (train_loader, val_loader))
    # Tạo cả các thư mục cha nếu chưa có; run resume được phép dùng thư mục đang tồn tại.
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        # Ghi môi trường, giờ UTC, số tham số và cấu hình để truy vết kết quả run mới.
        metadata = {**environment(), 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'parameters': count_parameters(model), 'device': str(device), 'config': config}
        # JSON dùng cho metadata; indent=2 để người đọc kiểm tra được trực tiếp.
        (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        # Chụp lại YAML đã dùng, không phụ thuộc file cấu hình bên ngoài sau đó có bị sửa hay không.
        (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    # end là mốc thoát lần gọi này; scheduler vẫn có T_max=epochs như khi chạy liền.
    end = stop_after_epoch if stop_after_epoch is not None else epochs
    # Resume ở epoch start+1: purge_step bỏ các log cũ từ bước đó trở đi nếu trước đây ghi dở.
    # Ở đây bước trên TensorBoard là số epoch, không phải global_step của từng batch.
    writer = SummaryWriter(str(run_dir / 'tensorboard'), purge_step=start + 1 if resume else None)
    # finally bên dưới bảo đảm đóng writer cả khi vòng train phát sinh lỗi.
    try:
        # range không gồm cận phải nên dùng end+1 để vẫn chạy epoch cuối.
        for epoch in range(start + 1, end + 1):
            if device.type == 'cuda':
                # Đặt lại thống kê đỉnh để đo riêng epoch này, không lấy đỉnh của toàn run.
                torch.cuda.reset_peak_memory_stats(device)
            # Chờ tác vụ GPU trước đó hoàn tất rồi mới bắt đầu đồng hồ.
            sync(device)
            # perf_counter là đồng hồ đo khoảng thời gian, phù hợp hơn giờ lịch để benchmark.
            started = time.perf_counter()
            # Ghi lr thực sự dùng cho epoch hiện tại trước khi scheduler giảm nó ở cuối epoch.
            lr = optimizer.param_groups[0]['lr']
            # Có optimizer: forward, backward và cập nhật trọng số trên toàn tập train.
            train = run_epoch(model, train_loader, device, optimizer)
            # Chờ train trên GPU xong để số giây đo được không chỉ là thời gian gửi lệnh.
            sync(device)
            # Khoảng này gồm vòng train/đọc batch, chưa gồm validation hay lưu checkpoint.
            train_seconds = time.perf_counter() - started
            # Không truyền optimizer: chỉ validation, không backward/cập nhật trọng số.
            val = run_epoch(model, val_loader, device)
            # Chờ validation hoàn tất trước khi chốt số đo cả train+validation.
            sync(device)
            # Chỉ cộng số batch train vì mỗi batch train có một lần optimizer.step.
            global_step += train['steps']
            # Dấu > giữ checkpoint sớm hơn khi hai epoch có cùng validation accuracy.
            improved = val['accuracy'] > best
            if improved:
                # Cập nhật giá trị và vị trí best; không sử dụng tập test để chọn checkpoint.
                best, best_epoch = val['accuracy'], epoch
            # Một dòng CSV chứa trạng thái epoch/bước/lr, chất lượng train/val và chi phí chạy.
            # epoch_seconds gồm train+val tới thời điểm này, chưa gồm ghi checkpoint/log phía dưới.
            # Chia 2**20 đổi byte sang MiB; đây là bộ nhớ tensor CUDA cấp phát, không phải toàn VRAM.
            row = {'epoch': epoch, 'global_step': global_step, 'lr': lr,
                   'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
                   'val_loss': val['loss'], 'val_accuracy': val['accuracy'],
                   'train_seconds': train_seconds, 'epoch_seconds': time.perf_counter() - started,
                   'peak_cuda_mb': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0}
            # Giữ lại từng epoch để dựng CSV và tiếp tục lịch sử khi resume.
            history.append(row)
            # Đặt lr cho epoch sau; gọi sau optimizer.step trong vòng train.
            scheduler.step()
            # Checkpoint đủ để tiếp tục: trọng số, optimizer, lịch lr, bộ đếm, cấu hình, RNG và lịch sử.
            # random_state còn lấy generator của cả train_loader và val_loader.
            state = {'epoch': epoch, 'global_step': global_step, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'best_val_accuracy': best, 'best_epoch': best_epoch, 'config': config,
                     'random': random_state((train_loader, val_loader)), 'history': history}
            if improved:
                # best.pt dùng đánh giá/suy luận: chỉ ghi khi validation accuracy tăng.
                save_checkpoint(run_dir / 'best.pt', state)
            # last.pt luôn là epoch mới nhất, dùng để tiếp tục thí nghiệm.
            save_checkpoint(run_dir / 'last.pt', state)
            # Ghi CSV từ toàn history, gồm các epoch cũ đã nạp từ checkpoint khi resume.
            write_metrics(run_dir / 'metrics.csv', history)
            # Mỗi chỉ số số học được ghi thành một đường biểu diễn trên TensorBoard.
            for key, value in row.items():
                # epoch đã làm trục ngang nên không cần ghi thêm một đường tên 'epoch'.
                if key != 'epoch':
                    writer.add_scalar(key, value, epoch)
            # Đẩy log đang đệm ra file sau mỗi epoch để dễ xem và giảm mất log khi thoát.
            writer.flush()
            # In JSON một dòng ra terminal để theo dõi; flush=True không chờ đầy bộ đệm mới hiện.
            print(json.dumps(row), flush=True)
    finally:
        # Giải phóng writer và đẩy phần log còn lại; không có nghĩa là lỗi train bị bỏ qua.
        writer.close()
    # Bên gọi có thể dùng danh sách này để kiểm tra kết quả mà không cần đọc lại CSV.
    return history


# Hàm factory: tìm một hàm tạo mô hình/dữ liệu theo tên module:function trong YAML.
# Ví dụ src.models.resnet:build_model => trả đối tượng hàm build_model, chưa gọi hàm đó.
# Đầu vào là chuỗi cấu hình tin cậy của nhóm; đầu ra thường là một callable.
def factory(path):
    # Tách vị trí module Python và tên hàm cần lấy theo dấu hai chấm.
    module, name = path.split(':')
    # import_module nạp module; getattr lấy hàm theo tên, giúp đổi ViT/ResNet mà không sửa trainer.
    return getattr(importlib.import_module(module), name)


# Hàm main: điểm vào khi chạy python -m src.train từ terminal.
# Nhận tham số qua dòng lệnh, đọc YAML, tạo model/loaders rồi gọi fit; không trả dữ liệu cho terminal.
def main():
    # argparse tạo giao diện lệnh và phần trợ giúp --help từ mô tả module.
    parser = argparse.ArgumentParser(description=__doc__)
    # Đường dẫn YAML bắt buộc, chứa model/data/training/seed.
    parser.add_argument('--config', required=True)
    # Mỗi lượt chạy phải có thư mục riêng để không ghi đè kết quả.
    parser.add_argument('--run-dir', required=True)
    # Mặc định dùng GPU; có thể chọn CPU cho các phép kiểm tra nhỏ.
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    # Bỏ trống để train mới; truyền last.pt của cùng run nếu muốn tiếp tục.
    parser.add_argument('--resume')
    # Chỉ dừng lần gọi này sớm, không đổi tổng epoch trong YAML.
    parser.add_argument('--stop-after-epoch', type=int)
    # Chuyển các chuỗi người dùng gõ thành đối tượng args có các thuộc tính tương ứng.
    args = parser.parse_args()
    # safe_load đọc cấu trúc dữ liệu YAML, không dùng cơ chế tạo đối tượng Python tùy ý.
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    # Phải seed trước khi tạo model để cùng cấu hình thì khởi tạo trọng số cũng lặp lại được.
    seed_everything(config['seed'])
    # factory(...) trả hàm; cặp (config) tiếp theo gọi hàm đó để tạo mô hình thực tế.
    model = factory(config['model']['factory'])(config)
    # Tương tự cho dữ liệu: hàm của Hiền có thể thay adapter pilot khi đã bàn giao.
    train_loader, val_loader = factory(config['data']['factory'])(config)
    # Chuyển các đối tượng đã tạo và tùy chọn terminal vào bộ điều phối thí nghiệm.
    fit(model, train_loader, val_loader, config, args.run_dir, device=args.device,
        resume=args.resume, stop_after_epoch=args.stop_after_epoch)


if __name__ == '__main__':
    # Chỉ mở giao diện dòng lệnh khi chạy trực tiếp; import fit từ test sẽ không tự train.
    main()
