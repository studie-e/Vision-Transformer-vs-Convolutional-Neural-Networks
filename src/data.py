"""Chuẩn bị dữ liệu chính — Hiền.

Tạo split 45k train / 5k val / 10k test theo phân tầng; lưu manifest JSON cho mỗi seed.
Cung cấp hàm build_loaders tương thích với trainer của Minh (src.train).

Quy ước:
- Tập con lồng nhau: 10% nằm trong 50%, 50% nằm trong 100%.
- Hai mô hình dùng đúng cùng danh sách ảnh từ manifest.
- Augmentation (RandomCrop, RandomHorizontalFlip) chỉ áp dụng trên train; val/test dùng transform eval.
- Seed chia dữ liệu (split_seed) và seed train (seed) tách biệt.

Nguồn CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import json
import hashlib
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Hằng số toàn cục
# ---------------------------------------------------------------------------

# 10 lớp CIFAR-10 theo thứ tự chỉ số nhãn 0–9.
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = 10

# Mean và std đo từ toàn bộ 50.000 ảnh train CIFAR-10 (giá trị chuẩn được dùng rộng rãi).
# Dùng thống kê này thay vì (0.5,0.5,0.5) để chuẩn hóa phù hợp hơn với phân phối CIFAR-10.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Các tỷ lệ tập con; mẫu số là 45.000 ảnh train sau khi tách validation.
RATIOS = (0.10, 0.50, 1.00)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def make_train_transform():
    """Transform cho tập train: augmentation → ToTensor → Normalize.

    RandomCrop và RandomHorizontalFlip là augmentation cơ bản phổ biến cho CIFAR-10.
    Chỉ áp dụng trên train, không áp dụng trên val/test.
    """
    return transforms.Compose([
        # Thêm padding 4 pixel rồi cắt ngẫu nhiên 32×32; giúp mạng bất biến hơn với dịch nhỏ.
        transforms.RandomCrop(32, padding=4),
        # Lật ngang ngẫu nhiên với xác suất 0.5; phù hợp vì CIFAR-10 không phân biệt chiều ngang.
        transforms.RandomHorizontalFlip(),
        # Chuyển ảnh PIL/numpy sang tensor float [0,1] rồi chuẩn hóa về phân phối chuẩn.
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def make_eval_transform():
    """Transform cho val/test: chỉ ToTensor và Normalize, không augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# Tạo split và manifest
# ---------------------------------------------------------------------------

def _stratified_indices(targets, n_per_class, rng):
    """Lấy n_per_class chỉ số cho mỗi lớp theo thứ tự ngẫu nhiên từ rng.

    Đầu vào:
        targets: list/tensor nhãn của toàn bộ dataset.
        n_per_class: số ảnh cần lấy mỗi lớp.
        rng: torch.Generator đã được seed để tái lập.
    Trả về:
        list chỉ số đã sắp xếp theo thứ tự ngẫu nhiên trong mỗi lớp.

    Phân tầng: đảm bảo mỗi lớp có đúng n_per_class ảnh, không lệch phân phối.
    """
    # Nhóm chỉ số theo nhãn trước.
    class_indices = [[] for _ in range(NUM_CLASSES)]
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    selected = []
    for cls_idx in range(NUM_CLASSES):
        pool = class_indices[cls_idx]
        # Xáo trộn trong từng lớp theo rng; không dùng random toàn cục để tránh ảnh hưởng RNG ngoài.
        perm = torch.randperm(len(pool), generator=rng).tolist()
        # Lấy n_per_class đầu tiên sau khi xáo trộn; đảm bảo đúng số lượng.
        if n_per_class > len(pool):
            raise ValueError(
                f'Yêu cầu {n_per_class} ảnh lớp {cls_idx} nhưng chỉ có {len(pool)}')
        selected.extend(pool[perm[i]] for i in range(n_per_class))
    return selected


def make_splits(split_seed, root='data', download=False):
    """Tạo split 45k/5k train/val từ tập train CIFAR-10 và tập con 10%/50%/100%.

    Đầu vào:
        split_seed: seed điều khiển chia train/val và thứ tự tập con; độc lập với seed train.
        root: thư mục lưu dữ liệu CIFAR-10.
        download: có tải CIFAR-10 từ mạng nếu chưa có không.
    Trả về:
        dict với các key:
          'train_indices': list 45.000 chỉ số trong tập train CIFAR-10 gốc dùng làm train.
          'val_indices':   list 5.000 chỉ số dùng làm validation.
          'subsets': dict {ratio_str: list chỉ số tập con} cho '0.10', '0.50', '1.00'.
          'split_seed': seed đã dùng (để kiểm tra).
          'stats': dict thống kê số ảnh/lớp cho train, val và từng tập con.

    Quy tắc lồng nhau:
        10% nằm trong 50%, 50% nằm trong 100% (= toàn bộ train).
        Đạt được bằng cách lấy dần: 450/lớp, 2250/lớp, 4500/lớp từ cùng thứ tự ngẫu nhiên.
    """
    # Nạp dataset chỉ để lấy nhãn; không cần transform vì chỉ dùng chỉ số và nhãn.
    base = datasets.CIFAR10(root=root, train=True, download=download)
    # 50.000 ảnh CIFAR-10 train, 5.000 ảnh/lớp; kiểm tra để phát hiện lỗi tải sớm.
    if len(base) != 50_000:
        raise ValueError(f'CIFAR-10 train cần 50.000 ảnh; tìm thấy {len(base)}')

    rng = torch.Generator().manual_seed(split_seed)

    # --- Bước 1: tách 5.000 validation phân tầng (500/lớp) từ 50.000 ảnh train gốc ---
    # 500 ảnh/lớp × 10 lớp = 5.000 val; phần còn lại (4.500/lớp) là train.
    val_indices = _stratified_indices(base.targets, n_per_class=500, rng=rng)
    val_set = set(val_indices)

    # Các chỉ số không thuộc val là train; giữ thứ tự ổn định bằng cách lọc theo thứ tự gốc.
    train_indices = [i for i in range(len(base)) if i not in val_set]
    # Phải có đúng 45.000; kiểm tra để phát hiện lỗi logic.
    assert len(train_indices) == 45_000, f'Kỳ vọng 45.000 train; thực tế {len(train_indices)}'
    # Không được có ảnh nào trong cả train và val (rò rỉ dữ liệu).
    assert set(train_indices).isdisjoint(val_set), 'Train và val có chỉ số trùng!'

    # --- Bước 2: tạo tập con lồng nhau từ train_indices ---
    # Mỗi lớp trong train_indices có đúng 4.500 ảnh; lấy dần 450 / 2.250 / 4.500.
    train_targets = [base.targets[i] for i in train_indices]

    # Xáo trộn train_indices theo lớp để lấy tập con phân tầng và lồng nhau.
    # Dùng rng tiếp theo sau bước 1 để bảo đảm cùng split_seed cho ra cùng kết quả.
    class_train = [[] for _ in range(NUM_CLASSES)]
    for local_i, global_i in enumerate(train_indices):
        class_train[train_targets[local_i]].append(global_i)

    # Xáo trộn trong từng lớp một lần duy nhất → lấy 450/2250/4500 theo thứ tự này.
    ordered_per_class = []
    for cls in range(NUM_CLASSES):
        perm = torch.randperm(len(class_train[cls]), generator=rng).tolist()
        ordered_per_class.append([class_train[cls][perm[i]] for i in range(len(class_train[cls]))])

    n_per_class = {
        '0.10': 450,    # 10% × 4.500/lớp
        '0.50': 2_250,  # 50% × 4.500/lớp
        '1.00': 4_500,  # 100% × 4.500/lớp
    }
    subsets = {}
    for ratio_str, n in n_per_class.items():
        # Lấy n đầu tiên từ mỗi lớp; vì cùng thứ tự nên 10% ⊂ 50% ⊂ 100%.
        subset = []
        for cls in range(NUM_CLASSES):
            subset.extend(ordered_per_class[cls][:n])
        subsets[ratio_str] = subset

    # Kiểm tra tính lồng nhau: 10% phải nằm trong 50%, 50% phải nằm trong 100%.
    assert set(subsets['0.10']).issubset(subsets['0.50']), '10% không nằm trong 50%!'
    assert set(subsets['0.50']).issubset(subsets['1.00']), '50% không nằm trong 100%!'
    assert set(subsets['1.00']) == set(train_indices), '100% phải bằng toàn bộ train!'

    # --- Bước 3: thống kê số ảnh/lớp để kiểm tra ---
    def count_per_class(indices):
        counts = [0] * NUM_CLASSES
        for i in indices:
            counts[base.targets[i]] += 1
        return dict(zip(CLASSES, counts))

    stats = {
        'train_total': len(train_indices),
        'val_total': len(val_indices),
        'train_per_class': count_per_class(train_indices),
        'val_per_class': count_per_class(val_indices),
        'subsets': {r: {'total': len(idx), 'per_class': count_per_class(idx)}
                    for r, idx in subsets.items()},
    }

    return {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'subsets': subsets,
        'split_seed': split_seed,
        'stats': stats,
    }


def save_manifest(split_data, out_dir='results/splits'):
    """Lưu manifest JSON và thống kê ra thư mục results/splits/.

    Đầu vào:
        split_data: dict từ make_splits().
        out_dir: thư mục đầu ra.
    Trả về:
        Path của manifest đã lưu.

    File đầu ra:
        split_seed{SEED}_manifest.json — toàn bộ chỉ số và thống kê.
        split_seed{SEED}_stats.json   — chỉ phần thống kê (nhỏ gọn hơn để xem nhanh).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seed = split_data['split_seed']
    manifest_path = out / f'split_seed{seed}_manifest.json'
    stats_path    = out / f'split_seed{seed}_stats.json'

    # Ghi manifest đầy đủ (có thể lớn vì chứa 60.000 chỉ số).
    manifest_path.write_text(json.dumps(split_data, indent=2), encoding='utf-8')
    # Ghi riêng stats để kiểm tra nhanh không cần đọc toàn bộ.
    stats_path.write_text(json.dumps(split_data['stats'], indent=2), encoding='utf-8')

    return manifest_path


def load_manifest(split_seed, splits_dir='results/splits'):
    """Nạp manifest đã lưu từ file JSON.

    Đầu vào:
        split_seed: seed đã dùng khi tạo manifest.
        splits_dir: thư mục chứa manifest.
    Trả về:
        dict tương đương với kết quả make_splits().
    """
    path = Path(splits_dir) / f'split_seed{split_seed}_manifest.json'
    if not path.exists():
        raise FileNotFoundError(
            f'Manifest không tồn tại: {path}\n'
            f'Chạy: python -m src.data --generate để tạo split.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    validate_manifest(manifest, split_seed=split_seed)
    return manifest


def validate_manifest(m, targets=None, split_seed=None):
    """Validate actual indices; optionally recompute balance from CIFAR labels."""
    if split_seed is not None and m['split_seed'] != split_seed:
        raise ValueError('Manifest split_seed mismatch')
    groups = {'train': m['train_indices'], 'val': m['val_indices'], **m['subsets']}
    sizes = {'train': 45000, 'val': 5000, '0.10': 4500, '0.50': 22500, '1.00': 45000}
    for name, size in sizes.items():
        idx = groups[name]
        if len(idx) != size or len(set(idx)) != size:
            raise ValueError(f'{name}: wrong size or duplicate indices')
        if any(type(i) is not int or not 0 <= i < 50000 for i in idx):
            raise ValueError(f'{name}: invalid index')
        if targets is not None:
            counts = Counter(int(targets[i]) for i in idx)
            if counts != {c: size // 10 for c in range(10)}:
                raise ValueError(f'{name}: not stratified against actual labels')
    train, val = set(groups['train']), set(groups['val'])
    if train & val or train | val != set(range(50000)):
        raise ValueError('Train/validation overlap or incomplete coverage')
    if not set(groups['0.10']) <= set(groups['0.50']) <= set(groups['1.00']) == train:
        raise ValueError('Subsets are not nested within training data')
    expected_stats = {
        'train_total': 45000, 'val_total': 5000,
        'train_per_class': dict.fromkeys(CLASSES, 4500),
        'val_per_class': dict.fromkeys(CLASSES, 500),
        'subsets': {r: {'total': sizes[r], 'per_class': dict.fromkeys(CLASSES, sizes[r] // 10)}
                    for r in ('0.10', '0.50', '1.00')},
    }
    if m['stats'] != expected_stats:
        raise ValueError('Manifest statistics differ from the required split sizes')
    return True


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Xây dựng DataLoader cho trainer
# ---------------------------------------------------------------------------

def build_loaders(config):
    """Tạo (train_loader, val_loader) từ manifest đã lưu, tương thích với src.train.

    Đầu vào:
        config: dict YAML với các key:
            config['data']['root']:       thư mục CIFAR-10.
            config['data']['download']:   có tải không (bool, mặc định False).
            config['data']['split_seed']: seed chia dữ liệu.
            config['data']['ratio']:      tỷ lệ tập con ('0.10', '0.50', '1.00').
            config['data']['batch_size']: kích thước batch.
            config['data']['splits_dir']: thư mục manifest (mặc định 'results/splits').
            config['seed']:               seed train (dùng cho DataLoader generator).
    Trả về:
        (train_loader, val_loader): hai DataLoader tương thích với fit() của Minh.

    Lưu ý:
        - Train dùng make_train_transform() (có augmentation).
        - Val dùng make_eval_transform() (không augmentation).
        - num_workers=0 để tương thích với resume xác định của trainer tuần 1.
    """
    data_cfg = config['data']
    root        = data_cfg.get('root', 'data')
    download    = data_cfg.get('download', False)
    split_seed  = data_cfg['split_seed']
    ratio       = f"{float(data_cfg['ratio']):.2f}"
    batch_size  = data_cfg.get('batch_size', 128)
    splits_dir  = data_cfg.get('splits_dir', 'results/splits')
    train_seed  = config['seed']

    # Nạp manifest; không tự tạo split khi chạy build_loaders để tránh sự không nhất quán.
    manifest = load_manifest(split_seed, splits_dir)

    # Lấy danh sách chỉ số tập con theo tỷ lệ.
    if ratio not in manifest['subsets']:
        raise ValueError(f'Tỷ lệ {ratio} không có trong manifest; hợp lệ: {list(manifest["subsets"])}')
    train_idx = manifest['subsets'][ratio]
    val_idx   = manifest['val_indices']

    # Dataset gốc không có transform để dùng lại cho cả train/val với transform khác nhau.
    train_base = datasets.CIFAR10(root=root, train=True, transform=make_train_transform(),
                                  download=download)
    val_base   = datasets.CIFAR10(root=root, train=True, transform=make_eval_transform(),
                                  download=download)
    validate_manifest(manifest, train_base.targets, split_seed)

    train_ds = Subset(train_base, train_idx)
    val_ds   = Subset(val_base,   val_idx)

    # Generator riêng cho shuffle train; được trainer lưu vào checkpoint khi resume.
    train_gen = torch.Generator().manual_seed(train_seed)
    val_gen   = torch.Generator().manual_seed(train_seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=train_gen)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, generator=val_gen)
    train_loader.manifest_sha256 = val_loader.manifest_sha256 = manifest_digest(manifest)

    return train_loader, val_loader


def build_test_loader(config):
    """Tạo test_loader từ tập test CIFAR-10 chính thức.

    Chỉ dùng SAU khi toàn bộ phương pháp và checkpoint đã được khóa.
    Không dùng để tuning hoặc chọn cấu hình.
    """
    data_cfg = config['data']
    root       = data_cfg.get('root', 'data')
    download   = data_cfg.get('download', False)
    batch_size = data_cfg.get('batch_size', 128)

    test_ds = datasets.CIFAR10(root=root, train=False, transform=make_eval_transform(),
                               download=download)
    # 10.000 ảnh test; kiểm tra để phát hiện tải thiếu.
    if len(test_ds) != 10_000:
        raise ValueError(f'CIFAR-10 test cần 10.000 ảnh; tìm thấy {len(test_ds)}')

    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Kiểm tra nhanh (chạy từ terminal)
# ---------------------------------------------------------------------------

def _verify_splits(splits_dir='results/splits', seeds=(2026, 2027, 2028), targets=None):
    """Kiểm tra tính đúng đắn của tất cả manifest đã tạo.

    Chạy: python -m src.data --verify
    """
    all_ok = True
    for seed in seeds:
        path = Path(splits_dir) / f'split_seed{seed}_manifest.json'
        if not path.exists():
            print(f'[FAIL] Missing manifest seed={seed}')
            all_ok = False
            continue
        m = json.loads(path.read_text(encoding='utf-8'))
        try:
            validate_manifest(m, targets, seed)
        except (ValueError, KeyError, TypeError) as error:
            print(f'[FAIL] seed={seed}: {error}')
            all_ok = False
            continue
        s = m['stats']

        ok = True
        # Kiểm tra số lượng
        if s['train_total'] != 45_000:
            print(f'[FAIL] seed={seed}: train cần 45.000, có {s["train_total"]}'); ok = False
        if s['val_total'] != 5_000:
            print(f'[FAIL] seed={seed}: val cần 5.000, có {s["val_total"]}'); ok = False

        # Kiểm tra cân bằng lớp trong train (phải là 4.500/lớp)
        for cls, cnt in s['train_per_class'].items():
            if cnt != 4_500:
                print(f'[FAIL] seed={seed} train lớp {cls}: {cnt} (cần 4.500)'); ok = False

        # Kiểm tra cân bằng lớp trong val (phải là 500/lớp)
        for cls, cnt in s['val_per_class'].items():
            if cnt != 500:
                print(f'[FAIL] seed={seed} val lớp {cls}: {cnt} (cần 500)'); ok = False

        # Kiểm tra tỷ lệ tập con
        expected_totals = {'0.10': 4_500, '0.50': 22_500, '1.00': 45_000}
        for ratio, exp in expected_totals.items():
            if ratio in s['subsets']:
                got = s['subsets'][ratio]['total']
                if got != exp:
                    print(f'[FAIL] seed={seed} subset {ratio}: {got} (cần {exp})'); ok = False

        # Kiểm tra không giao nhau: đọc lại chỉ số
        train_set = set(m['train_indices'])
        val_set   = set(m['val_indices'])
        if not train_set.isdisjoint(val_set):
            print(f'[FAIL] seed={seed}: train và val có chỉ số chung!'); ok = False

        # Kiểm tra lồng nhau
        sub10 = set(m['subsets']['0.10'])
        sub50 = set(m['subsets']['0.50'])
        sub100 = set(m['subsets']['1.00'])
        if not sub10.issubset(sub50):
            print(f'[FAIL] seed={seed}: 10% không nằm trong 50%'); ok = False
        if not sub50.issubset(sub100):
            print(f'[FAIL] seed={seed}: 50% không nằm trong 100%'); ok = False
        if sub100 != train_set:
            print(f'[FAIL] seed={seed}: 100% không khớp train_indices'); ok = False

        if ok:
            subset_info = [f'{r}:{s["subsets"][r]["total"]}' for r in s['subsets']]
            print(f'[OK]   seed={seed}: train={s["train_total"]}, val={s["val_total"]}, '
                  f'subsets={subset_info}')
        else:
            all_ok = False

    return all_ok


if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Tạo và kiểm tra split CIFAR-10 cho nhóm.')
    parser.add_argument('--generate', action='store_true',
                        help='Tạo split cho 3 seed (2026, 2027, 2028).')
    parser.add_argument('--verify', action='store_true',
                        help='Kiểm tra tất cả manifest đã tạo.')
    parser.add_argument('--root', default='data',
                        help='Thư mục dữ liệu CIFAR-10 (mặc định: data).')
    parser.add_argument('--download', action='store_true',
                        help='Tải CIFAR-10 nếu chưa có.')
    parser.add_argument('--verify-labels', action='store_true',
                        help='Recompute stratification using real CIFAR-10 labels.')
    parser.add_argument('--splits-dir', default='results/splits',
                        help='Thư mục lưu manifest (mặc định: results/splits).')
    args = parser.parse_args()

    if args.generate:
        for seed in (2026, 2027, 2028):
            print(f'Đang tạo split seed={seed} ...')
            split_data = make_splits(split_seed=seed, root=args.root, download=args.download)
            path = save_manifest(split_data, out_dir=args.splits_dir)
            s = split_data['stats']
            print(f'  Lưu tại: {path}')
            print(f'  train={s["train_total"]}, val={s["val_total"]}')
            for ratio, info in s['subsets'].items():
                print(f'  subset {ratio}: {info["total"]} ảnh')
        print('Xong. Chạy --verify để kiểm tra.')

    if args.verify:
        targets = (datasets.CIFAR10(root=args.root, train=True, download=args.download).targets
                   if args.verify_labels else None)
        ok = _verify_splits(splits_dir=args.splits_dir, targets=targets)
        sys.exit(0 if ok else 1)

    if not args.generate and not args.verify:
        parser.print_help()
