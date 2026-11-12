"""Week-1 diagnostic data only; replace with Hien's official split factory.

Loads only the CIFAR-10 training partition. No test data is used for validation.
Source: https://docs.pytorch.org/vision/0.26/generated/torchvision.datasets.CIFAR10.html
"""

import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url


# Hàm pilot_indices: lấy ngẫu nhiên hai nhóm chỉ số train/validation không giao nhau.
# Đầu vào: tổng số ảnh, số ảnh mỗi nhóm và split seed; đầu ra: hai list chỉ số.
# Đây chỉ là chia pilot ngẫu nhiên, chưa phân tầng theo lớp và chưa phải split chính của Hiền.
def pilot_indices(size, train_samples, val_samples, seed):
    # Mỗi nhóm phải có ít nhất một ảnh và tổng hai nhóm không được vượt số ảnh có sẵn.
    if min(train_samples, val_samples) < 1 or train_samples + val_samples > size:
        raise ValueError('Pilot sizes must be positive, disjoint and fit the training partition')
    # randperm tạo hoán vị 0..size-1, mỗi chỉ số xuất hiện đúng một lần.
    # Generator riêng được seed nên cùng size/seed sẽ lấy cùng thứ tự, không phụ thuộc RNG nơi khác.
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    # Cắt hai đoạn liền nhau, không trùng chỉ số; phần ảnh chưa cần dùng được để lại.
    return indices[:train_samples], indices[train_samples:train_samples + val_samples]


# Hàm build_loaders: dựng hai DataLoader phục vụ kiểm tra kỹ thuật ResNet tuần 1.
# Đầu vào: config chứa data và seed; trả (train_loader, val_loader) cho trainer dùng chung.
# Cả hai nhóm đều lấy từ tập train CIFAR-10, không lấy tập test chính thức làm validation.
def build_loaders(config):
    # Đọc nhóm cấu hình dữ liệu: nguồn, nơi lưu, số ảnh, split_seed, batch_size và quyền tải.
    data = config['data']
    # Các phép tiền xử lý được áp dụng theo thứ tự trên từng ảnh khi DataLoader lấy mẫu.
    transform = transforms.Compose([
        # ToTensor: ảnh uint8 H×W×3 thành tensor float [3,H,W], giá trị trong [0,1].
        # Normalize: (x-0.5)/0.5 cho từng kênh RGB => [-1,1]; không phải mean/std đo từ dataset.
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    # Nhánh fastai đọc kho CIFAR đã đổi thành PNG; dùng khi nguồn archive Toronto tải chậm.
    if data.get('source', 'torchvision') == 'fastai':
        # fastai's documented CIFAR image archive, used only to unblock this pilot.
        # Source: https://docs.fast.ai/data.external.html (URLs.CIFAR).
        # Path giúp nối đường dẫn đúng cú pháp Windows/Linux thay vì tự ghép dấu gạch chéo.
        root = Path(data['root'])
        # Chỉ đọc thư mục train của archive; thư mục test nếu được giải nén cũng không được sử dụng.
        train_dir = root / 'cifar10' / 'train'
        # Chỉ tải/giải nén khi train chưa có và cấu hình cho phép download.
        if not train_dir.exists() and data.get('download', False):
            # Tải hoặc dùng archive cache hợp lệ; MD5 kiểm tra file khớp bản archive pilot đã ghi nhận.
            download_url('https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz', str(root),
                         filename='cifar10-fastai.tgz', md5='7d7b892d003476893d94d030e4f7c0f0')
            # with tự đóng archive; filter='data' áp dụng bộ lọc đường dẫn/liên kết khi giải nén.
            with tarfile.open(root / 'cifar10-fastai.tgz') as archive:
                archive.extractall(root, filter='data')
        # ImageFolder suy nhãn từ tên thư mục lớp và sắp thứ tự file ổn định khi tạo dataset.
        dataset = datasets.ImageFolder(str(train_dir), transform=transform)
        # Phát hiện giải nén thiếu hoặc trỏ nhầm dữ liệu trước khi tạo các tập con.
        if len(dataset) != 50000 or len(dataset.classes) != 10:
            raise ValueError('Expected 50,000 training images across 10 CIFAR classes')
    # Nếu không khai báo source thì mặc định dùng CIFAR10 chuẩn của torchvision.
    elif data.get('source', 'torchvision') == 'torchvision':
        # train=True đọc tập train 50.000 ảnh; transform giống nhánh PNG, download theo cấu hình.
        dataset = datasets.CIFAR10(root=data['root'], train=True, transform=transform,
                                   download=data.get('download', False))
    else:
        # Không tự chọn nguồn khác khi YAML gõ sai, tránh đổi dataset mà người chạy không biết.
        raise ValueError('Pilot source must be torchvision or fastai')
    # Split seed điều khiển danh sách ảnh, tách khỏi run seed điều khiển shuffle/khởi tạo mạng.
    train_indices, val_indices = pilot_indices(
        len(dataset), data['train_samples'], data['val_samples'], data['split_seed'])
    # These are diagnostic random subsets, NOT the final stratified 45k/5k split.
    # Trả tuple hai loader; Subset chỉ chọn chỉ số, vẫn dùng ảnh/transform của dataset gốc.
    return (
        # Train xáo trộn ảnh mỗi epoch; batch_size lấy từ YAML, num_workers=0 để hỗ trợ resume bản đầu.
        # Generator riêng của loader được checkpoint chụp lại, nên resume tiếp tục đúng thứ tự ngẫu nhiên.
        DataLoader(Subset(dataset, train_indices), batch_size=data['batch_size'],
                   shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(config['seed'])),
        # Validation không shuffle; dùng generator khác (seed+1) để không dùng chung luồng RNG train.
        # Dù không shuffle, DataLoader vẫn có thể dùng generator để tạo base seed khi mở iterator.
        DataLoader(Subset(dataset, val_indices), batch_size=data['batch_size'],
                   shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(config['seed'] + 1)),
    )
