"""Epoch-boundary checkpoints using state_dict, including random generators.

Sources: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
https://docs.pytorch.org/docs/2.11/notes/randomness.html
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


# Hàm random_state: chụp vị trí hiện tại của các bộ sinh số ngẫu nhiên để resume.
# Đầu vào: các DataLoader theo thứ tự train, val; đầu ra: dict có thể lưu trong checkpoint.
# Seed chỉ là điểm xuất phát; trạng thái RNG cho biết đã đi tới đâu sau nhiều batch/epoch.
def random_state(loaders):
    # NumPy trả tuple gồm tên thuật toán, mảng trạng thái, vị trí và thông tin bộ đệm Gaussian.
    numpy_state = np.random.get_state()
    # Lưu riêng từng RNG vì seed Python không tự điều khiển NumPy, PyTorch hay generator của loader.
    return {
        # random.getstate() lấy trạng thái đầy đủ của thư viện random trong Python.
        'python': random.getstate(),
        # Chuyển mảng NumPy thành list để checkpoint chỉ chứa tensor và kiểu dữ liệu cơ bản.
        # *numpy_state[2:] trải các phần còn lại của tuple, không bỏ thông tin bộ sinh.
        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        # RNG mặc định CPU của PyTorch, ví dụ dùng cho các phép tạo tensor ngẫu nhiên.
        'torch': torch.get_rng_state(),
        # GPU có RNG riêng, ví dụ dropout trên CUDA; không có CUDA thì lưu danh sách rỗng.
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # Loader có thể có generator riêng cho shuffle/base seed; phải chụp nó ngoài RNG mặc định.
        # None ghi rõ loader không dùng generator riêng, để resume kiểm tra cấu trúc tương ứng.
        'loaders': [loader.generator.get_state() if loader.generator else None for loader in loaders],
    }


# Hàm restore_random_state: đặt lại các RNG đúng vị trí đã chụp bởi random_state.
# Đầu vào: dict trong checkpoint và các loader đúng thứ tự cũ. Không trả dữ liệu.
# Phải gọi sau khi tạo/nạp model, optimizer và loader, trước batch tiếp theo của run resume.
def restore_random_state(state, loaders):
    # Khôi phục RNG Python, không chỉ đặt lại seed ban đầu.
    random.setstate(state['python'])
    # Tách tuple NumPy đã lưu thành từng thành phần để dựng lại định dạng NumPy yêu cầu.
    name, keys, pos, has_gauss, cached = state['numpy']
    # keys trước đó lưu dạng list; đổi lại uint32 để np.random.set_state nhận đúng kiểu.
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    # API set_rng_state nhận tensor byte ở CPU, nên .cpu() bảo đảm đúng thiết bị.
    torch.set_rng_state(state['torch'].cpu())
    # Chỉ nạp RNG CUDA khi checkpoint có lưu và môi trường hiện tại có CUDA.
    if state['cuda'] and torch.cuda.is_available():
        # Danh sách trạng thái tương ứng các thiết bị CUDA; không phải tensor trọng số mô hình.
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    # Ghép loader hiện tại với RNG cũ; strict=True phát hiện số loader hai phía không khớp.
    for loader, saved in zip(loaders, state['loaders'], strict=True):
        # Tránh resume một loader có generator riêng bằng trạng thái của loader từng không có, hoặc ngược lại.
        if (loader.generator is None) != (saved is None):
            raise ValueError('DataLoader generator differs from checkpoint')
        if saved is not None:
            # Tiếp tục vị trí shuffle/base seed; nếu chỉ manual_seed lại thì dễ lặp thứ tự của epoch đầu.
            loader.generator.set_state(saved.cpu())


# Hàm save_checkpoint: ghi dict trạng thái ra đĩa qua file tạm, rồi thay file đích.
# Đầu vào: Path tới best.pt/last.pt và state; không trả dữ liệu. Thư mục cha phải có sẵn.
# Cách này tránh để file đích chứa nội dung ghi dở; vẫn cần tránh hai tiến trình ghi cùng run.
def save_checkpoint(path: Path, state: dict):
    # Ví dụ last.pt -> last.tmp: giữ file last.pt trước đó trong lúc đang ghi bản mới.
    temporary = path.with_suffix('.tmp')
    # Tuần tự hóa tensor/state_dict, optimizer, scheduler, RNG và metadata vào file tạm.
    torch.save(state, temporary)
    # Chỉ thay file đích sau khi torch.save thành công; lỗi giữa lúc save không ghi đè bản đích cũ.
    os.replace(temporary, path)


# Hàm load_checkpoint: đọc checkpoint từ đĩa, trả dict cho fit khôi phục từng thành phần.
# Đầu vào: đường dẫn checkpoint của nhóm; không tạo lại cả đối tượng nn.Module từ file.
def load_checkpoint(path):
    # Tensor/primitive-only checkpoints; do not enable arbitrary pickle objects.
    # map_location='cpu' cho phép đọc cả file từng lưu trên GPU khi đang kiểm tra bằng CPU.
    # weights_only=True giới hạn loại đối tượng được nạp; dict vẫn chứa được optimizer/RNG cơ bản.
    # Tên weights_only không có nghĩa chỉ trả trọng số model hoặc tự chọn best checkpoint.
    return torch.load(path, map_location='cpu', weights_only=True)
