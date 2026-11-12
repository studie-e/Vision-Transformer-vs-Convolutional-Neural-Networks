"""Count a model and, optionally, search a ResNet width to match a ViT budget."""

import argparse
import json
from pathlib import Path

import yaml

from src.models.resnet import CifarResNet, count_parameters
from src.train import factory


# Hàm main: đếm tham số trainable của mô hình; tùy chọn tìm width ResNet gần ngân sách ViT.
# Nhận --config và --target từ terminal; in JSON, không sửa cấu hình và không huấn luyện.
def main():
    # Khai báo giao diện lệnh, cho phép xem cách dùng bằng --help.
    parser = argparse.ArgumentParser(description=__doc__)
    # Mặc định đọc cấu hình pilot có đủ model.factory và model.kwargs.
    parser.add_argument('--config', default='configs/train.yaml')
    # target là số tham số ViT thật Lan Anh đã đếm; không phải accuracy hay số ảnh train.
    parser.add_argument('--target', type=int, help='Actual trainable parameter count from Lan Anh')
    # Đọc các tùy chọn thành args.config và args.target.
    args = parser.parse_args()
    # Nạp YAML cấu hình; safe_load tạo dict thay vì khởi tạo đối tượng Python tùy ý.
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    # factory tìm hàm tạo model, (config) tạo model, count_parameters đếm phần tử được phép học.
    # JSON ghi cả tên factory để biết số vừa đếm thuộc mô hình nào.
    print(json.dumps({'factory': config['model']['factory'],
                      'parameters': count_parameters(factory(config['model']['factory'])(config))}))
    # Không truyền --target thì dừng ở việc đếm; chưa giả định đã có số tham số ViT.
    if args.target is not None:
        # Ngân sách mục tiêu phải dương; parser.error in lỗi kèm cách dùng lệnh.
        if args.target <= 0:
            parser.error('--target must be positive')
        # Giữ danh sách width, số tham số và độ lệch để chọn ứng viên gần nhất.
        candidates = []
        # Thử width từ 4 tới 128; chỉ thay độ rộng, giữ mặc định 3 block/stage và 10 lớp.
        for width in range(4, 129):
            # Tạo từng ResNet trên CPU để đếm, không cần forward/backward hay dữ liệu.
            parameters = count_parameters(CifarResNet(width=width))
            # Độ lệch tương đối theo quy ước nhóm: chênh lệch chia số lớn hơn của hai mô hình.
            gap = abs(parameters - args.target) / max(parameters, args.target)
            # Lưu một ứng viên; gap=0.05 nghĩa là lệch 5% theo công thức trên.
            candidates.append({'width': width, 'parameters': parameters, 'relative_gap': gap})
        # key chọn theo relative_gap nhỏ nhất, không theo width nhỏ nhất.
        best = min(candidates, key=lambda row: row['relative_gap'])
        # **best trải các trường dict; thêm cờ đạt ngưỡng 10% nội bộ, chưa tự chốt width cuối.
        print(json.dumps({**best, 'within_10_percent': best['relative_gap'] <= 0.10}))


if __name__ == '__main__':
    # Import module không tự đếm; chỉ khi chạy python -m src.inspect_model mới gọi main.
    main()
