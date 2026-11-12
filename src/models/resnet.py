"""CIFAR-style ResNet with configurable width and projection shortcuts.

Architecture source: He et al., https://arxiv.org/abs/1512.03385 (section 4.2).
This implementation uses learned 1x1 projections when shapes change (option B),
instead of the parameter-free option A in the paper's CIFAR experiment.
"""

from torch import Tensor, nn


# BasicBlock là một khối residual: nhánh học F(x) cộng với nhánh tắt shortcut(x), rồi ReLU.
# nn.Module giúp PyTorch tự quản lý các lớp con, tham số, train/eval và state_dict.
class BasicBlock(nn.Module):
    # Hàm __init__: dựng các lớp của một block, chưa chạy ảnh qua mạng.
    # in_channels/out_channels là số kênh vào/ra; stride=2 giảm chiều cao/rộng khi đổi stage.
    # Kết quả là đối tượng block có nhánh residual và shortcut cộng được với nhau.
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        # Khởi tạo phần quản lý module của lớp cha trước khi gắn các lớp con.
        super().__init__()
        # Sequential thực thi các lớp theo thứ tự để tạo nhánh F(x).
        self.residual = nn.Sequential(
            # Conv 3×3: đổi số kênh và có thể giảm kích thước; padding=1 giữ H/W khi stride=1.
            # bias=False vì ngay sau đó BatchNorm đã có tham số dịch chuyển học được.
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            # Chuẩn hóa theo kênh; train dùng thống kê batch, eval dùng thống kê tích lũy đã lưu.
            nn.BatchNorm2d(out_channels),
            # ReLU tạo phi tuyến: max(0, x), giúp mạng học được ánh xạ phức tạp hơn tuyến tính.
            nn.ReLU(),
            # Conv thứ hai giữ nguyên số kênh/kích thước của đầu ra conv thứ nhất.
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # Chưa ReLU ở đây: kết quả nhánh residual sẽ được cộng shortcut trước.
            nn.BatchNorm2d(out_channels),
        )
        # Mặc định shortcut(x)=x, không thêm tham số hay thay kích thước tensor.
        self.shortcut = nn.Identity()
        # Cộng từng phần tử yêu cầu hai nhánh cùng shape; đổi stride hoặc kênh thì phải chiếu lại x.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Conv 1×1 đổi số kênh, dùng cùng stride với nhánh residual để khớp H/W.
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                # BatchNorm cho nhánh projection; bản này dùng shortcut có tham số (option B).
                nn.BatchNorm2d(out_channels),
            )
        # ReLU dùng sau phép cộng hai nhánh trong forward.
        self.relu = nn.ReLU()

    # Hàm forward của BasicBlock: tính y=ReLU(F(x)+shortcut(x)).
    # Nhận tensor [B,C_in,H,W], trả tensor có C_out và kích thước tùy stride.
    # Phép cộng tạo đường truyền trực tiếp cho tín hiệu/gradient qua shortcut khi học.
    def forward(self, x: Tensor) -> Tensor:
        # Hai nhánh nhận cùng x; cộng theo phần tử, không nối kênh như torch.cat.
        return self.relu(self.residual(x) + self.shortcut(x))


# CifarResNet ghép stem -> 3 stage BasicBlock -> global average pooling -> lớp phân loại.
# Đây là bản cho ảnh nhỏ, không dùng stem 7×7/max-pool kiểu ResNet ImageNet.
class CifarResNet(nn.Module):
    # Hàm __init__: dựng cả mô hình và khởi tạo một số loại tham số.
    # width=16 cho các stage 16/32/64 kênh; blocks_per_stage=3 cho 9 BasicBlock; num_classes=10.
    # Tạo cấu trúc nn.Module, chưa huấn luyện và chưa tự đưa lên GPU.
    def __init__(self, width: int = 16, blocks_per_stage: int = 3, num_classes: int = 10):
        # Để PyTorch đăng ký đúng các layer và tham số gắn vào self bên dưới.
        super().__init__()
        # Không có kênh/block/lớp phân loại bằng 0 hoặc âm.
        if min(width, blocks_per_stage, num_classes) < 1:
            raise ValueError('width, blocks_per_stage and num_classes must be positive')
        # Stem đổi ảnh RGB 3 kênh sang width kênh; stride=1 giữ ảnh nhỏ không bị giảm quá sớm.
        self.stem = nn.Sequential(
            # Với input [B,3,32,32], output stem là [B,width,32,32].
            nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        # Danh sách tạm để gom các block theo thứ tự rồi chuyển thành nn.Sequential.
        stages = []
        # Số kênh mà block kế tiếp nhận, ban đầu chính là đầu ra của stem.
        in_channels = width
        # enumerate cho cả chỉ số stage 0/1/2 và số kênh tương ứng width/2width/4width.
        for stage, out_channels in enumerate((width, width * 2, width * 4)):
            # Mỗi stage có blocks_per_stage khối, các khối trong cùng stage giữ cùng số kênh ra.
            for block in range(blocks_per_stage):
                # Chỉ block đầu stage 1/2 downsample; stage 0 và các block còn lại dùng stride=1.
                stride = 2 if stage > 0 and block == 0 else 1
                # Block tự chọn identity/projection shortcut theo số kênh và stride.
                stages.append(BasicBlock(in_channels, out_channels, stride))
                # Đầu ra block vừa tạo trở thành đầu vào block tiếp theo.
                in_channels = out_channels
        # *stages trải list thành các đối số; Sequential đăng ký block để optimizer thấy tham số.
        self.stages = nn.Sequential(*stages)
        # Lấy trung bình trên toàn H/W của từng kênh => [B,4width,1,1], không có tham số học.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Linear đổi vector đặc trưng 4width chiều thành num_classes logits cho từng ảnh.
        self.head = nn.Linear(in_channels, num_classes)
        # Duyệt cả các lớp nằm sâu trong block để khởi tạo trọng số, không chỉ các lớp trực tiếp của self.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming normal phù hợp mạng dùng ReLU; fan_out dùng số kết nối đi ra khi tính phương sai.
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                # Hệ số scale (gamma) bắt đầu ở 1 và hệ số shift (beta) ở 0; vẫn học được khi train.
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # Hàm forward của CifarResNet: biến batch ảnh thành logits dùng cho CrossEntropyLoss.
    # Ví dụ width=16, ảnh 32×32: [B,3,32,32] -> [B,16,32,32] -> [B,64,8,8]
    # -> [B,64,1,1] -> [B,64] -> [B,10]. B là batch size, không bị flatten mất.
    def forward(self, x: Tensor) -> Tensor:
        # Đọc biểu thức từ trong ra ngoài: stem, stages, pool, flatten từ chiều 1, rồi head.
        # Không softmax: logits là đầu vào đúng cho hàm cross_entropy trong trainer.
        return self.head(self.pool(self.stages(self.stem(x))).flatten(1))


# Hàm count_parameters: đếm số giá trị vô hướng của các tham số có thể học.
# Đầu vào: một nn.Module; đầu ra: số nguyên, không phải số layer hoặc dung lượng file.
# Không tính buffer BatchNorm (running_mean/var) hay tham số đã requires_grad=False.
def count_parameters(model: nn.Module) -> int:
    # numel() đếm phần tử từng tensor; cộng toàn bộ tensor đang bật requires_grad.
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# Hàm build_model: đầu nối để trainer tạo ResNet từ YAML theo quy ước factory chung.
# Nhận dict config đầy đủ, lấy model.kwargs; trả CifarResNet mới, chưa train.
def build_model(config: dict) -> nn.Module:
    # ** trải dict thành đối số có tên, ví dụ width=16, blocks_per_stage=3, num_classes=10.
    return CifarResNet(**config['model']['kwargs'])
