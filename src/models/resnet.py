"""CIFAR-style ResNet with configurable width and projection shortcuts.

Architecture source: He et al., https://arxiv.org/abs/1512.03385 (section 4.2).
This implementation uses learned 1x1 projections when shapes change (option B),
instead of the parameter-free option A in the paper's CIFAR experiment.
"""

from torch import Tensor, nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.residual(x) + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, width: int = 16, blocks_per_stage: int = 3, num_classes: int = 10):
        super().__init__()
        if min(width, blocks_per_stage, num_classes) < 1:
            raise ValueError('width, blocks_per_stage and num_classes must be positive')
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        stages = []
        in_channels = width
        for stage, out_channels in enumerate((width, width * 2, width * 4)):
            for block in range(blocks_per_stage):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.pool(self.stages(self.stem(x))).flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(config: dict) -> nn.Module:
    return CifarResNet(**config['model']['kwargs'])
