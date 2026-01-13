"""Model definitions for 1D-CNN / ResNet-style backbones."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, padding: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, base_channels: int = 64, num_blocks: int = 3):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels, num_classes),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.blocks(out)
        out = self.head(out)
        return out.squeeze(-1)


def get_model(in_channels: int, num_classes: int = 1, architecture: str = "resnet"):
    if architecture == "resnet":
        return ResNet1D(in_channels=in_channels, num_classes=num_classes)
    if architecture == "cnn":
        return nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, padding=3),
            ConvBlock(64, 128, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    raise ValueError(f"Unknown architecture: {architecture}")
