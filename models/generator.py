
from logging import getLogger
from typing import Tuple, Union

import torch
import torch.nn as nn

logger = getLogger(__name__)
logger.debug('スクリプトを読み込みました。')


def init_xavier_uniform(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        if hasattr(layer, "weight"):
            torch.nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if hasattr(layer.bias, "data"):
                layer.bias.data.fill_(0)


class GBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False)
        self.conv.apply(init_xavier_uniform)  # 重みの初期化

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Generator(nn.Module):
    def __init__(
        self, nz: int, nc: int,
    ):
        super().__init__()
        logger.debug('Generatorのインスタンスを作成します。')
        self.blocks = nn.ModuleList([
            # 1 -> 4
            GBlock(
                nz, 256, 4,
                stride=1, padding=0),
            # 4 -> 8
            GBlock(
                256, 128, 4,
                stride=2, padding=1),
            # 8 -> 16
            GBlock(
                128, 64, 4,
                stride=2, padding=1),
            # 16 -> 32
            GBlock(
                64, 32, 4,
                stride=2, padding=1),
        ])

        self.last = nn.Conv2d(32, nc, 1, stride=1, padding=0)

    def forward(
        self, z: torch.Tensor,
    ):
        x = z.view(-1, z.size(1), 1, 1)
        for block in self.blocks:
            x = block(x)
        return self.last(x)
