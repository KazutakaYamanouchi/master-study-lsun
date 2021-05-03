from logging import getLogger
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn.utils import spectral_norm

logger = getLogger(__name__)
logger.debug('スクリプトを読み込みました。')


def init_xavier_uniform(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        if hasattr(layer, "weight"):
            torch.nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if hasattr(layer.bias, "data"):
                layer.bias.data.fill_(0)


class DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        prob_dropout: float = 0.3
    ):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size, stride, padding, bias=True)),
            nn.LeakyReLU(0.2),
        )
        self.main.apply(init_xavier_uniform)  # 重みの初期化

    def forward(self, inputs):
        return self.main(inputs)


class Discriminator(nn.Module):
    def __init__(
        self, nc: int
    ):
        super().__init__()
        logger.debug('Discriminatorのインスタンスを作成します。')

        self.blocks = nn.Sequential(
            # 128 -> 64
            DBlock(nc, 32, 3, stride=2, padding=1),
            # 64 -> 32
            DBlock(32, 64, 3, stride=2, padding=1),
            # 32 -> 16
            DBlock(64, 128, 3, stride=2, padding=1),
            # 16 -> 8
            DBlock(128, 256, 3, stride=2, padding=1),
            # 8 -> 4
            DBlock(256, 512, 3, stride=2, padding=1),
            # 4 -> 1
            DBlock(512, 512, 4, stride=1, padding=0),
        )

        self.real_fake = nn.Linear(512, 1)

    def forward(
        self, x, classes=None,
        detach: bool = False
    ):
        if detach:
            x = x.detach()
        x = self.blocks(x)
        h = x.view(-1, x.size(1))
        real_fake = self.real_fake(h)
        return real_fake
