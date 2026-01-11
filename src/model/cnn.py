from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

@dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    num_classes: int
    channels: List[int]
    kernel_sizes: List[int]
    pool_sizes: List[int]
    conv_dropout: float
    head_hidden_dim: int
    head_dropout: float

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, pool: int, dropout: float):
        super().__init__()
        padding = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CNNClassifier(nn.Module):
    """
    Parametric CNN built from lists (ModuleList).
    Input: [B, 1, n_mels, frames]
    """
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        assert len(cfg.channels) == len(cfg.kernel_sizes) == len(cfg.pool_sizes), (
            "channels/kernel_sizes/pool_sizes must have same length"
        )

        blocks = []
        in_ch = cfg.in_channels
        for out_ch, k, p in zip(cfg.channels, cfg.kernel_sizes, cfg.pool_sizes):
            blocks.append(ConvBlock(in_ch, out_ch, k, p, cfg.conv_dropout))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, cfg.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.head_dropout) if cfg.head_dropout > 0 else nn.Identity(),
            nn.Linear(cfg.head_hidden_dim, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        x = self.gap(x)
        return self.head(x)
