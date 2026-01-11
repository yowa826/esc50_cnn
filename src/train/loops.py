from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class EpochStats:
    loss: float
    acc: float

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    mixed_precision: bool,
    desc: str,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y, _fn in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=mixed_precision):
            logits = model(x)
            loss = criterion(logits, y)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits.detach(), y) * bs
        n += bs

        pbar.set_postfix(loss=total_loss / max(1, n), acc=total_acc / max(1, n))

    return EpochStats(loss=total_loss / max(1, n), acc=total_acc / max(1, n))
