from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.loops import run_one_epoch
from src.utils.config import dump_yaml

@dataclass
class EarlyStopping:
    enabled: bool
    patience: int
    best: float = float("inf")
    bad_epochs: int = 0

    def step(self, metric: float) -> bool:
        """
        Returns True if should stop.
        """
        if not self.enabled:
            return False
        if metric < self.best:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )

def train_model(
    cfg: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    logger,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # snapshot config
    dump_yaml(cfg, run_dir / "config_snapshot.yaml")

    criterion = nn.CrossEntropyLoss()

    opt_name = cfg["train"]["optimizer"]["name"].lower()
    lr = float(cfg["train"]["optimizer"]["lr"])
    wd = float(cfg["train"]["optimizer"]["weight_decay"])

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    sch_cfg = cfg["train"]["scheduler"]
    scheduler = None
    if sch_cfg["name"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(sch_cfg["factor"]), patience=int(sch_cfg["patience"]), min_lr=float(sch_cfg["min_lr"])
        )

    mixed = bool(cfg["train"]["mixed_precision"]) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=mixed)

    es_cfg = cfg["train"]["early_stopping"]
    early = EarlyStopping(enabled=bool(es_cfg["enabled"]), patience=int(es_cfg["patience"]))

    metrics_path = run_dir / "metrics.csv"
    metrics_path.write_text("epoch,lr,train_loss,train_acc,val_loss,val_acc\n", encoding="utf-8")

    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    model.to(device)

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch} | lr={lr_now:.3e}")

        train_stats = run_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, mixed, desc="train"
        )
        val_stats = run_one_epoch(
            model, val_loader, criterion, None, device, None, mixed_precision=False, desc="val"
        )

        logger.info(
            f"  train loss={train_stats.loss:.4f} acc={train_stats.acc:.4f} | val loss={val_stats.loss:.4f} acc={val_stats.acc:.4f}"
        )

        # write metrics
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{lr_now:.8e},{train_stats.loss:.6f},{train_stats.acc:.6f},{val_stats.loss:.6f},{val_stats.acc:.6f}\n")

        # scheduler step
        if scheduler is not None:
            scheduler.step(val_stats.loss)

        # checkpoints
        save_checkpoint(last_ckpt, model, optimizer, epoch, cfg)
        if val_stats.loss <= early.best:
            save_checkpoint(best_ckpt, model, optimizer, epoch, cfg)
            logger.info("  ✓ saved best checkpoint")

        # early stopping
        if early.step(val_stats.loss):
            logger.info(f"Early stopping triggered (patience={early.patience}).")
            break

    # return summary
    return {
        "best_val_loss": early.best,
        "run_dir": str(run_dir),
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
    }
