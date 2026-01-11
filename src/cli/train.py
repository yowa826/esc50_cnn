from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_yaml, dump_yaml
from src.utils.seed import set_seed
from src.utils.logging import setup_logging

from src.data.metadata import load_esc50_metadata
from src.data.split import SplitConfig, make_splits, save_split_manifests
from src.data.features import FeatureConfig, FeatureCache
from src.data.dataset import Esc50Dataset, AugmentConfig

from src.model.cnn import CNNClassifier, CNNConfig
from src.train.trainer import train_model

def _pick_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    set_seed(int(cfg["project"]["seed"]))

    device = _pick_device(cfg["project"]["device"])

    # run id
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg["paths"]["reports_dir"]) / run_id
    log = setup_logging(run_dir / "train.log")
    log.info(f"Run: {run_id} | device={device}")

    # data
    df = load_esc50_metadata(Path(cfg["paths"]["meta_csv"]))
    log.info(f"Loaded meta: {len(df)} rows, {df['target'].nunique()} classes")

    s_cfg = cfg["split"]
    split_cfg = SplitConfig(
        method=s_cfg["method"],
        train_folds=list(s_cfg.get("train_folds", [])),
        val_folds=list(s_cfg.get("val_folds", [])),
        test_folds=list(s_cfg.get("test_folds", [])),
        kfold_num_folds=int(s_cfg.get("kfold", {}).get("num_folds", 5)),
        kfold_test_fold=int(s_cfg.get("kfold", {}).get("test_fold", 1)),
        kfold_val_fold=int(s_cfg.get("kfold", {}).get("val_fold", 2)),
        random_seed=int(cfg["project"]["seed"]),
    )
    splits = make_splits(df, split_cfg)

    # save manifests (useful for reproducibility)
    split_dir = Path(cfg["paths"]["processed_dir"]) / "splits" / run_id
    save_split_manifests(splits, split_dir)
    log.info(f"Saved split manifests: {split_dir}")

    # features + cache
    f = cfg["features"]
    a = cfg["audio"]
    feat_cfg = FeatureConfig(
        sample_rate=int(a["sample_rate"]),
        duration_sec=float(a["duration_sec"]),
        n_fft=int(f["n_fft"]),
        hop_length=int(f["hop_length"]),
        win_length=int(f["win_length"]),
        n_mels=int(f["n_mels"]),
        f_min=float(f["f_min"]),
        f_max=None if f["f_max"] in (None, "null") else float(f["f_max"]),
        log_eps=float(f["log_eps"]),
        normalize=str(f["normalize"]),
    )
    cache = None
    if bool(f["cache"]["enabled"]):
        cache = FeatureCache(Path(cfg["paths"]["processed_dir"]), feat_cfg)
        log.info(f"Feature cache enabled: {cache.dir}")

    aug = cfg["augment"]
    aug_cfg = AugmentConfig(
        enabled=bool(aug["enabled"]),
        time_mask_param=int(aug["time_mask_param"]),
        freq_mask_param=int(aug["freq_mask_param"]),
        gain_db=float(aug["gain_db"]),
        time_shift_pct=float(aug["time_shift_pct"]),
    )

    audio_dir = Path(cfg["paths"]["audio_dir"])

    ds_train = Esc50Dataset(splits["train"], audio_dir, feat_cfg, cache, aug_cfg, is_train=True)
    ds_val = Esc50Dataset(splits["val"], audio_dir, feat_cfg, cache, aug_cfg, is_train=False)

    dl_cfg = cfg["dataloader"]
    train_loader = DataLoader(
        ds_train,
        batch_size=int(dl_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg["pin_memory"]) and device.type == "cuda",
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=int(dl_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg["pin_memory"]) and device.type == "cuda",
    )

    # model
    m = cfg["model"]
    cnn_cfg = CNNConfig(
        in_channels=int(m["in_channels"]),
        num_classes=int(m["num_classes"]),
        channels=list(m["conv_blocks"]["channels"]),
        kernel_sizes=list(m["conv_blocks"]["kernel_sizes"]),
        pool_sizes=list(m["conv_blocks"]["pool_sizes"]),
        conv_dropout=float(m["conv_blocks"]["dropout"]),
        head_hidden_dim=int(m["head"]["hidden_dim"]),
        head_dropout=float(m["head"]["dropout"]),
    )
    model = CNNClassifier(cnn_cfg)

    # train
    summary = train_model(cfg, model, train_loader, val_loader, device, run_dir, log)
    (run_dir / "train_summary.json").write_text(__import__("json").dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Training done.")
    log.info(f"Summary: {summary}")

if __name__ == "__main__":
    main()
