from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_yaml
from src.data.metadata import load_esc50_metadata
from src.data.split import SplitConfig, make_splits
from src.data.features import FeatureConfig, FeatureCache
from src.data.dataset import Esc50Dataset, AugmentConfig

from src.eval.evaluate import load_model_from_checkpoint, inference, write_inference_summary, compute_and_save_metrics, plot_confusion_matrix

def _pick_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="reports/<run_id>")
    ap.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--config", type=str, default=None, help="Optional: override config path (else use run_dir/config_snapshot.yaml)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = Path(args.config) if args.config else (run_dir / "config_snapshot.yaml")
    cfg = load_yaml(cfg_path)

    device = _pick_device(cfg["project"]["device"])

    ckpt_path = run_dir / f"{args.checkpoint}.pt"
    model, cfg_from_ckpt = load_model_from_checkpoint(ckpt_path, device)

    # data
    df = load_esc50_metadata(Path(cfg["paths"]["meta_csv"]))
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

    # features + cache (same as train)
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

    # no augmentation for test
    aug_cfg = AugmentConfig(
        enabled=False, time_mask_param=0, freq_mask_param=0, gain_db=0.0, time_shift_pct=0.0
    )

    audio_dir = Path(cfg["paths"]["audio_dir"])
    ds_test = Esc50Dataset(splits["test"], audio_dir, feat_cfg, cache, aug_cfg, is_train=False)

    dl_cfg = cfg["dataloader"]
    test_loader = DataLoader(
        ds_test,
        batch_size=int(dl_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg["pin_memory"]) and device.type == "cuda",
    )

    # inference
    class_names = df.sort_values("target")["category"].drop_duplicates().tolist()
    filenames, y_true, probs = inference(model, test_loader, device, num_classes=int(cfg["model"]["num_classes"]))

    out_csv = run_dir / "inference_summary.csv"
    summary_df = write_inference_summary(out_csv, filenames, y_true, probs, class_names=class_names)

    y_pred = probs.argmax(axis=1)
    metrics = compute_and_save_metrics(run_dir / "test_metrics.json", y_true, y_pred, class_names=class_names)

    plot_confusion_matrix(run_dir / "confusion_matrix.png", y_true, y_pred, class_names=None)

    print("Wrote:", out_csv)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
