from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

from src.model.cnn import CNNClassifier, CNNConfig

def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    mcfg = cfg["model"]
    cnn_cfg = CNNConfig(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        channels=list(mcfg["conv_blocks"]["channels"]),
        kernel_sizes=list(mcfg["conv_blocks"]["kernel_sizes"]),
        pool_sizes=list(mcfg["conv_blocks"]["pool_sizes"]),
        conv_dropout=float(mcfg["conv_blocks"]["dropout"]),
        head_hidden_dim=int(mcfg["head"]["hidden_dim"]),
        head_dropout=float(mcfg["head"]["dropout"]),
    )
    model = CNNClassifier(cnn_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg

@torch.no_grad()
def inference(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    filenames = []
    y_true = []
    probs = []

    for x, y, fn in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()

        probs.append(p)
        y_true.append(y.numpy())
        filenames.extend(list(fn))

    probs = np.concatenate(probs, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return filenames, y_true, probs

def write_inference_summary(
    out_csv: Path,
    filenames: List[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str] | None = None,
) -> pd.DataFrame:
    y_pred = probs.argmax(axis=1)
    df = pd.DataFrame(index=filenames)
    df.index.name = "filename"
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    # probability columns
    for c in range(probs.shape[1]):
        col = f"prob_{c}" if class_names is None else f"prob_{c}_{class_names[c]}"
        df[col] = probs[:, c]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    return df

def compute_and_save_metrics(
    out_json: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] | None = None,
) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # also dump a full report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    (out_json.parent / "classification_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics

def plot_confusion_matrix(
    out_png: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] | None = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # avoid clutter: label only if few classes
    if class_names is not None and len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
