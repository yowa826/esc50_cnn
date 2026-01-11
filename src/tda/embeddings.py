from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EmbeddingResult:
    X: np.ndarray          # [N, D]
    y: np.ndarray          # [N]
    filenames: np.ndarray  # [N]


class PenultimateExtractor(nn.Module):
    """
    Wrap a classifier model and return penultimate features.
    Assumes the model has a 'feature_extractor' + 'classifier' structure OR
    a method to get embeddings. If your CNN is a single forward, adapt here.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # --- Adaptation point ---
        # If your CNNClassifier has: self.backbone and self.head, use them.
        # Otherwise, a robust trick is to register a forward hook.
        self._feat = None
        self._hook_handle = None

    def hook_on(self, module: nn.Module):
        def _hook(_m, _inp, out):
            self._feat = out
        self._hook_handle = module.register_forward_hook(_hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model(x)  # hook captures features
        if self._feat is None:
            raise RuntimeError("Feature hook did not capture anything.")
        feat = self._feat
        # flatten: [B, ...] -> [B, D]
        return feat.flatten(1)


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hook_module: nn.Module,
) -> EmbeddingResult:
    model.eval().to(device)

    extractor = PenultimateExtractor(model).to(device)
    extractor.hook_on(hook_module)

    xs, ys, fns = [], [], []
    for x, y, fn in loader:
        x = x.to(device, non_blocking=True)
        emb = extractor(x).detach().cpu().numpy()
        xs.append(emb)
        ys.append(y.numpy())
        fns.append(np.array(fn))

    extractor.remove_hook()

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    filenames = np.concatenate(fns, axis=0)
    return EmbeddingResult(X=X, y=y, filenames=filenames)


def save_embeddings_npz(out_path: Path, res: EmbeddingResult, meta: Dict[str, str] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=res.X,
        y=res.y,
        filenames=res.filenames,
        meta=np.array([meta or {}], dtype=object),
    )