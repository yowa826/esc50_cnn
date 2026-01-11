from __future__ import annotations

import numpy as np
import torch


def mel_frames_to_pointcloud(feat: torch.Tensor, max_points: int = 200) -> np.ndarray:
    """
    feat: [1, n_mels, frames] (log-mel)
    returns point cloud: [P, n_mels] where each point = a time frame vector.
    Downsamples frames to max_points for speed.
    """
    if feat.dim() != 3:
        raise ValueError(f"expected [1,n_mels,frames], got {tuple(feat.shape)}")

    x = feat.squeeze(0)             # [n_mels, frames]
    x = x.transpose(0, 1)           # [frames, n_mels]
    X = x.detach().cpu().numpy()

    T = X.shape[0]
    if T > max_points:
        idx = np.linspace(0, T - 1, max_points).astype(int)
        X = X[idx]
    return X