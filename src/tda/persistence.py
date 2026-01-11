from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ripser import ripser
from persim import PersistenceImager


@dataclass(frozen=True)
class PHResult:
    diagrams: list[np.ndarray]  # diagrams[d] = [num_pts, 2] for dimension d


def compute_ph(X: np.ndarray, maxdim: int = 1) -> PHResult:
    """
    X: [N, D] point cloud
    """
    out = ripser(X, maxdim=maxdim)
    return PHResult(diagrams=out["dgms"])


def persistence_image(dgm: np.ndarray, pixel_size: float = 0.05) -> np.ndarray:
    """
    dgm: [K,2] birth-death for a single homology dimension.
    Returns 2D image as numpy array.
    """
    pimgr = PersistenceImager(pixel_size=pixel_size)
    pimgr.fit(dgm)
    img = pimgr.transform(dgm)
    return img