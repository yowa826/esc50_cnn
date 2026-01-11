from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio

from src.utils.config import dict_hash

@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int
    duration_sec: float
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    f_min: float
    f_max: Optional[float]
    log_eps: float
    normalize: str  # "per_sample" | "none"

def _to_dict(cfg: FeatureConfig) -> Dict[str, Any]:
    return {
        "sample_rate": cfg.sample_rate,
        "duration_sec": cfg.duration_sec,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "win_length": cfg.win_length,
        "n_mels": cfg.n_mels,
        "f_min": cfg.f_min,
        "f_max": cfg.f_max,
        "log_eps": cfg.log_eps,
        "normalize": cfg.normalize,
    }

class LogMelExtractor:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
            normalized=False,
        )

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [1, T]
        spec = self.mel(wav)  # 典型: [1, n_mels, frames]
        spec = torch.log(spec + self.cfg.log_eps)

        if self.cfg.normalize == "per_sample":
            mean = spec.mean()
            std = spec.std().clamp_min(1e-8)
            spec = (spec - mean) / std

        # Conv2d用に [1, n_mels, frames] を返す
        # もし環境差で [n_mels, frames] が返る場合だけ補正
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        return spec
    
class FeatureCache:
    def __init__(self, cache_root: Path, feature_cfg: FeatureConfig):
        self.cache_root = cache_root
        self.feature_cfg = feature_cfg
        self.version = dict_hash(_to_dict(feature_cfg))
        self.dir = cache_root / "features" / self.version
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, filename: str) -> Path:
        return self.dir / (filename.replace(".wav", ".pt"))

    def load(self, filename: str) -> Optional[torch.Tensor]:
        p = self.path_for(filename)
        if p.exists():
            return torch.load(p, map_location="cpu")
        return None

    def save(self, filename: str, feat: torch.Tensor) -> None:
        p = self.path_for(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feat.cpu(), p)
