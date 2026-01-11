from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

import torch
from torch.utils.data import Dataset
import torchaudio

from src.data.metadata import resolve_audio_path
from src.data.features import FeatureConfig, LogMelExtractor, FeatureCache


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool
    time_mask_param: int
    freq_mask_param: int
    gain_db: float
    time_shift_pct: float


class Esc50Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        feature_cfg: FeatureConfig,
        cache: Optional[FeatureCache],
        augment: AugmentConfig,
        is_train: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.feature_cfg = feature_cfg
        self.cache = cache
        self.is_train = is_train

        self.extractor = LogMelExtractor(feature_cfg)

        self.augment = augment
        self._time_mask = torchaudio.transforms.TimeMasking(time_mask_param=augment.time_mask_param)
        self._freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=augment.freq_mask_param)

    def __len__(self) -> int:
        return len(self.df)

    def _load_wav(self, filename: str) -> torch.Tensor:
        """
        Load a WAV file using soundfile (robust on Windows without torchcodec).
        Returns: wav tensor [1, T] float32
        """
        path = resolve_audio_path(self.audio_dir, filename)
        # soundfile: data shape -> [T] or [T, C]
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
        # -> torch [C, T]
        wav = torch.from_numpy(data.T)  # [C, T]

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.feature_cfg.sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.feature_cfg.sample_rate
            )

        # trim/pad to fixed duration
        target_len = int(round(self.feature_cfg.sample_rate * self.feature_cfg.duration_sec))
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        elif wav.shape[1] < target_len:
            pad = target_len - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))

        return wav

    def _apply_aug(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [1, n_mels, frames]
        if not (self.augment.enabled and self.is_train):
            return feat

        x = feat
        x = self._freq_mask(x)
        x = self._time_mask(x)

        # random gain in log domain = add constant (approx)
        if self.augment.gain_db > 0:
            gain = (torch.rand(1).item() * 2 - 1) * self.augment.gain_db
            x = x + (gain / 20.0)

        # time shift
        if self.augment.time_shift_pct > 0:
            _, _, frames = x.shape
            max_shift = int(frames * self.augment.time_shift_pct)
            if max_shift > 0:
                shift = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,)).item())
                x = torch.roll(x, shifts=shift, dims=2)

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        filename = row["filename"]
        y = int(row["target"])

        feat = None
        if self.cache is not None:
            feat = self.cache.load(filename)

        if feat is None:
            wav = self._load_wav(filename)
            feat = self.extractor(wav)
            if self.cache is not None:
                self.cache.save(filename, feat)

        feat = self._apply_aug(feat)
        return feat, y, filename