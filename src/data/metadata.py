from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class Esc50Paths:
    audio_dir: Path
    meta_csv: Path

def load_esc50_metadata(meta_csv: Path) -> pd.DataFrame:
    """
    Loads `meta/esc50.csv` from the official ESC-50 release.
    Expected columns: filename, fold, target, category, esc10, src_file, take
    """
    df = pd.read_csv(meta_csv)
    required = {"filename", "fold", "target", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"meta csv missing columns: {missing}. got={list(df.columns)}")

    # normalize types
    df["fold"] = df["fold"].astype(int)
    df["target"] = df["target"].astype(int)
    df["filename"] = df["filename"].astype(str)

    return df

def resolve_audio_path(audio_dir: Path, filename: str) -> Path:
    p = audio_dir / filename
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    return p
