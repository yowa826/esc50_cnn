from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SplitMethod = Literal["folds", "random_stratified", "kfold"]

@dataclass(frozen=True)
class SplitConfig:
    method: SplitMethod
    train_folds: List[int]
    val_folds: List[int]
    test_folds: List[int]
    kfold_num_folds: int = 5
    kfold_test_fold: int = 1
    kfold_val_fold: int = 2
    random_seed: int = 42

def make_splits(df: pd.DataFrame, cfg: SplitConfig) -> Dict[str, pd.DataFrame]:
    """
    Returns dict with keys train/val/test and values being subsets of df.
    """
    if cfg.method == "folds":
        train = df[df["fold"].isin(cfg.train_folds)].copy()
        val = df[df["fold"].isin(cfg.val_folds)].copy()
        test = df[df["fold"].isin(cfg.test_folds)].copy()
        return {"train": train, "val": val, "test": test}

    if cfg.method == "kfold":
        # pick folds by index
        test = df[df["fold"] == cfg.kfold_test_fold].copy()
        val = df[df["fold"] == cfg.kfold_val_fold].copy()
        train = df[~df["fold"].isin([cfg.kfold_test_fold, cfg.kfold_val_fold])].copy()
        return {"train": train, "val": val, "test": test}

    if cfg.method == "random_stratified":
        # Warning: can introduce leakage because ESC-50 has multiple fragments from same source clip.
        # Use folds for leakage-safe split unless you explicitly want a random split.
        y = df["target"].values
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.random_seed)
        idx_trainval, idx_test = next(sss1.split(df, y))
        df_trainval = df.iloc[idx_trainval].copy()
        df_test = df.iloc[idx_test].copy()

        y_tv = df_trainval["target"].values
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.random_seed)
        idx_train, idx_val = next(sss2.split(df_trainval, y_tv))
        train = df_trainval.iloc[idx_train].copy()
        val = df_trainval.iloc[idx_val].copy()
        return {"train": train, "val": val, "test": df_test}

    raise ValueError(f"Unknown split method: {cfg.method}")

def save_split_manifests(splits: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, sdf in splits.items():
        sdf.to_csv(out_dir / f"{k}.csv", index=False)
