# ESC-50 CNN (PyTorch) Project Template

This project trains a configurable CNN on the **ESC-50** environmental sound dataset using **log-mel spectrograms** as input.

## Why this structure?
- **Reproducibility**: every run snapshots the config and writes metrics + artifacts into `reports/<run_id>/`.
- **Leakage-safe split**: ESC-50 is pre-arranged into **5 folds**; by default we use 3/1/1 folds → **6:2:2 per class** and keep fragments from the same original source inside one fold.
- **Parametric model**: the CNN is defined from config lists and built with `nn.ModuleList`.

## Folder layout
- `config/` : YAML configs
- `data/`
  - `audio/` : put ESC-50 wav files here (`{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`)
  - `meta/` : put `esc50.csv` here
  - `processed/` : cached spectrograms and split manifests
- `models/` : checkpoints (best/last) per run
- `reports/` : logs, metrics, inference summary CSV, confusion matrix, etc.
- `notebooks/` : experiments / EDA
- `src/` : package code

## Quickstart

### 1) Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download ESC-50 and place files
Place:
- WAVs into `data/audio/`
- `esc50.csv` into `data/meta/`

### 3) Train
```bash
python -m src.cli.train --config config/default.yaml
```

### 4) Evaluate + create inference CSV on the test split
```bash
python -m src.cli.evaluate --run_dir reports/<run_id> --checkpoint best
```

> Tip: The evaluation writes `inference_summary.csv` with index = filename and probability columns.

## Notes
- If you prefer official 5-fold cross validation, set `split.method: kfold` and iterate folds.
