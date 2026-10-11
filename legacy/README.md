# Legacy code (kept for reference)

Nothing here is used by the current pipeline (`manuai/` and `notebooks/`). It is kept unchanged so the capstone stays reproducible and its history stays readable.

## `v1/`: AIML339 capstone (Aug–Oct 2025)

This is the code and results submitted for AIML339 at Victoria University of Wellington (supervisor Prof. W. Bastiaan Kleijn). The git tag **`v1-capstone`** marks the repository exactly as it was at submission.

| File | What it is |
|---|---|
| `download_data.py` | Xeno-canto (+ Kaggle) download, 22.05 kHz |
| `preprocess_data.py` | 4 s windows (50 % overlap), librosa log-mel images. ⚠️ It deletes its own output folders, so don't run it in the repo root |
| `inference.py` | v1 inference (10-class map, tkinter file picker) |
| `bird_types.yaml` | v1 class list |
| `training_metrics.json`, `confusion_matrix.png` | **Headline result: 72.6 % accuracy, macro-F1 ≈ 0.73, AUC 0.97** (20 species) |
| `logs/recordings_data.csv` | Xeno-canto metadata for the v1 downloads |

The v1 split was made **per segment**, so recordings leaked between train and test. Its numbers are therefore not comparable with v2.1's grouped split.

These scripts expect to run from the folder they were written in, using relative paths such as `bird_types.yaml`. To re-run one, `cd legacy/v1` first.

## `art/`

`show_weights.py` renders a poster from model weights. It's a side project; the outputs live in the ignored `art/` folder at the repo root.

## Notebooks

The first v2 attempt (Mar–Apr 2026) lives in [`../notebooks/legacy/`](../notebooks/legacy/): `data_download`, `preprocess_data`, `finetune` (ViT), `ast_finetune` (AST at 1024 frames) and `analyse_dataset`. `AGENTS.md` §7 explains what went wrong and how v2.1 fixed it.
