# ManūAI 🦜 — New Zealand bird sound classification

ManūAI identifies New Zealand birds from their calls. (*Manu* is te reo Māori for "bird".) Audio is cut into 5-second segments and turned into log-mel spectrograms. A pretrained transformer, fine-tuned with LoRA, then predicts the species.

The long-term goal is a phone app that works **offline in the bush**: hold up the phone, record a bird, and get the species back.

> **Status (Oct 2026).** The v2 pipeline lives in the `manuai/` package. It has been smoke-tested on an M1 Pro (MPS). The first full training run is pending. The capstone (v1) is kept as a frozen reference; see the bottom of this page.

## Classes

There are 20 classes plus `Other` (any identified species not in the list). They are defined in [`types.yaml`](types.yaml):

tūī · korimako/Bellbird · pīwakawaka/Fantail · riroriro/Grey Warbler · tauhou/Silvereye · miromiro/Tomtit · toutouwai/Robin · tītitipounamu/Rifleman · kākā · kea · ruru/Morepork · kiwi (all species) · weka · pūkeko · koekoeā/Long-tailed Cuckoo · pīpīwharauroa/Shining Cuckoo · kōtare/Kingfisher · Blackbird · Chaffinch · Song Thrush

The selection is data-driven: every class has enough independent recordings to be tested on unseen sites. Rarer birds (kōkako, tīeke, hihi, kererū, kākāriki, …) are folded into `Other` until there is more data.

## Data

The data is the [New Zealand Bird Sound](https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound) Kaggle dataset, unzipped into `archive/`:

| Source | Contents |
|---|---|
| DOC_001_Tier1 | DOC acoustic recorder clips from about 1,200 recorders across NZ |
| DOC_002_DuncanBayParinga | DOC field recordings from one South Westland area |
| DOC_003_XenoCanto | Xeno-canto clips |

In total there are about 271k FLAC clips at 32 kHz (43 GB).

## Pipeline (v2)

```bash
conda activate ManuAI
pip install -r requirements.txt && pip install -e .   # fresh env: pinned deps + the manuai package

python -m manuai.build_index               # archive metadata → data/index.csv + grouped split
python -m manuai.preprocess                # → data/features/ (float16 cache, ~34 GB)
python -m manuai.train --model ast --run-name ast-v2-1   # AST + LoRA (primary)
python -m manuai.train --model vit --run-name vit-v2-1   # ViT + LoRA (baseline)
python -m manuai.evaluate --run models/finetuned/ast-v2-1 --splits test
python -m manuai.infer --run models/finetuned/ast-v2-1 my_recording.wav
python -m manuai.report --run models/finetuned/ast-v2-1   # → REPORT.md
```

**Notebooks** (in `notebooks/`): `01_data.ipynb` covers the data, split, spectrograms and audio, and builds the caches. `02_train.ipynb` trains and plots runs. `03_evaluate.ipynb` shows the report, confusion matrix and mistakes (with audio playback) and lets you try any audio file. They drive the same package, and their heavy cells are behind flags.

`ManuAI.config` sets `data_directory` (default `data`, relative to the repo). Smoke-test a change on a small balanced subset first:

```bash
python -m manuai.preprocess --name features_smoke --per-class 200 --per-class-eval 40
python -m manuai.train --model ast --smoke --features features_smoke
```

Run the fast invariant tests with `python -m pytest -q tests/`. They cover feature parity with `ASTFeatureExtractor`, segmenting, labels, and the absence of split leakage.

To stop a running training job gracefully, `touch data/logs/STOP`. The best checkpoint is kept, and the final evaluation can be run later with `manuai.evaluate`.

### What each step does

| Module | |
|---|---|
| `manuai/labels.py` | `types.yaml` → classes. Every other identified species becomes `Other`. Unidentified labels (`parake` = *Psittaciformes sp.*, `badbird`) are dropped. |
| `manuai/build_index.py` | Builds one row per clip, with its recording and group. The split is a `StratifiedGroupKFold` (7 folds, seed 21): **no recorder (DOC_001), field recording (DOC_002) or Xeno-canto recording appears in more than one split.** The result is ≈ 71 / 16 / 13 % train/val/test. |
| `manuai/features.py` | **The one feature definition**, used by training, evaluation and the app: 16 kHz mono, 5 s segments (short clips are looped, not zero-padded), Kaldi fbank exactly as HF `ASTFeatureExtractor` (128 mel, 10 ms hop), AudioSet normalisation → 498 × 128. |
| `manuai/preprocess.py` | Computes the features into a float16 memmap per split, plus a segment table. |
| `manuai/data.py` | √-frequency class sampling with a per-recording cap, SpecAugment and time-roll, and collate functions for AST and ViT. |
| `manuai/models.py` | **AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) at 512 frames, with its pretrained position embeddings centre-cropped. **ViT** (`google/vit-base-patch16-224-in21k`). LoRA r=16 on every encoder linear layer, with the classifier trained in full. |
| `manuai/train.py` | Plain cross-entropy (label smoothing 0.1, no class weights). "Virtual epochs" of sampled segments. Checkpoints are selected on macro-F1. Writes `training_metrics.json`. |
| `manuai/evaluate.py`, `metrics.py` | Macro-F1 (primary), accuracy, balanced accuracy and macro AUC, at **segment and recording level**, plus per-class reports and a confusion matrix. |
| `manuai/inference.py`, `infer.py` | Load a run and classify audio files. This is the basis for the app. |

Base checkpoints are loaded from `models/base/{mit-ast,google-vit}`, so training works offline. Runs are written to `models/finetuned/<run-name>/`.

### Hardware notes (M1 Pro, 16 GB)

- AST at 512 frames, batch 8 × 4 accumulation: about **6 s per optimizer step** and about 10–11 GB of MPS memory (Oct 2026 smoke test).
- fp16/bf16 stay off on MPS.
- DataLoader workers work, because all collate code lives in the package.
- The full feature cache takes about 34 GB of disk.

## Repository layout

```
ManuAI/
├── manuai/              # the pipeline package (features, labels, split, data, models, train, eval, inference)
├── notebooks/           # 01_data · 02_train · 03_evaluate  (interactive front end over manuai/)
│   └── legacy/          # first v2 attempt (Mar 2026)
├── scripts/             # agent_runner.ipynb + agent_job.sh (detached runs), fetch_base_models.py
├── tests/               # invariant tests (pytest)
├── legacy/              # v1 capstone code and results, weights-poster script
├── docs/                # working notes, ready-to-enable CI workflow (docs/ci/)
├── types.yaml           # the class list
├── ManuAI.config        # data_directory
└── requirements.txt · pyproject.toml · pytest.ini · AGENTS.md
```

Kept locally and never committed (see `.gitignore`):
- `archive/`: the dataset
- `data/`: index, feature caches, logs
- `models/base/`: pretrained checkpoints
- `models/finetuned/`: runs
- `saved_models/`, `reports/` and `art/`

### Pretrained base models

Base checkpoints live in `models/base/<name>/` and are loaded from there, so training works offline. To add a new one (for example to trial Bird-MAE or CED):

```bash
python scripts/fetch_base_models.py --list               # registry + what's already downloaded
python scripts/fetch_base_models.py bird-mae-base ced-base
```

Each download records its exact Hugging Face revision in `models/base/<name>/SOURCE.json`.

## Branches and tags

- **`main`**: the project, with full history back to the capstone.
- **`v1-capstone`** (tag): the repo exactly as submitted for AIML339 in Oct 2025.
- Work happens on short-lived feature branches merged into `main` by pull request.
- **CI (optional):** to have GitHub run `pytest` on every push and PR, copy [`docs/ci/tests.yml`](docs/ci/tests.yml) to `.github/workflows/tests.yml` and commit it.

## Capstone (v1) — frozen reference

ManūAI started as Harry Wills's AIML339 capstone at Victoria University of Wellington (2025, supervised by Prof. W. Bastiaan Kleijn). v1 fine-tuned ViT on librosa log-mel images of 4 s segments. Its headline result was **72.6 % accuracy, macro-F1 ≈ 0.73, AUC 0.97** on 20 species.

That split was made per segment, so segments from the same recording appeared in both training and test. The v1 figure is therefore optimistic and **not comparable** with v2's grouped split. The v1 code and results are kept unchanged in [`legacy/v1/`](legacy/v1/) (see its README), the git tag `v1-capstone` marks the repo as submitted, and the `saved_models/` deliverables stay on disk.

The first v2 attempt (Mar 2026) is in `notebooks/legacy/`. The numbered notebooks plus the `manuai/` package replace it.

## Acknowledgments

- The Department of Conservation and the Kaggle dataset curator, for the acoustic recorder data
- [Xeno-canto](https://xeno-canto.org/) recordists
- [Hugging Face](https://huggingface.co/) Transformers and PEFT
- Gong, Chung & Glass — *AST: Audio Spectrogram Transformer* (2021)

## Citation

```bibtex
@misc{manuai2025,
  title={ManuAI: New Zealand Bird Sound Classification},
  author={Harry Wills},
  year={2025},
  url={https://github.com/harrywillss/ManuAI}
}
```

## Contact

Harry Wills — [@harrywillss](https://github.com/harrywillss)

*Made with ❤️ for Aotearoa's birds.*
