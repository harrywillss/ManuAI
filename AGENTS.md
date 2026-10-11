# AGENTS.md — ManūAI

Guidance for AI coding agents (Claude, Codex, Copilot, etc.) working in this repository.
Read this whole file before making changes. Keep it current (see "Documentation duties").

---

## 1. What this project is

**ManūAI** classifies New Zealand bird vocalisations. *Manu* is te reo Māori for "bird", so keep the macron in prose ("ManūAI"). The repo, folder and code identifiers stay `ManuAI`.

Audio is converted to log-mel (Kaldi fbank) features and passed to a pretrained transformer that is fine-tuned with **LoRA** (PEFT).

- **Origin:** Harry Wills's AIML339 capstone at Victoria University of Wellington (2025), supervised by Prof. W. Bastiaan Kleijn. It was graded and is finished. ManūAI is now a personal project that Harry is actively developing.
- **Long-term goal:** a phone app that works **offline in the bush**. You hold up the phone, record a bird, and get back the species plus information about it. Conservation groups (DOC, Forest & Bird) are possible future audiences.
- **Owner:** Harry Wills ([@harrywillss](https://github.com/harrywillss)). Remote: `https://github.com/harrywillss/ManuAI`.

### Current priorities (in order)

1. **Get a first full training run done on the grouped split.** AST is the primary model. Report it as the v2 baseline (§8).
2. **Review the architecture and build, and recommend improvements.** Back every problem you raise with evidence: numbers, file paths, line numbers.
3. **Help develop the application.** Prefer offline-capable designs, and lay out the trade-offs before committing to one (§9).
4. **Clean up the environment and repo** so the project is reproducible and has no stale or duplicated artefacts (§7.9).
5. **Keep documentation accurate and consistent** with the code.

---

## 2. Repository map

There are three generations side by side:

- **v2.1 is canonical.** Reusable logic lives in the `manuai/` package. Harry likes notebooks for ML and data work, so the numbered notebooks (`01_data`, `02_train`, `03_evaluate`) are the interactive front end over the package. Use and extend them freely.
- The Mar 2026 v2 notebooks are superseded.
- v1 is the frozen capstone reference.

### v2.1 — canonical (Oct 2026 onward): the `manuai/` package

Run each step from the repo root with `python -m manuai.<module>` (see README):

| Module | Purpose |
|---|---|
| `config.py` | Paths. Reads `ManuAI.config` (`data_directory: data`); relative paths resolve against the repo root. |
| `labels.py` | Class set from `types.yaml`. Every other identified species → `Other`. `EXCLUDED_CODES` = `parake` (*Psittaciformes sp.*, an unidentified parrot), `badbird`, `spybird`. |
| `build_index.py` | Archive metadata → `data/index.csv`: path, source, eBird code, label, secondary labels, `recording_id`, `group`, split. Also writes `data/split_summary.md`. |
| `features.py` | **Single source of truth for features** (training, eval and the app): 16 kHz mono, 5 s segments, no overlap. An end-aligned extra segment is added if the tail is ≥ 50 % of a segment. Clips < 5 s are **tiled**. Kaldi fbank identical to HF `ASTFeatureExtractor` (verified to within 6e-6), AudioSet normalisation → `(498, 128)`. `pad_frames` pads to the model length with the AST pad value. `FEATURE_VERSION` guards the cache. |
| `preprocess.py` | Writes `data/<name>/{split}.npy`: a float16 memmap `(n, 498, 128)`, plus `{split}_segments.csv`, plus `meta.json` (feature spec, classes, counts, train mean/std). `--per-class` builds small subsets. |
| `data.py` | `SegmentDataset` (memmap), `AugmentConfig` (time-roll, 2× freq masks ≤ 16, 2× time masks ≤ 40), `Collate` (AST: pad to 512; ViT: area-resize to 224², global standardisation), `balanced_sampler` (class ∝ n^0.5, per-recording cap 20). |
| `models.py` | AST at **512 frames**, with pretrained position embeddings centre-cropped (`crop_ast_positions`). ViT. LoRA r=16/α=32 on every encoder `nn.Linear` (found by walking the model, so it works on transformers 4.x and 5.x). `modules_to_save=["classifier"]`. |
| `train.py` | HF `Trainer` subclass with plain CE (label smoothing 0.1, **no class weights**). Virtual epochs (`--epoch-size`). Eval on a capped val subset, checkpoints selected on **macro-F1**. Final full eval on `--final-eval-splits`. `--smoke` runs a 40-step check. Honours `data/logs/STOP`. |
| `evaluate.py`, `metrics.py` | Segment- and recording-level metrics, per-class report, confusion matrix → `training_metrics.json` + `probs_<split>.npy`. |
| `report.py` | `training_metrics.json` → `REPORT.md` with headline numbers, per-class F1 and top confusions. The overnight job runs it automatically. |
| `inference.py`, `infer.py` | `load_run` (prefers `merged/`, else base + `adapter/`), `predict_file`. CLI: `python -m manuai.infer --run <dir> file.wav`. |

Notebooks (repo root). Each one imports `manuai` and has `%autoreload` on. Heavy cells are guarded by flags (`BUILD_FULL`, `TRAIN`, `EVALUATE`) so *Run All* is safe in the daytime:

| Notebook | Purpose |
|---|---|
| `01_data.ipynb` | Build the index and split, plot class balance and sampler shares, show spectrograms and play audio per class, build the feature caches |
| `02_train.ipynb` | Configure and run `manuai.train` in the kernel (smoke tests and short runs); plot loss, val macro-F1, s/step and memory for any run; render the report |
| `03_evaluate.ipynb` | Report, row-normalised confusion matrix, per-class F1, the most confident mistakes with spectrogram and audio, and try-any-file inference |

Every CLI module also takes `main(argv)`, so notebooks call e.g. `train.main(["--model", "ast", "--smoke"])`. `train.main` returns the run directory.

Other v2.1 files:

- `types.yaml`: **the class list** (§6).
- `requirements.txt`: pins from the real env.
- `tests/test_pipeline.py`: invariant tests (`python -m pytest -q tests/`). Run them after touching `features.py`, `labels.py` or the split.
- `scripts/agent_runner.ipynb` + `scripts/agent_job.sh`: how agents run jobs on the Mac (§10).

Each run in `models/finetuned/<run>/` contains:

- `adapter/`
- `merged/` (full runs only)
- `checkpoints/`
- `runs/` (TensorBoard)
- `train_log.jsonl` (loss, sec/step, MPS memory)
- `manuai_model.json` (classes, feature spec, input length): everything an app needs
- `training_metrics.json`

### v2 (Mar 2026) notebooks: legacy, kept for reference

- `data_download.ipynb`
- `preprocess_data.ipynb`: 4 s, padded to 1024, per-file split
- `finetune.ipynb`: ViT, weighted CE
- `ast_finetune.ipynb`: AST at 1024
- `analyse_dataset.ipynb`

Their failure modes are recorded in §7. Don't build on their code. New notebook work goes in the numbered notebooks, on top of `manuai/`.

### v1: frozen capstone reference (Aug–Oct 2025)

- `download_data.py`, `preprocess_data.py`, `inference.py` and `bird_types.yaml` (the v1 class list) make up the v1 code.
- The committed (HEAD) `finetune.ipynb` is the v1 training notebook.
- v1 used:
  - 22.05 kHz librosa log-mel at 224²
  - 4 s windows with 50 % overlap
  - an **80/10/10 split per segment**, which leaks
- `saved_models/` holds the v1 deliverables. `reports/` holds the submitted coursework: **do not edit it.**
- **Headline capstone result to cite: 72.6 % accuracy, macro-F1 ≈ 0.73, AUC 0.97.** This is the 20-species model from `training_metrics.json` at the repo root. Because of the leaky split it is **not comparable** to v2 numbers.
- Don't modify v1 files unless asked.

### Other contents

- `archive/`: the raw Kaggle dataset `ollypowell/new-zealand-bird-sound`:
  - DOC_001_Tier1: 5 s clips from about 1,234 recorders
  - DOC_002_DuncanBayParinga: 3–5 s clips, all from one region
  - DOC_003_XenoCanto: 10–12 s clips

  That is 271,365 FLAC files at 32 kHz (43 GB). **Read-only.** Each source has `*_metadata.csv` and `bird_naming_map.csv`. Files are 8 folders deep below Desktop, which is too deep for the device staging tool, so copy samples to `data/tmp_samples/` if you need them elsewhere.
- `models/base/{mit-ast,google-vit}`: local HF checkpoints (offline training).
- `art/`: a weights poster. It's a side project, so leave it alone.
- `brainstorm.md` and `for_final.md` are Harry's notes.
- **Ignored by git:** `archive/`, `models/`, `saved_models/`, `data/`, `reports/`, `art/`, `.env`, `__pycache__/`, `.DS_Store`, `.vscode/`, zips.

---

## 3. Environment

- **Python 3.11.13**, conda env **`ManuAI`** (`~/miniconda3/envs/ManuAI`).
- Key packages: torch 2.8.0, torchaudio 2.8.0, transformers 4.55.4, peft 0.17.0, scikit-learn 1.7.2, soundfile 0.13.1. These are pinned in `requirements.txt`. A full freeze is in `data/logs/pip-freeze.txt`.
- The code is also tested on transformers 5.19 / torch 2.14 (Linux CPU). transformers 5 renamed the modules (`layers.N.attention.q_proj`) and dropped `logging_dir`; the package handles both versions.
- System tools: `ffmpeg`/`ffprobe` (v1 only).
- Secrets: `.env` holds `XC_API_KEY`. Never print it, log it or commit it. Xeno-canto is **not reachable** from agent sandboxes (proxy 403).

---

## 4. Hardware constraints (design for these)

The primary machine is a **MacBook M1 Pro with 16 GB unified memory**. Training runs on **MPS**. Harry uses this laptop day to day: **run training overnight or when he says the laptop is free.**

**Measured (11 Oct 2026 smoke tests, while the laptop was also in use):**

| Model | Batch | Time | Peak MPS memory | Notes |
|---|---|---|---|---|
| AST, 512 frames | 8 × accum 4 | ≈ 5.8 s per optimizer step | ≈ 10.7 GB | Eval ≈ 5–6 segments/s through Trainer |
| ViT | 32 | ≈ 3–10 s/step (contended) | ≈ 10.3 GB | |

- After 40 steps, val-subset macro-F1 was **AST 0.16** and **ViT 0.05** (chance = 0.048).
- Starting losses were 3.14 and 3.07 (ln 21 = 3.04). This confirms the old loss bug is gone.

**Settings:**

- `fp16=False`, `bf16=False`.
- DataLoader workers are fine (`--workers 4`) because collate code lives in the package.
- Feature preprocessing runs at about 370 clips/s with 8 workers. The full cache takes about 12–15 min and ~34 GB.
- **Disk:** ~77 GB free of 927 GB after the 35 GB feature cache (12 Oct 2026). Check `df -h` before writing large outputs.
- Cloud GPU is acceptable for heavy runs. The code is device-agnostic (cuda → mps → cpu) and paths come from config.

---

## 5. Data location policy

- `ManuAI.config` → `data_directory: data` (relative to the repo).
- `data/` holds:
  - `index.csv` and `split_summary.md`
  - `features/`: the full cache, built by the overnight job
  - `features_smoke/`: a 200/40-per-class subset
  - `logs/`
- Audio is referenced **in place** in `archive/`. Nothing is copied. The USB-drive layout and `pre/` / `recordings.csv` from the Mar 2026 notebooks are obsolete.
- Xeno-canto API downloads are **not** part of v2.1; DOC_003 provides the XC clips. Adding more XC data needs Harry's machine (the API is blocked from agent sandboxes) and must keep `recording_id` = `XC:<id>`.
- If the config path doesn't exist, **stop and ask**. Don't silently fall back to another location.

---

## 6. Classes (decided 11 Oct 2026)

**20 classes + `Other`.** See the project doc `claude/class-set-recommendation.md` for the full 133-label inventory and the reasoning.

`Bellbird, Blackbird, Chaffinch, Fantail, GreyWarbler, Kaka, Kea, Kingfisher, Kiwi (all 5 species), LongTailedCuckoo, Morepork, Pukeko, Rifleman, Robin (NI + SI), ShiningCuckoo, Silvereye, SongThrush, Tomtit, Tui, Weka, Other`

- The criterion was enough independent recordings and sites for a grouped test set. The smallest class (Kingfisher) has 67 test recordings.
- **`Other`** is every identified non-target species plus Tree wētā (~8.8k clips). `Other` clips with a target species in their secondary labels are dropped (4,156 clips).
- **Stretch candidates:** kārearea, pīpipi, pōpokotea, kākāriki (130 recordings once `parake` is excluded), kererū, redpoll, dunnock, magpie, spur-winged plover. Add one only if a run shows stable per-class F1.
- **Not trainable from the archive:** kōkako (20 recordings), tīeke (24), hihi (31), mohua, takahē, kākāpō. These need Harry's own field recordings (e.g. Zealandia), each visit its own group.
- Korimako is in the background of 42k clips and miromiro of 25k, so expect those confusions. Tūī mimicry is inherent label noise.
- `bird_types.yaml` is the **v1** list. Harry may edit it; v2.1 ignores it.

---

## 7. Known issues and open problems (as of 11 Oct 2026)

Check whether an item has been fixed before changing anything listed here, and keep this list updated.

1. ✅ **The v2 ViT collapse is fixed in code.**
   - **Cause:** PyTorch applies class weights to the label-smoothing term too. With balanced weights (sum ≈ 340), the loss-optimal prediction for Bellbird, Morepork, Silvereye and Tomtit was Kōkako.
   - **Fix:** `manuai/train.py` uses unweighted CE plus a √-frequency sampler. The smoke runs start at ≈ ln K.
2. ✅ **The padding problem is fixed.** Features are 5 s / 498 real frames. AST runs at 512 frames with cropped position embeddings. ViT gets cropped, area-resized images.
3. ✅ **The split leak is fixed.** `build_index.py` uses `StratifiedGroupKFold` by DOC_001 recorder, DOC_002 recording and XC id, and asserts that no group or recording crosses splits:

   | Split | Clips | Recordings | Groups |
   |---|---|---|---|
   | Train | 187,893 | 41,325 | 19,498 |
   | Val | 41,929 | 8,959 | 3,897 |
   | Test | 34,957 | 7,449 | 3,888 |

   DOC_002 is all one region, so site leakage within DOC_002 is unavoidable. Robin's test set comes from only 20 groups.
4. ✅ **AST trains on the M1.** The Mar 2026 failure was most likely memory: 1,214 tokens, and the model was loaded twice.
   - ✅ **The full feature cache is built** (12 Oct, 09:21): `data/features`, 35 GB. It holds 204,187 train / 45,598 val / 37,791 test segments, with 0 errors, at ~550 clips/s.
   - ⏳ **First full run `ast-v2-1`:** scheduled for 12:30 pm NZDT on 12 Oct (Harry's choice). `scripts/agent_job.sh` is now training-only: 6 × 19,200-segment epochs, then the full test set and `REPORT.md`, ≈ 7.5–8 h.
   - ⚠️ **Use the Mac's `date` for scheduling decisions.** The agent sandbox clock stalled while the Mac slept on 11 Oct, which caused a launch at 09:12 instead of 23:30. Training was blocked with `data/logs/STOP` before it began.
5. ✅ **There is a v2 inference path** (`manuai/inference.py`, `infer.py`). The v1 `inference.py` is untouched and still v1-only.
6. ✅ **The README is rewritten for v2.1.**
7. ✅ **Checkpoints are selected on macro-F1.**
8. **`preprocess_data.py` (v1) is destructive** (it `rmtree`s its outputs). Don't run it.
9. **Repo hygiene is still open.**
   - v2 and v2.1 work isn't committed. Branch `clean-main`; Harry had files staged on 11 Oct, so **ask before committing**.
   - Tracked `__pycache__/*.pyc` and `.DS_Store`.
   - Duplicate model zips (≈ 1.6 GB).
   - Two YAML class lists.
   - A `.gitignore` line padded with spaces.
   - `data/tmp_samples/` and `models/finetuned/smoke-*` are disposable.
10. **Evaluation through `Trainer` is slow on MPS** (about 5–6 AST segments/s in the smoke test). `manuai.evaluate` uses a plain inference loop. Measure its throughput on the first full test eval.
11. **Open modelling questions** for after the baseline:
    - mixup / background-noise mixing
    - soft secondary labels
    - a "no bird" class
    - `alpha` and LoRA rank sweeps
    - full fine-tuning on a cloud GPU
    - distillation to a small CNN for the phone (§9)

---

## 8. Evaluation conventions

- **The primary metric is macro-F1.** Also report:
  - accuracy and balanced accuracy
  - per-class precision, recall and F1
  - the confusion matrix
  - OvR AUC (macro)
- Select checkpoints on macro-F1.
- Evaluate on the **group-held-out test set**. Report results at **segment level and recording level** (recording level = the mean of segment probabilities).
- Always state these next to any reported number: class set, split method, segment length, feature type, test-set size. Never compare v2 numbers with v1's 72.6 % without saying the v1 split leaked.
- Seeds: `21` everywhere. Hyperparameters and versions are recorded in each run's `training_metrics.json`.

---

## 9. Application direction

- The goal is an **offline-capable** iPhone app for use in the bush.
- Before building anything, write up the trade-offs:
  - on-device inference: Core ML / ExecuTorch / TFLite, int8 quantisation, size, latency, battery
  - a cloud API
  - a hybrid
- An 86M-parameter AST is large for a phone. Consider distilling to a small CNN on the same fbank features.
- **Feature extraction must be identical** between training and the app. Port `manuai/features.py` exactly; `manuai_model.json` carries the spec.
- Species information shown to users should come from reputable sources (e.g. NZ Birds Online, DOC).
- Use te reo Māori names with correct macrons: tūī, kākā, kererū, pīwakawaka, kōkako, pūkeko, tīeke, korimako, ruru, miromiro, tauhou, pōpokotea, toutouwai, hihi, mohua, riroriro, kea, kiwi, kākāriki, kōtare, tītitipounamu, koekoeā, pīpīwharauroa, kārearea, pīpipi.

---

## 10. How agents should work here

### Running jobs on the Mac

Agent shells on the Mac are a Linux VM: no MPS, no torch. Terminal and VS Code only accept clicks from agents. So jobs run like this:

1. Edit **`scripts/agent_job.sh`** (it runs with the `ManuAI` kernel's Python via `$PY`). **Never edit it while a job is running**: bash reads scripts incrementally.
2. Open **`scripts/agent_runner.ipynb`** in VS Code (kernel: ManuAI) and click **Run All**. It launches the script detached under `caffeinate -i`, logs to `data/logs/agent_job.log`, writes `data/logs/agent_job.pid`, and refuses to start a second job.
3. Monitor the log (and `models/finetuned/<run>/train_log.jsonl`) from the VM.
4. **Stop gracefully:** create `data/logs/STOP`. The training callback stops at the next step, keeps the best checkpoint and skips the final eval. New launches clear the file.

**Ask Harry before using the laptop's GPU in the daytime.** He prefers training runs overnight.

### Granted latitude

- ✅ **May launch long-running jobs**, within the timing rule above. Before any long run:
  - smoke-test first (`--smoke`, `features_smoke`)
  - estimate time and disk
  - log to a file
- ✅ **May edit code and notebooks.** Don't change a notebook's kernel name.
- ✅ **May create git commits** on a feature branch with clear messages. **Never push, force-push, rebase shared branches or rewrite history without asking.** Check `git status` first, because Harry may have work staged.
- ✅ **May write under `data/`.** Deletion from agent shells needs an explicit grant.

### Guardrails

- **Never modify or delete** `archive/`, `reports/`, `saved_models/` or `art/` without an explicit request.
- Never commit data, model weights, zips, `.env` or anything else in `.gitignore`.
- Don't start two heavy jobs at once on the 16 GB machine.
- Reusable logic (features, labels, datasets, models, metrics) lives in `manuai/`, so the notebooks, the CLI, DataLoader workers and the app all share one definition. Notebooks are welcome for exploration, plots, data operations and short runs.
- When you recommend a change, give the evidence, the expected effect, the cost (time, memory, disk), and what you would measure to confirm it.
- After writing files to the Mac through the device bridge, **verify sizes or contents**. One write landed stale on 11 Oct.

### Style

- Python 3.11, type hints, `pathlib`.
- Config comes from `ManuAI.config` and `.env`.
- Friendly tqdm/emoji logging is fine.
- Seed everything.

---

## 11. Documentation duties

- Update **this file** (§2 map, §6 classes, §7 issues) and **`README.md`** when structure, classes or known issues change.
- Every training run writes `training_metrics.json` with its config, versions and metrics.
- Terms:
  - "segment" = a **5 s** window
  - "recording" = a source recording (`recording_id`)
  - "group" = the split unit (recorder / recording / XC id)
  - "class"/"label" = a `types.yaml` key or `Other`
- Project docs (claude.ai Project "ManuAI"):
  - `claude/AGENTS.md`: a mirror of this file
  - `claude/model-setup-review.md`
  - `claude/class-set-recommendation.md`
