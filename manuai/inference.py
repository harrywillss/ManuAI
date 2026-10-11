"""Load a trained run and predict — shared by evaluation, the CLI and (later) the app.

    python -m manuai.infer --run models/finetuned/<run> recording.wav [more.wav ...]
    python -m manuai.evaluate --run models/finetuned/<run> --splits test
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import features as F
from .data import Collate
from .models import AST_BASE, VIT_BASE, crop_ast_positions


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_device_memory(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def load_spec(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manuai_model.json").read_text())


def make_collate(spec: dict) -> Collate:
    norm = spec.get("vit_input_norm") or {"mean": 0.0, "std": 0.5}
    return Collate(spec["model"], spec["input_frames"], vit_mean=norm["mean"], vit_std=norm["std"])


def load_run(run_dir: str | Path, device: str | None = None):
    """Return (model, spec, collate) for a finished run. Prefers `merged/`, else base + `adapter/`."""
    run_dir = Path(run_dir)
    spec = load_spec(run_dir)
    device = device or pick_device()
    classes = spec["classes"]
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}
    if spec["model"] == "ast":
        from transformers import ASTForAudioClassification as Cls
        base = AST_BASE
    else:
        from transformers import ViTForImageClassification as Cls
        base = VIT_BASE
    if (run_dir / "merged" / "config.json").exists():
        model = Cls.from_pretrained(run_dir / "merged")
    else:
        from peft import PeftModel
        model = Cls.from_pretrained(base, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
        if spec["model"] == "ast":
            crop_ast_positions(model, spec["input_frames"])
        model = PeftModel.from_pretrained(model, run_dir / "adapter")
    return model.to(device).eval(), spec, make_collate(spec)


class _ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"x": torch.from_numpy(np.asarray(self.x[i], dtype=np.float32)), "label": -1}


@torch.inference_mode()
def predict_dataset(model, ds: Dataset, collate, device: str, batch_size: int = 16,
                    workers: int = 2, log_every: int = 200) -> np.ndarray:
    """Softmax probabilities for every item of a dataset (dict items with 'x')."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                    num_workers=workers, pin_memory=False)
    out, t0 = [], time.time()
    for i, batch in enumerate(dl, 1):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        if log_every and i % log_every == 0:
            done = i * batch_size
            rate = done / (time.time() - t0)
            print(f"   … {done:,}/{len(ds):,} segments ({rate:.0f}/s)", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def predict_file(path: str | Path, model, spec: dict, collate, device: str, top_k: int = 3) -> dict:
    """Classify one audio file: per-segment probabilities and a recording-level ranking."""
    x = F.features_from_file(path)
    if len(x) == 0:
        return {"file": str(path), "segments": 0, "top": []}
    probs = predict_dataset(model, _ArrayDataset(x), collate, device, batch_size=8, workers=0, log_every=0)
    mean = probs.mean(axis=0)
    order = np.argsort(-mean)[:top_k]
    classes = spec["classes"]
    return {
        "file": str(path),
        "segments": int(len(x)),
        "top": [{"class": classes[i], "prob": round(float(mean[i]), 4)} for i in order],
        "per_segment": [classes[i] for i in probs.argmax(axis=1)],
    }
