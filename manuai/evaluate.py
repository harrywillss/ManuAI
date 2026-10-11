"""Evaluate a trained run on the full val/test feature cache (segment- and recording-level).

    python -m manuai.evaluate --run models/finetuned/ast-v2-1 --splits test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import data_dir
from .data import SegmentDataset, read_rows
from .inference import free_device_memory, load_run, pick_device, predict_dataset
from .metrics import full_report


def evaluate_split(model, collate, feat_dir: Path, split: str, rows: pd.DataFrame, classes: list[str],
                   device: str, batch_size: int = 16, workers: int = 2) -> tuple[dict, np.ndarray]:
    ds = SegmentDataset(feat_dir, split, rows)
    t0 = time.time()
    probs = predict_dataset(model, ds, collate, device, batch_size=batch_size, workers=workers)
    rep = full_report(probs, ds.labels, ds.recording_ids, classes)
    rep["seconds"] = round(time.time() - t0, 1)
    print(f"📊 {split}: segment macro-F1 {rep['segment']['macro_f1']:.3f}, acc {rep['segment']['accuracy']:.3f} | "
          f"recording macro-F1 {rep['recording']['macro_f1']:.3f}, acc {rep['recording']['accuracy']:.3f} "
          f"(n={rep['segment']['n']:,} segs / {rep['recording']['n']:,} recs, {rep['seconds'] / 60:.1f} min)", flush=True)
    return rep, probs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--features", default="features")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    device = pick_device()
    model, spec, collate = load_run(run_dir, device)
    feat_dir = data_dir() / args.features
    metrics_path = run_dir / "training_metrics.json"
    summary = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    summary.setdefault("eval", {})
    for split in args.splits:
        free_device_memory(device)
        rep, probs = evaluate_split(model, collate, feat_dir, split, read_rows(feat_dir, split), spec["classes"],
                                    device, args.batch_size, args.workers)
        summary["eval"][split] = rep
        np.save(run_dir / f"probs_{split}.npy", probs.astype(np.float16))
        metrics_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"📝 {metrics_path}")


if __name__ == "__main__":
    main()
