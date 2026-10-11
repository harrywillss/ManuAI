"""Compute the float16 feature cache from `data/index.csv`.

    python -m manuai.preprocess                          # full cache → data/features/
    python -m manuai.preprocess --name features_smoke \
        --per-class 200 --per-class-eval 40              # small balanced subset

Output per split (in data/<name>/):
  <split>.npy            float16 memmap, shape (n_segments, 498, 128) — real frames only
  <split>_segments.csv   one row per segment: seg, label, label_id, recording_id, group, source, path, seg_no
  meta.json              feature spec, class list, counts

Size: ~127 KB per segment, so the full archive (~265k segments) is ~34 GB.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import features as F
from .config import REPO_ROOT, SEED, data_dir
from .labels import class_names


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(10)
    except OSError:
        pass


def _count(path: str) -> int:
    """Number of segments a file will produce (header read only)."""
    try:
        info = sf.info(str(REPO_ROOT / path))
    except Exception:
        return 0
    n16 = math.ceil(info.frames * F.SR / info.samplerate)
    return F.n_segments(n16)


def _featurise(args: tuple[int, str, int]) -> tuple[int, np.ndarray | None, str | None]:
    i, path, expected = args
    try:
        x = F.features_from_file(REPO_ROOT / path)
    except Exception as e:  # noqa: BLE001 — log and continue
        return i, None, f"{path}: {type(e).__name__}: {e}"
    if x.shape[0] != expected:
        # Header length and decoded length can differ by a few samples; keep the cache
        # layout fixed by trimming/padding the segment count.
        if x.shape[0] > expected:
            x = x[:expected]
        else:
            pad = np.repeat(x[-1:], expected - x.shape[0], axis=0) if len(x) else np.zeros((expected, F.N_FRAMES, F.N_MELS), np.float32)
            x = np.concatenate([x, pad])
    return i, x.astype(np.float16), None


def subsample(df: pd.DataFrame, per_class: int | None, per_class_eval: int | None, seed: int) -> pd.DataFrame:
    if not per_class:
        return df
    parts = []
    for (split, _), g in df.groupby(["split", "label"]):
        n = per_class if split == "train" else (per_class_eval or max(1, per_class // 5))
        # Spread the sample across recordings: shuffle, then take one clip per recording first.
        g = g.sample(frac=1.0, random_state=seed)
        g = g.assign(_rank=g.groupby("recording_id").cumcount()).sort_values("_rank", kind="stable")
        parts.append(g.head(n).drop(columns="_rank"))
    return pd.concat(parts).sort_index()


def build_split(df: pd.DataFrame, split: str, out_dir: Path, workers: int, log) -> dict:
    sub = df[df["split"] == split].reset_index(drop=True)
    names = class_names()
    label_id = {c: i for i, c in enumerate(names)}

    log(f"🔎 {split}: counting segments for {len(sub):,} clips …")
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        counts = pool.map(_count, sub["path"].tolist(), chunksize=256)
    sub["n_seg"] = counts
    sub = sub[sub["n_seg"] > 0].reset_index(drop=True)
    offsets = np.concatenate([[0], np.cumsum(sub["n_seg"].to_numpy())])
    total = int(offsets[-1])
    gb = total * F.N_FRAMES * F.N_MELS * 2 / 1e9
    log(f"   {split}: {total:,} segments from {len(sub):,} clips → {gb:.1f} GB")

    arr = np.lib.format.open_memmap(out_dir / f"{split}.npy", mode="w+", dtype=np.float16,
                                    shape=(total, F.N_FRAMES, F.N_MELS))
    bad = np.zeros(total, dtype=bool)
    jobs = [(i, p, int(n)) for i, (p, n) in enumerate(zip(sub["path"], sub["n_seg"]))]
    t0 = time.time()
    errors = 0
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for k, (i, x, err) in enumerate(pool.imap(_featurise, jobs, chunksize=32), 1):
            a, b = offsets[i], offsets[i + 1]
            if x is None:
                errors += 1
                bad[a:b] = True
                if errors <= 20:
                    log(f"   ⚠️ {err}")
            else:
                arr[a:b] = x
            if k % 5000 == 0 or k == len(jobs):
                rate = k / (time.time() - t0)
                log(f"   {split}: {k:,}/{len(jobs):,} clips ({rate:.0f}/s, ETA {(len(jobs) - k) / rate / 60:.1f} min)")
    arr.flush()
    del arr

    rows = sub.loc[sub.index.repeat(sub["n_seg"])].reset_index().rename(columns={"index": "clip"})
    rows["seg_no"] = rows.groupby("clip").cumcount()
    rows["seg"] = np.arange(total)
    rows["label_id"] = rows["label"].map(label_id)
    rows["bad"] = bad
    rows = rows[["seg", "label", "label_id", "recording_id", "group", "source", "ebird", "path", "seg_no", "bad"]]
    rows.to_csv(out_dir / f"{split}_segments.csv", index=False)
    return {"clips": len(sub), "segments": total, "errors": errors, "gb": round(gb, 2)}


def feature_stats(path: Path, n: int = 4000, seed: int = SEED) -> dict | None:
    """Mean/std of the cached (normalised) features over a random sample of segments."""
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r")
    if len(arr) == 0:
        return None
    idx = np.sort(np.random.default_rng(seed).choice(len(arr), size=min(n, len(arr)), replace=False))
    x = np.asarray(arr[idx], dtype=np.float32)
    return {"mean": round(float(x.mean()), 5), "std": round(float(x.std()), 5), "n_segments": int(len(idx))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="features")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--per-class", type=int, default=None, help="train clips per class (subset mode)")
    ap.add_argument("--per-class-eval", type=int, default=None, help="val/test clips per class (subset mode)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root = data_dir()
    out_dir = root / args.name
    if (out_dir / "meta.json").exists() and not args.force:
        print(f"{out_dir}/meta.json exists — use --force to rebuild.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"preprocess_{args.name}.log"
    log_path.parent.mkdir(exist_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    df = pd.read_csv(root / "index.csv")
    df = subsample(df, args.per_class, args.per_class_eval, args.seed)
    log(f"🐦 Preprocessing {len(df):,} clips into {out_dir} with {args.workers} workers")

    counts = {}
    for split in args.splits:
        counts[split] = build_split(df, split, out_dir, args.workers, log)

    stats = feature_stats(out_dir / "train.npy") if "train" in args.splits else None
    if stats:
        log(f"   train feature stats (after AST normalisation): {stats}")
    meta = {
        "features": F.feature_spec(),
        "train_stats": stats,
        "classes": class_names(),
        "counts": counts,
        "subset": {"per_class": args.per_class, "per_class_eval": args.per_class_eval},
        "seed": args.seed,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"✅ Done: {json.dumps(counts)}")


if __name__ == "__main__":
    main()
