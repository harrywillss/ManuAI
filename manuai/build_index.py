"""Build `data/index.csv`: one row per archive clip, with its class, recording group and split.

    python -m manuai.build_index            # writes data/index.csv + data/split_summary.md

Grouping (what must never straddle train/val/test):
  DOC_001  recorder id         e.g. AE142_BIRP_120321_124316_000.flac → DOC_001:AE142
           (one physical recorder ≈ one site, so the test set is made of unseen sites)
  DOC_002  field recording     <UUID>_<n>.flac → DOC_002:<UUID>
           (all DOC_002 audio is from one region, so site leakage there is unavoidable)
  DOC_003  Xeno-canto id       silver3_895593_1.flac → XC:895593

The split is a StratifiedGroupKFold (default 7 folds, seed 21): fold 0 → test,
fold 1 → val, the rest → train (≈ 14 / 14 / 72 %).
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import ARCHIVE_SOURCES, REPO_ROOT, SEED, archive_source_dir, data_dir
from .labels import OTHER, class_names, ebird_to_class

_XC_RE = re.compile(r"_(\d+)_\d+\.[a-z0-9]+$")
_TAIL_RE = re.compile(r"_\d+\.[a-z0-9]+$")


def _parse_secondary(value) -> list[str]:
    if isinstance(value, str) and value.startswith("["):
        try:
            return [str(x) for x in ast.literal_eval(value)]
        except (ValueError, SyntaxError):
            return []
    return []


def _ids(source: str, filename: str) -> tuple[str, str]:
    """(recording_id, group) for one archive clip."""
    base = filename.split("/")[-1]
    short = source[:7]
    if short == "DOC_003":
        m = _XC_RE.search(base)
        xc = m.group(1) if m else _TAIL_RE.sub("", base)
        rid = f"XC:{xc}"
        return rid, rid
    stem = _TAIL_RE.sub("", base)
    rid = f"{short}:{stem}"
    if short == "DOC_001":
        return rid, f"{short}:{stem.split('_')[0]}"
    return rid, rid


def load_archive() -> pd.DataFrame:
    frames = []
    for source in ARCHIVE_SOURCES:
        src_dir = archive_source_dir(source)
        meta = pd.read_csv(next(src_dir.glob("*_metadata.csv")))
        meta = meta[["filename", "primary_label", "secondary_labels"]].copy()
        meta["source"] = source[:7]
        meta["path"] = [
            str((src_dir / "train_audio" / f).relative_to(REPO_ROOT)) for f in meta["filename"]
        ]
        ids = [_ids(source, f) for f in meta["filename"]]
        meta["recording_id"] = [i[0] for i in ids]
        meta["group"] = [i[1] for i in ids]
        frames.append(meta)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"primary_label": "ebird"})
    df["secondary"] = df["secondary_labels"].map(_parse_secondary)
    return df.drop(columns=["secondary_labels", "filename"])


def assign_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["label"] = df["ebird"].map(ebird_to_class)
    stats = {"clips_total": len(df), "dropped_unidentified": int(df["label"].isna().sum())}
    df = df[df["label"].notna()].copy()

    # An `Other` clip with a target species in the background would teach the model
    # that the target's call means `Other`, so drop those.
    targets = set(class_names(include_other=False))
    sec_has_target = df["secondary"].map(lambda s: any(ebird_to_class(c) in targets for c in s))
    conflict = (df["label"] == OTHER) & sec_has_target
    stats["dropped_other_with_target_background"] = int(conflict.sum())
    df = df[~conflict].copy()
    df["secondary"] = df["secondary"].map(lambda s: ";".join(s))
    return df.reset_index(drop=True), stats


def grouped_split(df: pd.DataFrame, n_folds: int = 7, seed: int = SEED) -> pd.Series:
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for i, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), df["label"], df["group"])):
        fold[test_idx] = i
    split = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    return pd.Series(split, index=df.index, name="split")


def summarise(df: pd.DataFrame) -> str:
    rows = []
    for label in class_names():
        sub = df[df["label"] == label]
        r = {"class": label}
        for sp in ("train", "val", "test"):
            s = sub[sub["split"] == sp]
            r[f"{sp}_clips"] = len(s)
            r[f"{sp}_recs"] = s["recording_id"].nunique()
            r[f"{sp}_groups"] = s["group"].nunique()
        rows.append(r)
    t = pd.DataFrame(rows)
    total = {"class": "**Total**", **{c: int(t[c].sum()) for c in t.columns if c != "class"}}
    for sp in ("train", "val", "test"):  # groups/recordings can hold several classes: count unique
        total[f"{sp}_recs"] = int(df.loc[df["split"] == sp, "recording_id"].nunique())
        total[f"{sp}_groups"] = int(df.loc[df["split"] == sp, "group"].nunique())
    t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
    header = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    body = "".join("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in row) + " |\n"
                   for row in t.itertuples(index=False))
    return header + body


def check_no_leak(df: pd.DataFrame) -> None:
    for key in ("group", "recording_id"):
        n = df.groupby(key)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            raise AssertionError(f"{len(bad)} {key}s appear in more than one split, e.g. {bad.index[:3].tolist()}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    out_dir = data_dir(create=True)
    print("📚 Reading archive metadata …")
    df, stats = assign_labels(load_archive())
    print(f"   {stats['clips_total']:,} clips; dropped {stats['dropped_unidentified']:,} unidentified, "
          f"{stats['dropped_other_with_target_background']:,} Other-with-target-background")
    print(f"✂️  Grouped split ({args.folds} folds, seed {args.seed}) over {df['group'].nunique():,} groups …")
    df["split"] = grouped_split(df, args.folds, args.seed)
    check_no_leak(df)

    cols = ["path", "source", "ebird", "label", "secondary", "recording_id", "group", "split"]
    df[cols].to_csv(out_dir / "index.csv", index=False)
    summary = summarise(df)
    (out_dir / "split_summary.md").write_text(
        f"# Split summary\n\nClass set: {', '.join(class_names())}\n\n"
        f"Grouped StratifiedGroupKFold, {args.folds} folds, seed {args.seed}; fold 0 = test, fold 1 = val.\n\n"
        f"Stats: {stats}\n\n{summary}"
    )
    print(summary)
    print(f"✅ Wrote {out_dir / 'index.csv'} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
