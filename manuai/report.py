"""Turn a run's training_metrics.json into a readable REPORT.md (AGENTS.md §8 conventions).

    python -m manuai.report --run models/finetuned/ast-v2-1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

V1_NOTE = ("v1 capstone: 72.6 % accuracy / macro-F1 ≈ 0.73 on a *per-segment* split that leaked "
           "recordings between train and test — not comparable with these numbers.")


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def top_confusions(cm: list[list[int]], classes: list[str], k: int = 10) -> list[tuple[str, str, int, float]]:
    m = np.asarray(cm, dtype=float)
    rows = m.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    frac = m / rows
    np.fill_diagonal(frac, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-frac, axis=None), frac.shape))[0][:k]
    return [(classes[i], classes[j], int(m[i, j]), float(frac[i, j])) for i, j in idx if m[i, j] > 0]


def render(summary: dict) -> str:
    classes = summary["classes"]
    a = summary.get("args", {})
    feats = summary.get("features", {})
    lines = [f"# ManūAI run report — `{summary['run_name']}`", ""]
    lines += [
        f"- **Model:** {a.get('model', '?').upper()} + LoRA r={a.get('lora_r')} · lr {a.get('lr')} · "
        f"batch {a.get('batch_size')}×{a.get('grad_accum')} · {summary.get('steps')} steps "
        f"({summary.get('train_minutes')} min, median {summary.get('sec_per_step_median')} s/step, "
        f"peak {summary.get('device_mem_gb_max')} GB on {summary.get('device')})",
        f"- **Classes:** {len(classes)} ({', '.join(classes)})",
        f"- **Split:** {summary.get('split')}",
        f"- **Features:** {feats.get('feature_version')} — {feats.get('segment_seconds')} s segments, "
        f"{feats.get('n_mels')} mel @ {feats.get('sample_rate')} Hz",
        f"- **Sanity:** first logged loss {summary.get('first_logged_loss')} (ln K = {summary.get('ln_k', 0):.2f})",
        f"- **Best val-subset macro-F1:** {_pct(summary['eval'].get('val_subset_best_macro_f1'))} % "
        f"(checkpoint `{Path(str(summary.get('best_checkpoint'))).name}`)",
        "",
    ]
    hist = summary["eval"].get("val_subset_history") or []
    if hist:
        lines += ["## Validation subset per epoch", "", "| step | macro-F1 | accuracy | loss |", "|---:|---:|---:|---:|"]
        lines += [f"| {h.get('step')} | {_pct(h.get('eval_macro_f1'))} | {_pct(h.get('eval_accuracy'))} | "
                  f"{h.get('eval_loss', float('nan')):.3f} |" for h in hist]
        lines.append("")

    for split in ("test", "val"):
        rep = summary["eval"].get(split)
        if not rep:
            continue
        seg, rec = rep["segment"], rep["recording"]
        lines += [f"## {split.capitalize()} set (full)", "",
                  "| level | n | macro-F1 | accuracy | balanced acc | macro AUC |", "|---|---:|---:|---:|---:|---:|",
                  f"| segment | {seg['n']:,} | {_pct(seg['macro_f1'])} | {_pct(seg['accuracy'])} | "
                  f"{_pct(seg['balanced_accuracy'])} | {_pct(seg['macro_auc'])} |",
                  f"| recording | {rec['n']:,} | {_pct(rec['macro_f1'])} | {_pct(rec['accuracy'])} | "
                  f"{_pct(rec['balanced_accuracy'])} | {_pct(rec['macro_auc'])} |", ""]
        pc = rep["per_class"]
        rows = sorted(((c, pc[c]) for c in classes if c in pc), key=lambda kv: -kv[1]["f1-score"])
        lines += ["### Per class (segment level, sorted by F1)", "",
                  "| class | F1 | precision | recall | support |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {c} | {_pct(r['f1-score'])} | {_pct(r['precision'])} | {_pct(r['recall'])} | {int(r['support']):,} |"
                  for c, r in rows]
        conf = top_confusions(rep["confusion_matrix"], classes)
        if conf:
            lines += ["", "### Most common confusions (share of the true class)", "",
                      "| true | predicted as | segments | share |", "|---|---|---:|---:|"]
            lines += [f"| {t} | {p} | {n:,} | {_pct(f)} % |" for t, p, n, f in conf]
        lines.append("")

    lines += ["---", f"_{V1_NOTE}_", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--metrics", default=None, help="override path to training_metrics.json")
    args = ap.parse_args(argv)
    run = Path(args.run)
    summary = json.loads(Path(args.metrics or run / "training_metrics.json").read_text())
    out = run / "REPORT.md"
    out.write_text(render(summary))
    print(f"📝 {out}")


if __name__ == "__main__":
    main()
