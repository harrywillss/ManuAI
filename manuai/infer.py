"""Classify audio files with a trained run.

    python -m manuai.infer --run models/finetuned/ast-v2-1 path/to/recording.wav [...]
"""
from __future__ import annotations

import argparse
import json

from .inference import load_run, pick_device, predict_file


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (contains manuai_model.json)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)

    device = pick_device()
    model, spec, collate = load_run(args.run, device)
    for path in args.files:
        res = predict_file(path, model, spec, collate, device, top_k=args.top_k)
        if args.json:
            print(json.dumps(res))
            continue
        tops = ", ".join(f"{t['class']} {t['prob']:.0%}" for t in res["top"])
        print(f"🐦 {path}  ({res['segments']} × 5 s)  →  {tops}")


if __name__ == "__main__":
    main()
