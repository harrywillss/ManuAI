"""Download pretrained base models into models/base/<name>/ so training stays offline.

    python scripts/fetch_base_models.py --list
    python scripts/fetch_base_models.py bird-mae-base ced-base

Each folder gets a SOURCE.json (repo id, exact revision, date) so runs are reproducible.
models/ is git-ignored: weights live on the Mac (or a cloud box), never in the repo.
Run this on a machine with internet access (the agent sandboxes can't reach Hugging Face).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "base"

# name -> (Hugging Face repo id, note)
REGISTRY: dict[str, tuple[str, str]] = {
    # in use
    "mit-ast": ("MIT/ast-finetuned-audioset-10-10-0.4593", "AST, AudioSet; current primary model (16 kHz)"),
    "google-vit": ("google/vit-base-patch16-224-in21k", "ImageNet ViT; v1/v2 baseline"),
    # candidates to trial (see claude/model-setup-review.md and the model research notes)
    "bird-mae-base": ("DBD-research-group/Bird-MAE-Base", "ViT-B/16 MAE pretrained on bird audio (BirdSet), 32 kHz, trust_remote_code"),
    "bird-mae-large": ("DBD-research-group/Bird-MAE-Large", "ViT-L/16 version of Bird-MAE"),
    "birdset-effnet-b1": ("DBD-research-group/EfficientNet-B1-BirdSet-XCL", "Bird-pretrained CNN; phone-sized"),
    "ced-base": ("mispeech/ced-base", "CED-Base, 86M, AudioSet mAP 50.0, 16 kHz, trust_remote_code"),
    "ced-mini": ("mispeech/ced-mini", "CED-Mini, 9.6M, AudioSet mAP 49.0; app candidate"),
    "perch2-torch": ("DAS-DeepAudioSegmenter/perch2-torch", "Perch 2.0 PyTorch port (EfficientNet-B3, 32 kHz, 5 s); embeddings + probe"),
}


def fetch(name: str, revision: str | None = None) -> Path:
    from huggingface_hub import HfApi, snapshot_download

    repo, note = REGISTRY[name]
    dest = BASE / name
    if (dest / "SOURCE.json").exists():
        print(f"✅ {name}: already in {dest.relative_to(ROOT)} (delete the folder to re-download)")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    sha = HfApi().model_info(repo, revision=revision).sha
    print(f"⬇️  {name}: {repo} @ {sha[:10]} → {dest.relative_to(ROOT)}")
    snapshot_download(repo_id=repo, revision=sha, local_dir=dest)
    (dest / "SOURCE.json").write_text(json.dumps(
        {"name": name, "repo_id": repo, "revision": sha, "note": note,
         "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    return dest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="registry names to download")
    ap.add_argument("--list", action="store_true", help="show the registry and what is already downloaded")
    ap.add_argument("--revision", default=None, help="pin a specific revision (single model only)")
    args = ap.parse_args(argv)

    if args.list or not args.names:
        for name, (repo, note) in REGISTRY.items():
            have = "✅" if (BASE / name).exists() else "  "
            print(f"{have} {name:18} {repo:48} {note}")
        return
    unknown = [n for n in args.names if n not in REGISTRY]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. Add them to REGISTRY or use --list.")
    for name in args.names:
        fetch(name, args.revision if len(args.names) == 1 else None)


if __name__ == "__main__":
    main()
