"""Paths and project configuration.

`ManuAI.config` (YAML) holds `data_directory`. It may be absolute or relative; a
relative path is resolved against the repository root, so the same config works on
the Mac, in a Linux VM that mounts the repo elsewhere, and on a cloud GPU box.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = REPO_ROOT / "ManuAI.config"
TYPES_PATH: Path = REPO_ROOT / "types.yaml"
ARCHIVE_DIR: Path = REPO_ROOT / "archive"
MODELS_DIR: Path = REPO_ROOT / "models"

ARCHIVE_SOURCES: tuple[str, ...] = (
    "DOC_001_Tier1",
    "DOC_002_DuncanBayParinga",
    "DOC_003_XenoCanto",
)

SEED: int = 21


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} not found — create it with `data_directory: data`.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def data_dir(create: bool = False) -> Path:
    """Return the configured data directory.

    Per AGENTS.md §5: if it does not exist we stop rather than silently falling back,
    unless the caller explicitly asks to create it (the index builder does).
    """
    raw = load_config().get("data_directory")
    if not raw:
        raise KeyError("`data_directory` missing from ManuAI.config")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"Data directory {path} does not exist. Run `python -m manuai.build_index` first, "
                "or fix ManuAI.config."
            )
    return path


def features_dir(name: str = "features", create: bool = False) -> Path:
    path = data_dir(create=create) / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def archive_source_dir(source: str) -> Path:
    """`archive/<source>/<source>` — the Kaggle layout nests each source twice."""
    return ARCHIVE_DIR / source / source
