"""Fast checks for the invariants the whole pipeline relies on (no GPU, no archive needed
except where marked). Run from the repo root:  python -m pytest -q tests/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from manuai import features as F
from manuai.config import REPO_ROOT


# ---------- features: the single source of truth ----------

def test_frame_count_for_5s():
    assert F.N_FRAMES == 498
    x = F.fbank(torch.randn(F.SEG_SAMPLES) * 0.01)
    assert x.shape == (498, 128) and x.dtype == np.float32


@pytest.mark.parametrize("seconds,expected", [(0.05, 0), (3.0, 1), (5.0, 1), (5.79, 1), (8.0, 2), (12.0, 2), (12.6, 3)])
def test_segment_counts(seconds, expected):
    n = int(seconds * F.SR)
    assert F.n_segments(n) == expected
    assert len(F.segments(torch.zeros(n))) == expected


def test_short_clips_are_tiled_not_zero_padded():
    wave = torch.linspace(-1, 1, 3 * F.SR)
    seg = F.segments(wave)[0]
    assert seg.numel() == F.SEG_SAMPLES
    assert torch.equal(seg[3 * F.SR: 4 * F.SR], wave[: F.SR])  # loops back to the start


def test_matches_hf_ast_feature_extractor():
    transformers = pytest.importorskip("transformers")
    base = REPO_ROOT / "models" / "base" / "mit-ast"
    if not (base / "preprocessor_config.json").exists():
        pytest.skip("AST base model not available")
    fe = transformers.ASTFeatureExtractor.from_pretrained(base)
    wave = torch.randn(F.SEG_SAMPLES) * 0.05
    ours = F.fbank(wave)
    hf = fe((wave - wave.mean()).numpy(), sampling_rate=F.SR, return_tensors="np")["input_values"][0]
    assert np.abs(hf[: F.N_FRAMES] - ours).max() < 1e-4
    assert abs(float(hf[-1, 0]) - F.PAD_VALUE) < 1e-5


def test_pad_frames():
    x = np.zeros((498, 128), np.float32)
    padded = F.pad_frames(x, 512)
    assert padded.shape == (512, 128)
    assert torch.allclose(padded[498:], torch.full((14, 128), F.PAD_VALUE))


# ---------- labels ----------

def test_class_list_and_exclusions():
    pytest.importorskip("yaml")
    from manuai.labels import OTHER, class_names, ebird_to_class
    names = class_names()
    assert names[-1] == OTHER and len(names) == len(set(names))
    if not (REPO_ROOT / "archive").exists():
        pytest.skip("archive not available")
    assert ebird_to_class("parake") is None            # unidentified parrot
    assert ebird_to_class("tui1") == "Tui"
    assert ebird_to_class("sobkiw2") == ebird_to_class("nibkiw1") == "Kiwi"
    assert ebird_to_class("purswa6") == "Pukeko"
    assert ebird_to_class("kokako3") == OTHER


# ---------- split (needs data/index.csv) ----------

def test_index_has_no_group_leakage():
    from manuai.config import data_dir
    try:
        path = data_dir() / "index.csv"
    except FileNotFoundError:
        pytest.skip("no data directory")
    if not path.exists():
        pytest.skip("index not built")
    df = pd.read_csv(path)
    for key in ("group", "recording_id"):
        assert (df.groupby(key)["split"].nunique() == 1).all(), f"{key} crosses splits"
    assert set(df["split"]) == {"train", "val", "test"}
    assert "parake" not in set(df["ebird"])
    per_class_test = df[df["split"] == "test"].groupby("label")["recording_id"].nunique()
    assert per_class_test.min() >= 30, per_class_test.sort_values().head()
