"""Audio → features. The single source of truth for training, evaluation and the app.

Pipeline (identical everywhere):
  1. decode → mono float32 → resample to 16 kHz (torchaudio sinc resampler)
  2. cut into 5 s segments (no overlap; an end-aligned extra segment if the leftover
     tail is ≥ 50 % of a segment); clips shorter than 5 s are *tiled* (looped) to
     5 s rather than zero-padded, so digital silence never leaks the clip length
  3. per-segment DC removal
  4. Kaldi fbank exactly as HF `ASTFeatureExtractor` (128 mel, 25 ms Hanning window,
     10 ms shift, htk_compat, no dither) → 498 frames for a 5 s segment
  5. AST normalisation  (x − mean) / (2·std)  with the AudioSet statistics

Frame padding to the model's input length (512 for AST) happens at load time with
`pad_frames`, using the same padding value ASTFeatureExtractor uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR: int = 16_000
SEG_SECONDS: float = 5.0
SEG_SAMPLES: int = int(SR * SEG_SECONDS)  # 80 000
MIN_TAIL: float = 0.5                      # keep an end-aligned segment if tail ≥ 50 %
MIN_SAMPLES: int = int(0.1 * SR)           # ignore clips shorter than 100 ms

N_MELS: int = 128
FRAME_SHIFT_MS: float = 10.0
FRAME_LENGTH_MS: float = 25.0
N_FRAMES: int = int((SEG_SAMPLES - SR * FRAME_LENGTH_MS / 1000) // (SR * FRAME_SHIFT_MS / 1000)) + 1  # 498

AST_MEAN: float = -4.2677393
AST_STD: float = 4.5689974
# ASTFeatureExtractor pads the *raw* fbank with zeros before normalising.
PAD_VALUE: float = (0.0 - AST_MEAN) / (AST_STD * 2)

FEATURE_VERSION = "fbank128-16k-5s-v1"


def feature_spec() -> dict:
    """Everything an inference client needs to reproduce the features."""
    return {
        "feature_version": FEATURE_VERSION,
        "sample_rate": SR,
        "segment_seconds": SEG_SECONDS,
        "n_mels": N_MELS,
        "frame_shift_ms": FRAME_SHIFT_MS,
        "frame_length_ms": FRAME_LENGTH_MS,
        "n_frames": N_FRAMES,
        "window": "hanning",
        "htk_compat": True,
        "dither": 0.0,
        "normalise_mean": AST_MEAN,
        "normalise_std": AST_STD,
        "normalise_formula": "(x - mean) / (2 * std)",
        "pad_value": PAD_VALUE,
        "short_clip_policy": "tile",
        "min_tail_fraction": MIN_TAIL,
    }


def load_audio(path: str | Path) -> torch.Tensor:
    """Decode any libsndfile-readable file to a mono float32 1-D tensor at 16 kHz."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return to_mono_16k(torch.from_numpy(data.T.copy()), sr)


def to_mono_16k(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """(channels, n) or (n,) float tensor at `sr` → mono 1-D float32 at 16 kHz."""
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.to(torch.float32)
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    return waveform


def segment_starts(n: int, seg: int = SEG_SAMPLES, hop: int = SEG_SAMPLES) -> list[int]:
    """Start indices of segments for a signal of length n (see module docstring)."""
    if n <= seg:
        return [0]
    starts = list(range(0, n - seg + 1, hop))
    tail = n - (starts[-1] + seg)
    if tail >= MIN_TAIL * seg:
        starts.append(n - seg)
    return starts


def n_segments(n_samples_16k: int) -> int:
    if n_samples_16k < MIN_SAMPLES:
        return 0
    return len(segment_starts(n_samples_16k))


def segments(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Cut a 1-D 16 kHz waveform into SEG_SAMPLES-long segments (tiling short clips)."""
    n = waveform.numel()
    if n < MIN_SAMPLES:
        return []
    if n < SEG_SAMPLES:
        reps = -(-SEG_SAMPLES // n)  # ceil
        return [waveform.repeat(reps)[:SEG_SAMPLES]]
    return [waveform[s:s + SEG_SAMPLES] for s in segment_starts(n)]


def fbank(segment: torch.Tensor) -> np.ndarray:
    """One SEG_SAMPLES segment → normalised (N_FRAMES, N_MELS) float32 features."""
    seg = segment - segment.mean()
    feats = torchaudio.compliance.kaldi.fbank(
        seg.unsqueeze(0),
        htk_compat=True,
        sample_frequency=SR,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=N_MELS,
        dither=0.0,
        frame_shift=FRAME_SHIFT_MS,
        frame_length=FRAME_LENGTH_MS,
    )
    feats = (feats - AST_MEAN) / (AST_STD * 2)
    return feats.numpy().astype(np.float32)


def features_from_waveform(waveform: torch.Tensor, sr: int) -> np.ndarray:
    """Any waveform → (n_segments, N_FRAMES, N_MELS) float32. Use this in the app."""
    wave = to_mono_16k(waveform, sr)
    segs = segments(wave)
    if not segs:
        return np.zeros((0, N_FRAMES, N_MELS), dtype=np.float32)
    return np.stack([fbank(s) for s in segs])


def features_from_file(path: str | Path) -> np.ndarray:
    segs = segments(load_audio(path))
    if not segs:
        return np.zeros((0, N_FRAMES, N_MELS), dtype=np.float32)
    return np.stack([fbank(s) for s in segs])


def pad_frames(x: np.ndarray | torch.Tensor, target: int) -> torch.Tensor:
    """Pad (or centre-crop) the time axis of (T, F) features to `target` frames."""
    t = torch.as_tensor(x, dtype=torch.float32)
    n = t.shape[0]
    if n == target:
        return t
    if n > target:
        start = (n - target) // 2
        return t[start:start + target]
    pad = torch.full((target - n, t.shape[1]), PAD_VALUE, dtype=torch.float32)
    return torch.cat([t, pad], dim=0)
