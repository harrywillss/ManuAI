"""Model construction: AST (primary) and ViT (baseline), both with LoRA.

AST runs at 512 input frames (5.12 s) instead of 1024. Its pretrained position
embeddings are centre-cropped along time, as in the original AST repo, rather than
re-initialised (which is what `max_length=512` + `ignore_mismatched_sizes` would do).
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from .config import MODELS_DIR

AST_BASE = MODELS_DIR / "base" / "mit-ast"
VIT_BASE = MODELS_DIR / "base" / "google-vit"
AST_FRAMES = 512

# LoRA goes on every linear layer inside the transformer blocks (q, k, v, attention
# output, MLP) but not on the classifier, which is trained in full via modules_to_save.
# Module names differ between transformers 4.x (`encoder.layer.0.attention.attention.query`)
# and 5.x (`layers.0.attention.q_proj`), so the targets are found by walking the model.
_BLOCK_RE = re.compile(r"(^|\.)(layer|layers)\.\d+\.")


def encoder_linear_names(model: nn.Module) -> list[str]:
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and _BLOCK_RE.search(n) and not n.startswith("classifier")]
    if not names:
        raise RuntimeError("No encoder Linear layers found for LoRA")
    return names


def lora_config(model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.1) -> LoraConfig:
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                      target_modules=encoder_linear_names(model), modules_to_save=["classifier"])


def crop_ast_positions(model, target_frames: int) -> None:
    """Centre-crop AST time position embeddings in place for a shorter input."""
    emb = model.audio_spectrogram_transformer.embeddings
    cfg = model.config
    f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
    t_old = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
    t_new = (target_frames - cfg.patch_size) // cfg.time_stride + 1
    if t_new == t_old:
        return
    if t_new > t_old:
        raise ValueError("crop_ast_positions only shortens the input")
    pe = emb.position_embeddings.data                     # (1, 2 + f*t, hidden), freq-major
    n_extra = pe.shape[1] - f_dim * t_old                 # cls + distillation tokens
    tok, grid = pe[:, :n_extra], pe[:, n_extra:].reshape(1, f_dim, t_old, -1)
    start = t_old // 2 - t_new // 2
    grid = grid[:, :, start:start + t_new]
    emb.position_embeddings = nn.Parameter(torch.cat([tok, grid.reshape(1, f_dim * t_new, -1)], dim=1))
    cfg.max_length = target_frames


def build_ast(num_labels: int, id2label: dict, label2id: dict, frames: int = AST_FRAMES,
              base: Path = AST_BASE, r: int = 16, alpha: int = 32, dropout: float = 0.1):
    from transformers import ASTForAudioClassification
    model = ASTForAudioClassification.from_pretrained(
        base, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,  # only the 527-way head is replaced
    )
    crop_ast_positions(model, frames)
    return get_peft_model(model, lora_config(model, r, alpha, dropout))


def build_vit(num_labels: int, id2label: dict, label2id: dict,
              base: Path = VIT_BASE, r: int = 16, alpha: int = 32, dropout: float = 0.1):
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        base, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return get_peft_model(model, lora_config(model, r, alpha, dropout))


def trainable_summary(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"trainable {train:,} / {total:,} ({100 * train / total:.2f}%)"
