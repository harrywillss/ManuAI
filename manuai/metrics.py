"""Metrics. Primary metric: macro-F1 (AGENTS.md §8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    present = np.unique(labels)
    if len(present) < 2:
        return float("nan")
    aucs = []
    for c in present:
        y = (labels == c).astype(int)
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def classification_metrics(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    preds = probs.argmax(axis=1)
    all_labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=all_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=all_labels, average="weighted", zero_division=0)),
        "macro_auc": _macro_auc(labels, probs),
        "n": int(len(labels)),
    }


def make_compute_metrics(n_classes: int):
    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        return classification_metrics(np.asarray(labels), softmax(np.asarray(logits)), n_classes)
    return compute_metrics


def recording_level(probs: np.ndarray, labels: np.ndarray, recording_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean segment probability per recording (a recording has a single label)."""
    df = pd.DataFrame(probs)
    df["rid"] = recording_ids
    df["y"] = labels
    g = df.groupby("rid", sort=False)
    p = g[list(range(probs.shape[1]))].mean().to_numpy()
    y = g["y"].agg(lambda s: s.mode().iloc[0]).to_numpy()
    return p, y


def full_report(probs: np.ndarray, labels: np.ndarray, recording_ids: np.ndarray, class_names: list[str]) -> dict:
    k = len(class_names)
    seg = classification_metrics(labels, probs, k)
    rp, ry = recording_level(probs, labels, recording_ids)
    rec = classification_metrics(ry, rp, k)
    preds = probs.argmax(axis=1)
    report = classification_report(labels, preds, labels=list(range(k)), target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(k)))
    return {"segment": seg, "recording": rec, "per_class": report, "confusion_matrix": cm.tolist()}
