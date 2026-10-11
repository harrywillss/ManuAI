"""Train AST (default) or ViT with LoRA on the feature cache.

    python -m manuai.train --model ast --run-name ast-v2-1               # full run
    python -m manuai.train --model ast --smoke --features features_smoke  # ~5 min check

Fixes from claude/model-setup-review.md:
  * plain cross-entropy (label smoothing 0.1), NO class weights — imbalance is handled by
    a √-frequency sampler with a per-recording cap
  * "virtual epochs" of --epoch-size sampled segments, evaluated on a capped val subset
  * checkpoints selected on macro-F1; final report on the full val and test sets,
    segment- and recording-level
  * AST at 512 frames with cropped position embeddings; LoRA on all encoder linears
  * seeds everywhere; device-agnostic (cuda → mps → cpu); workers via importable code
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader
from transformers import (EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

from . import features as F
from .config import MODELS_DIR, SEED, data_dir
from .data import (AugmentConfig, Collate, SegmentDataset, balanced_sampler, class_sampling_share,
                   read_rows, spread_subset)
from .evaluate import evaluate_split
from .inference import free_device_memory, pick_device
from .metrics import make_compute_metrics
from .models import AST_FRAMES, build_ast, build_vit, trainable_summary


def device_memory_gb(device: str) -> float | None:
    try:
        if device == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001
        return None
    return None


class ManuTrainer(Trainer):
    def __init__(self, *args, train_sampler=None, collate=None, label_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler = train_sampler
        self._collate = collate
        self._ls = label_smoothing

    def _loader(self, ds, batch_size, sampler=None):
        nw = self.args.dataloader_num_workers
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                          collate_fn=self._collate, num_workers=nw, pin_memory=False,
                          persistent_workers=nw > 0 and sampler is not None,
                          prefetch_factor=4 if nw > 0 else None)

    def evaluate(self, *args, **kwargs):
        free_device_memory(self.args.device.type)
        return super().evaluate(*args, **kwargs)

    def get_train_dataloader(self):
        return self._loader(self.train_dataset, self.args.per_device_train_batch_size, self._sampler)

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        return self._loader(ds, self.args.per_device_eval_batch_size)

    def get_test_dataloader(self, test_dataset):
        return self._loader(test_dataset, self.args.per_device_eval_batch_size)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if num_items_in_batch is not None and model.training:
            loss = TF.cross_entropy(logits, labels, label_smoothing=self._ls, reduction="sum")
            loss = loss / num_items_in_batch
        else:
            loss = TF.cross_entropy(logits, labels, label_smoothing=self._ls)
        return (loss, outputs) if return_outputs else loss


def stop_requested() -> bool:
    """Kill switch: `touch data/logs/STOP` stops any running/next training job cleanly."""
    try:
        return (data_dir() / "logs" / "STOP").exists()
    except FileNotFoundError:
        return False


class StopFileCallback(TrainerCallback):
    """Ends training at the next step if data/logs/STOP exists (the best checkpoint is still loaded)."""

    def on_step_end(self, args, state, control, **kw):
        if stop_requested():
            print("🛑 data/logs/STOP found — stopping training.")
            control.should_training_stop = True
        return control


class RunLogger(TrainerCallback):
    """Adds sec/step and device memory to every log line and mirrors logs to JSONL."""

    def __init__(self, path: Path, device: str):
        self.path = path
        self.device = device
        self.t_last = None
        self.step_last = 0

    def on_train_begin(self, args, state, control, **kw):
        self.t_last = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None:
            return
        now = time.time()
        if "loss" in logs and state.global_step > self.step_last:
            logs["sec_per_step"] = round((now - self.t_last) / (state.global_step - self.step_last), 3)
            self.t_last, self.step_last = now, state.global_step
        mem = device_memory_gb(self.device)
        if mem is not None:
            logs["device_mem_gb"] = round(mem, 2)
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": state.global_step, "time": now, **logs}) + "\n")


def make_training_args(**kwargs) -> TrainingArguments:
    """TrainingArguments that work on transformers 4.x and 5.x (unknown keys are dropped)."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    dropped = sorted(k for k in kwargs if k not in known)
    if dropped:
        print(f"   (TrainingArguments: ignoring unsupported {dropped})")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in known})


def versions() -> dict:
    import peft
    import transformers
    import torchaudio
    return {"python": sys.version.split()[0], "torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "platform": platform.platform()}


def main(argv: list[str] | None = None) -> Path | None:
    """CLI entry point; also callable from notebooks: main(["--model", "ast", "--smoke"]). Returns the run dir."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ast", "vit"], default="ast")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--features", default="features", help="feature cache dir inside data/")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--epoch-size", type=int, default=30_000, help="segments sampled per virtual epoch")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5, help="class sampling ∝ n_c^alpha")
    ap.add_argument("--per-recording-cap", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-per-class", type=int, default=200, help="capped val subset size per class")
    ap.add_argument("--final-eval-splits", nargs="+", default=["test"], choices=["val", "test"],
                    help="full-split evaluation after training (val is ~1 h on the M1)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--train-per-class", type=int, default=None, help="limit training rows (debug)")
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 40 steps, small eval, timing report")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    if stop_requested():
        print("🛑 data/logs/STOP exists — not starting. Delete it to allow training.")
        return
    if args.batch_size is None:
        args.batch_size = 8 if args.model == "ast" else 32
    if args.grad_accum is None:
        args.grad_accum = 4 if args.model == "ast" else 1
    if args.smoke:
        args.max_steps = 40 if args.max_steps < 0 else args.max_steps
        args.eval_per_class = min(args.eval_per_class, 40)
        args.warmup_steps = min(args.warmup_steps, 5)
        args.epochs = 1
    run_name = args.run_name or f"{args.model}-{time.strftime('%Y%m%d-%H%M')}{'-smoke' if args.smoke else ''}"

    set_seed(args.seed)
    device = pick_device()
    feat_dir = data_dir() / args.features
    meta = json.loads((feat_dir / "meta.json").read_text())
    if meta["features"]["feature_version"] != F.FEATURE_VERSION:
        raise RuntimeError(f"Feature cache {meta['features']['feature_version']} ≠ code {F.FEATURE_VERSION}; rebuild it.")
    classes: list[str] = meta["classes"]
    k = len(classes)
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}

    run_dir = MODELS_DIR / "finetuned" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = run_dir / "train_log.jsonl"

    train_rows = read_rows(feat_dir, "train")
    if args.train_per_class:
        train_rows = spread_subset(train_rows, args.train_per_class, args.seed)
    val_rows = read_rows(feat_dir, "val")
    test_rows = read_rows(feat_dir, "test")
    val_sub = spread_subset(val_rows, args.eval_per_class, args.seed)

    aug = AugmentConfig()
    train_ds = SegmentDataset(feat_dir, "train", train_rows, training=True, augment=aug)
    val_sub_ds = SegmentDataset(feat_dir, "val", val_sub)
    frames = AST_FRAMES if args.model == "ast" else F.N_FRAMES
    stats = meta.get("train_stats") or {"mean": 0.0, "std": 0.5}
    collate = Collate(args.model, frames, vit_mean=stats["mean"], vit_std=stats["std"])

    effective = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, args.epoch_size // effective)
    if args.smoke:
        steps_per_epoch = args.max_steps
    sampler = balanced_sampler(train_rows, num_samples=steps_per_epoch * effective, alpha=args.alpha,
                               per_recording_cap=args.per_recording_cap, seed=args.seed)

    print(f"🐦 ManūAI — {args.model.upper()} run '{run_name}' on {device}")
    print(f"   classes ({k}): {', '.join(classes)}")
    print(f"   train {len(train_rows):,} segs | val {len(val_rows):,} (eval subset {len(val_sub):,}) | test {len(test_rows):,}")
    print(f"   batch {args.batch_size} × accum {args.grad_accum} = {effective}; {steps_per_epoch} steps/epoch × {args.epochs} epochs")
    print("   sampling share (top 5 / bottom 3):",
          class_sampling_share(train_rows, args.alpha).round(3).iloc[[0, 1, 2, 3, 4, -3, -2, -1]].to_dict())

    if args.model == "ast":
        model = build_ast(k, id2label, label2id, frames=AST_FRAMES, r=args.lora_r, alpha=args.lora_alpha)
    else:
        model = build_vit(k, id2label, label2id, r=args.lora_r, alpha=args.lora_alpha)
    print(f"   {trainable_summary(model)}")

    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(run_dir / "runs"))
    targs = make_training_args(
        output_dir=str(run_dir / "checkpoints"),
        logging_dir=str(run_dir / "runs"),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,
        save_strategy="steps",
        save_steps=steps_per_epoch,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=5 if args.smoke else 25,
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        fp16=False,
        bf16=False,
        use_cpu=(device == "cpu"),
        max_grad_norm=1.0,
    )
    trainer = ManuTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_sub_ds,
        compute_metrics=make_compute_metrics(k),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), RunLogger(log_jsonl, device),
                   StopFileCallback()],
        train_sampler=sampler, collate=collate, label_smoothing=args.label_smoothing,
    )

    t0 = time.time()
    result = trainer.train()
    train_secs = time.time() - t0
    history = trainer.state.log_history
    losses = [h for h in history if "loss" in h]
    print(f"✅ trained {trainer.state.global_step} steps in {train_secs / 60:.1f} min; "
          f"first loss {losses[0]['loss'] if losses else 'n/a'} (ln K = {math.log(k):.2f})")

    # Save adapters + label/feature spec before merging.
    model.save_pretrained(run_dir / "adapter")
    spec = {"model": args.model, "classes": classes, "features": F.feature_spec(),
            "input_frames": frames, "base": str(model.peft_config["default"].base_model_name_or_path),
            "vit_input_norm": {"mean": stats["mean"], "std": stats["std"]} if args.model == "vit" else None}
    (run_dir / "manuai_model.json").write_text(json.dumps(spec, indent=2))

    evals: dict = {
        "val_subset_best_macro_f1": trainer.state.best_metric,
        "val_subset_history": [{k: v for k, v in h.items() if k.startswith("eval_") or k in ("step", "epoch")}
                               for h in history if "eval_macro_f1" in h],
    }
    if stop_requested():
        print("🛑 STOP requested — skipping the final full evaluation (run `python -m manuai.evaluate` later).")
    elif not args.no_final_eval:
        free_device_memory(device)
        for split in args.final_eval_splits:
            rows = {"val": val_rows, "test": test_rows}[split]
            if args.smoke:
                rows = spread_subset(rows, args.eval_per_class, args.seed)
            rep, probs = evaluate_split(model, collate, feat_dir, split, rows, classes, device,
                                        batch_size=args.eval_batch_size, workers=min(2, args.workers))
            evals[split] = rep
            np.save(run_dir / f"probs_{split}.npy", probs.astype(np.float16))

    if not args.smoke:
        merged = model.merge_and_unload()
        merged.save_pretrained(run_dir / "merged")

    sps = [h["sec_per_step"] for h in map(json.loads, open(log_jsonl)) if "sec_per_step" in h]
    mem = [h["device_mem_gb"] for h in map(json.loads, open(log_jsonl)) if "device_mem_gb" in h]
    summary = {
        "run_name": run_name,
        "args": vars(args),
        "device": device,
        "versions": versions(),
        "classes": classes,
        "features": F.feature_spec(),
        "feature_cache": {"dir": args.features, "counts": meta["counts"], "subset": meta.get("subset")},
        "split": "StratifiedGroupKFold by recorder/recording, 7 folds, seed 21 (fold0=test, fold1=val)",
        "train_rows": len(train_rows), "val_rows": len(val_rows), "test_rows": len(test_rows),
        "steps": trainer.state.global_step,
        "train_minutes": round(train_secs / 60, 2),
        "sec_per_step_median": float(np.median(sps)) if sps else None,
        "device_mem_gb_max": float(max(mem)) if mem else None,
        "first_logged_loss": losses[0]["loss"] if losses else None,
        "ln_k": math.log(k),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": result.metrics,
        "eval": evals,
    }
    (run_dir / "training_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"📝 {run_dir / 'training_metrics.json'}")
    return run_dir


if __name__ == "__main__":
    main()
