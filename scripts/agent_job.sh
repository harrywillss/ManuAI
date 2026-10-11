#!/bin/bash
# JOB — first full AST run on the ready-built feature cache (data/features, built 12 Oct 09:21).
# ~6 × (600 steps ≈ 60 min + capped-val eval ≈ 9 min) + full test eval ≈ 1 h  →  ~7.5–8 h.
# Launch: open scripts/agent_runner.ipynb (kernel: ManuAI) and click Run All.
# Stop gracefully at any time: touch data/logs/STOP  (keeps the best checkpoint; final test eval skipped)
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p data/logs
rm -f data/logs/STOP

echo "== $(date) disk"; df -h . | tail -1
test -f data/features/meta.json || { echo "!! data/features missing — build it first (01_data.ipynb or manuai.preprocess)"; exit 1; }

echo "== $(date) AST full run"
"$PY" -m manuai.train --model ast --run-name ast-v2-1 \
    --epochs 6 --epoch-size 19200 --eval-per-class 150 --final-eval-splits test \
    || echo "!! AST run FAILED"

"$PY" -m manuai.report --run models/finetuned/ast-v2-1 || true
echo "== $(date) JOB DONE"; df -h . | tail -1
