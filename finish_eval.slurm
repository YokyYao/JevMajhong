#!/bin/bash
#SBATCH --job-name=mahjong-finish-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/finish-eval-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [[ -f runs/champion.json ]]; then
    echo "Primary job already completed evaluation."
    exit 0
fi
if [[ ! -f runs/semantic-2026-full/adapter_model.safetensors ]]; then
    echo "Training has no final checkpoint; evaluation was not attempted." >&2
    exit 1
fi
if [[ ! -f runs/semantic-2026-full/validation-full.json ]]; then
    .venv/bin/python evaluate.py runs/semantic-2026-full data/semantic-2026-v1/validation.jsonl --out runs/semantic-2026-full/validation-full.json
fi
.venv/bin/python arena.py --checkpoint runs/semantic-2026-full --seeds 32 --out runs/arena-2026-model --resume
.venv/bin/python arena.py --checkpoint runs/semantic-2026-full --guarded --seeds 32 --out runs/arena-2026-guarded --resume
.venv/bin/python promote.py runs/arena-2026-model runs/arena-2026-guarded --out runs/champion.json
