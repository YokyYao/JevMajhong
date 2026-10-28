#!/bin/bash
#SBATCH --job-name=mahjong-semantic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/semantic-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/jevany train --config semantic-warmup.toml
.venv/bin/python evaluate.py runs/semantic-warmup data/semantic-v1/validation.jsonl --out runs/semantic-warmup/validation-full.json
.venv/bin/python arena.py --checkpoint runs/semantic-warmup --seeds 2 --out runs/arena-warmup
.venv/bin/python arena.py --checkpoint runs/semantic-warmup --guarded --seeds 2 --out runs/arena-warmup-guarded
