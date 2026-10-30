#!/bin/bash
#SBATCH --job-name=mahjong-point-bias
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/point-bias-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/train_point_bias.py fit --config head-rl-point-bias.json
if .venv/bin/python -c 'import json,sys; sys.exit(0 if json.load(open("runs/head-rl-point-bias-20261030/training-result.json"))["status"]=="fit_complete" else 1)'; then
  .venv/bin/python scripts/train_point_bias.py audit --run runs/head-rl-point-bias-20261030 \
    --out runs/head-rl-point-bias-20261030/fit-audit.json
  .venv/bin/python scripts/evaluate_diagonal_value.py --run runs/head-rl-point-bias-20261030
  .venv/bin/python scripts/train_point_bias.py advance --run runs/head-rl-point-bias-20261030
fi
