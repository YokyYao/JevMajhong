#!/bin/bash
#SBATCH --job-name=mahjong-diagonal-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:45:00
#SBATCH --output=logs/diagonal-value-development-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/evaluate_diagonal_value.py --run "${1:-runs/head-rl-diagonal-value-20261030}"
