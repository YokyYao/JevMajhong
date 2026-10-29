#!/bin/bash
#SBATCH --job-name=mahjong-anchor-diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/training-anchor-diagnosis-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/diagnose_training_anchor.py \
  --source runs/head-rl-score-single-decision-final \
  --out runs/single-decision-anchor-diagnosis
