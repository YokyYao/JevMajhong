#!/bin/bash
#SBATCH --job-name=mahjong-gradient-diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gradient-baseline-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/diagnose_gradient_baseline.py \
  --source "${1:-runs/head-rl-score-carry8}" --out "${2:-runs/carry8-gradient-baseline.json}"
