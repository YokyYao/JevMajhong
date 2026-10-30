#!/bin/bash
#SBATCH --job-name=mahjong-diagonal-advance
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/diagonal-value-advance-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/advance_diagonal_development.py --run "${1:-runs/head-rl-diagonal-value-20261030}"
