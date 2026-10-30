#!/bin/bash
#SBATCH --job-name=mahjong-diagonal-rep-finish
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/diagonal-replication-finish-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/finish_diagonal_replication.py --run "${1:-runs/diagonal-value-fixed-replication-20261030}"
