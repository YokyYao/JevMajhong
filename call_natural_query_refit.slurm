#!/bin/bash
#SBATCH --job-name=mahjong-natural-query
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/call-natural-query-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/refit_call_query.py --config call-natural-query-20261110.json
