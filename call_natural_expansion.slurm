#!/bin/bash
#SBATCH --job-name=mahjong-call-natural-data
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/call-natural-data-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_call_repair.py --config call-natural-expansion-20261110.json
