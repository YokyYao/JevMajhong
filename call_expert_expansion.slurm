#!/bin/bash
#SBATCH --job-name=mahjong-expert-expansion
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/call-expert-expansion-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_call_repair.py --config "${1:-call-expert-expansion-20261110.json}"
