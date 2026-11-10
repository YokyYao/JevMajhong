#!/bin/bash
#SBATCH --job-name=mahjong-clean-data
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clean-data-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_clean_policy.py --config "${1:-clean-policy-20261110.json}"
