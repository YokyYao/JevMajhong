#!/bin/bash
#SBATCH --job-name=mahjong-call-refit
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/call-prior-refit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
training_config="${1:-call-prior-followup-20261109.json}"
.venv/bin/python scripts/prepare_call_prior.py --config "$training_config"
.venv/bin/python scripts/refit_call_query.py --config "$training_config"
