#!/bin/bash
#SBATCH --job-name=mahjong-call-natural-val
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/call-natural-validation-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_natural_validation.py --config call-natural-validation-20261110.json
.venv/bin/python scripts/audit_call_validation.py --config call-natural-validation-20261110.json
