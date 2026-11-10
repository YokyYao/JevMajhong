#!/bin/bash
#SBATCH --job-name=mahjong-call-feature-probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/call-feature-probe-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/probe_call_features.py --config "${1:-call-feature-probe-20261110.json}"
