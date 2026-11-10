#!/bin/bash
#SBATCH --job-name=mahjong-call-watch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:05:00
#SBATCH --output=logs/call-prior-watch-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/watch_call_repair.py --config "${3:-call-prior-followup-20261109.json}" --train-job "${1:?Train job}" --eval-job "${2:?Eval job}" --max-seconds "${4:-21600}"
