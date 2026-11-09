#!/bin/bash
#SBATCH --job-name=mahjong-call-data
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/call-data-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv/bin/python -m unittest test_replay_call_repair test_semantic
.venv/bin/python scripts/prepare_call_repair.py --config call-repair-20261109.json
