#!/bin/bash
#SBATCH --job-name=mahjong-call-repair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/call-train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/train_call_repair.py --config call-repair-20261109.json
