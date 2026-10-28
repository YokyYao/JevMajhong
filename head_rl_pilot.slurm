#!/bin/bash
#SBATCH --job-name=mahjong-head-rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/head-rl-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/smoke_head_rl.py
.venv/bin/python scripts/train_head_rl.py
if [ -f runs/head-rl-pilot-v1/confirmation-selection.json ]; then
  sbatch --parsable --partition=ws-ia head_rl_confirm.slurm | tee runs/head-rl-pilot-v1/confirmation-job-id.txt
fi
