#!/bin/bash
#SBATCH --job-name=mahjong-score-rloo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/head-rl-score-rloo-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/smoke_head_rl.py --config head-rl-score-rloo.json --out runs/score-rloo-smoke.json
.venv/bin/python scripts/train_head_rl.py --config head-rl-score-rloo.json
if [ -f runs/head-rl-score-rloo/confirmation-selection.json ]; then
  sbatch --parsable --partition=ws-ia --time=01:00:00 head_rl_confirm.slurm runs/head-rl-score-rloo \
    | tee runs/head-rl-score-rloo/confirmation-job-id.txt
fi
