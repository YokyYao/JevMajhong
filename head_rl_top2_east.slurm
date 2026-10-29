#!/bin/bash
#SBATCH --job-name=mahjong-top2-east
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=logs/head-rl-top2-east-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/smoke_head_rl.py --config head-rl-top2-east.json --out runs/top2-east-smoke.json
.venv/bin/python scripts/train_head_rl.py --config head-rl-top2-east.json
if [ -f runs/head-rl-top2-east/confirmation-selection.json ]; then
  sbatch --parsable --partition=ws-ia --time=01:00:00 head_rl_confirm.slurm runs/head-rl-top2-east \
    | tee runs/head-rl-top2-east/confirmation-job-id.txt
fi
