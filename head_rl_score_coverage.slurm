#!/bin/bash
#SBATCH --job-name=mahjong-score-cover
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/head-rl-score-coverage-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/smoke_head_rl.py --config head-rl-score-coverage.json --out runs/score-coverage-smoke.json
.venv/bin/python scripts/train_head_rl.py --config head-rl-score-coverage.json
if [ -f runs/head-rl-score-coverage/confirmation-selection.json ]; then
  sbatch --parsable --partition=ws-ia --time=02:00:00 head_rl_confirm.slurm runs/head-rl-score-coverage \
    | tee runs/head-rl-score-coverage/confirmation-job-id.txt
fi
