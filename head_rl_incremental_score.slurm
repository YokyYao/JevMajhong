#!/bin/bash
#SBATCH --job-name=mahjong-incremental-score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/head-rl-incremental-score-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/train_incremental_score.py --config head-rl-score-incremental.json
if .venv/bin/python -c 'import json,sys; sys.exit(0 if json.load(open("runs/head-rl-score-incremental-20261030/incremental-selection.json"))["selected"] is not None else 1)'; then
  .venv/bin/python scripts/prepare_positive_confirmation.py \
    --source runs/head-rl-score-incremental-20261030 --out runs/head-rl-score-incremental-20261030-final
  sbatch --parsable --partition=ws-ia --time=02:00:00 \
    --dependency="afterok:${SLURM_JOB_ID}" --kill-on-invalid-dep=yes \
    --nodelist="${SLURMD_NODENAME}" head_rl_confirm.slurm runs/head-rl-score-incremental-20261030-final \
    | tee runs/head-rl-score-incremental-20261030-final/confirmation-job-id.txt
fi
