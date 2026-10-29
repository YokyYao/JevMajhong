#!/bin/bash
#SBATCH --job-name=mahjong-calibrated-cache
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=logs/head-rl-calibrated-cache-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/train_calibrated_cache.py --config head-rl-score-calibrated-cache.json
if [ -f runs/head-rl-score-calibrated-cache/confirmation-selection.json ]; then
  sbatch --parsable --partition=ws-ia --time=02:00:00 \
    --dependency="afterok:${SLURM_JOB_ID}" --kill-on-invalid-dep=yes \
    --nodelist="${SLURMD_NODENAME}" head_rl_confirm.slurm runs/head-rl-score-calibrated-cache \
    | tee runs/head-rl-score-calibrated-cache/confirmation-job-id.txt
fi
