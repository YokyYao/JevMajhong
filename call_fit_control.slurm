#!/bin/bash
#SBATCH --job-name=mahjong-call-fit-control
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/call-fit-control-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
control_config="${1:?Control config required}"
for control_arm in natural balanced; do
    .venv/bin/python scripts/check_gpu.py
    .venv/bin/python scripts/train_call_fit_control.py --config "$control_config" --arm "$control_arm"
done
