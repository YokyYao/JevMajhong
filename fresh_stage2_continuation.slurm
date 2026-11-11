#!/bin/bash
#SBATCH --job-name=mahjong-stage2-fresh-continuation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage2-fresh-continuation-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
fresh_stage2_config="${1:?Pass registered continuation config}"
.venv/bin/python scripts/train_feature_continuation.py --config "$fresh_stage2_config"
