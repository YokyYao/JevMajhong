#!/bin/bash
#SBATCH --job-name=mahjong-feature-point-value
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/feature-point-value-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
feature_point_value_config="${1:?Pass registered evaluation config}"
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/evaluate_feature_point_value.py --config "$feature_point_value_config" "${@:2}"
.venv/bin/python scripts/audit_feature_point_value.py --config "$feature_point_value_config"
