#!/bin/bash
#SBATCH --job-name=mahjong-fresh-score-curve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fresh-score-curve-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
fresh_score_config="${1:?Pass registered score-curve config}"
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/evaluate_fresh_score_curve.py --config "$fresh_score_config" "${@:2}"
.venv/bin/python scripts/audit_fresh_score_curve.py --config "$fresh_score_config"
