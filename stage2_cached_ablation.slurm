#!/bin/bash
#SBATCH --job-name=mahjong-stage2-cached-ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage2-cached-ablation-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
stage2_cached_config="${1:?Pass registered reuse config}"
.venv/bin/python scripts/evaluate_stage2_cached_ablation.py --config "$stage2_cached_config" "${@:2}"
.venv/bin/python scripts/audit_stage2_ablation.py --config "$stage2_cached_config"
