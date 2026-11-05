#!/bin/bash
#SBATCH --job-name=mahjong-desktop-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/desktop-model-smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
.venv/bin/python desktop/tests/smoke_model.py --out "${1:-runs/windows-assets/jev-model-smoke.json}"
