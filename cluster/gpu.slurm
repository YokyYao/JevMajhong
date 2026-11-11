#!/bin/bash
#SBATCH --job-name=jev-simulation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jev-simulation-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python tools/check_gpu.py
.venv/bin/python tools/run_experiment.py "${1:?collect or evaluate}" --config "${2:?configuration JSON}"
