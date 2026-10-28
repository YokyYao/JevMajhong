#!/bin/bash
#SBATCH --job-name=mahjong-head-rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/head-rl-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/smoke_head_rl.py
.venv/bin/python scripts/train_head_rl.py
task_rl_out=$(.venv/bin/python -c 'import json; print(json.load(open("head-rl-pilot.json"))["out"])')
if [ -f "$task_rl_out/confirmation-selection.json" ]; then
  sbatch --parsable --partition=ws-ia head_rl_confirm.slurm "$task_rl_out" | tee "$task_rl_out/confirmation-job-id.txt"
fi
