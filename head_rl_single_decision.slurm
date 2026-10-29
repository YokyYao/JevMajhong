#!/bin/bash
#SBATCH --job-name=mahjong-one-decision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/head-rl-single-decision-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/smoke_single_decision.py \
  --config head-rl-score-single-decision.json --out runs/single-decision-smoke.json
.venv/bin/python scripts/train_single_decision.py --config head-rl-score-single-decision.json
if [ -f runs/head-rl-score-single-decision/confirmation-selection.json ]; then
  .venv/bin/python scripts/prepare_benchmark_confirmation.py \
    --source runs/head-rl-score-single-decision --out runs/head-rl-score-single-decision-final
  sbatch --parsable --partition=ws-ia --time=02:00:00 \
    --dependency="afterok:${SLURM_JOB_ID}" --kill-on-invalid-dep=yes \
    --nodelist="${SLURMD_NODENAME}" head_rl_confirm.slurm runs/head-rl-score-single-decision-final \
    | tee runs/head-rl-score-single-decision-final/confirmation-job-id.txt
fi
