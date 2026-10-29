#!/bin/bash
#SBATCH --job-name=mahjong-point-ranking
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/head-rl-score-ranking-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/train_score_pairwise.py --config head-rl-score-relative-ranking.json
if [ -f runs/head-rl-score-relative-ranking/confirmation-selection.json ]; then
  .venv/bin/python scripts/prepare_benchmark_confirmation.py \
    --source runs/head-rl-score-relative-ranking --out runs/head-rl-score-relative-ranking-final
  sbatch --parsable --partition=ws-ia --time=02:00:00 \
    --dependency="afterok:${SLURM_JOB_ID}" --kill-on-invalid-dep=yes \
    --nodelist="${SLURMD_NODENAME}" head_rl_confirm.slurm runs/head-rl-score-relative-ranking-final \
    | tee runs/head-rl-score-relative-ranking-final/confirmation-job-id.txt
fi
