#!/bin/bash
# Data preparation follows reproduction/README.md; use a fresh v2 output directory.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-curriculum-stage1-stable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/curriculum-stage1-stable-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_BASE_MODEL_DIR:?Set the pinned Qwen snapshot directory}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-stable-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 3m python tools/check_gpu.py
timeout --signal=TERM --kill-after=15s 3m python -m unittest discover -s tests/core -p test_curriculum_order.py -v
timeout --signal=TERM --kill-after=15s 5m python tools/bootstrap_curriculum.py
timeout --signal=TERM --kill-after=15s 5m python tools/register_curriculum_stable.py --config configs/training/curriculum-stage1-stable-v2.json --base-path "$JEV_BASE_MODEL_DIR" --reference-path "${JEV_REFERENCE_MODEL_DIR:-models/jev-play-v2}"
timeout --signal=TERM --kill-after=30s 340m python tools/train_curriculum_stable.py --registered runs/curriculum-optimization-v2/fit-plan/registered.json
