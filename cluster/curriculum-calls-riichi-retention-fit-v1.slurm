#!/bin/bash
# Submit only after actual qualified-parent registration and launch checks.
# The older calls template invokes the hard-CE trainer and has an8h budget.
# This explicit entry retains RI class loss and allows the full larger course.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-calls-RI-retention-fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018
#SBATCH --output=logs/curriculum-calls-RI-retention-fit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_CALLS_REGISTERED:?Set the actual qualified RI-retention calls registration}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-calls-RI-retention-fit-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 2m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 710m .venv/bin/python tools/train_curriculum_calls_riichi_retention_v1.py --registered "$JEV_CALLS_REGISTERED"
