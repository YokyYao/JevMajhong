#!/bin/bash
# Submit only after the actual qualified-parent preflight and registration.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-calls-repair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013
#SBATCH --output=logs/curriculum-calls-repair-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_CALLS_REGISTERED:?Set the source-qualified calls registration}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-calls-repair-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 2m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 470m .venv/bin/python tools/train_curriculum_calls_repair.py --registered "$JEV_CALLS_REGISTERED"
