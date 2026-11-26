#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-early-final-confirmation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l2-010,ws-l1-018,ws-l2-011,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-confirmation-early-run-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-confirmation-early-run-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 1180m .venv/bin/python tools/curriculum_confirmation_early_v1.py --phase run
