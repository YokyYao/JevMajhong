#!/bin/bash
# Register and verify public-input parity in a separate GPU job first.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-curriculum-stage1-context
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/curriculum-stage1-context-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_CONTEXT_REGISTERED:?Set an explicitly admitted context fit manifest}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-context-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 3m python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 340m python tools/train_curriculum_context.py --registered "$JEV_CONTEXT_REGISTERED"
