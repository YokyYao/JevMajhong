#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-daiminkan-context
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=logs/curriculum-daiminkan-context-teacher-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"teacher_labels\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-daiminkan-context-teacher-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 12m .venv/bin/python tools/label_curriculum_daiminkan_context_v1.py
