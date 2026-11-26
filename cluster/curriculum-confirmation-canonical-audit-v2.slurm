#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-canonical-final-audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l2-010,ws-l1-018,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-confirmation-canonical-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_CONFIRMATION_RUN_JOB:?Actual successful final run job required}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-confirmation-canonical-audit-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=30s 80m .venv/bin/python tools/curriculum_confirmation_canonical_v2.py \
  --phase finish --run-job "$JEV_CONFIRMATION_RUN_JOB"
