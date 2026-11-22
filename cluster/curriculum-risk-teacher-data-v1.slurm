#!/bin/bash
# Set JEV_RISK_PHASE=collect or audit; outputs must be fresh.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-risk-teacher-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --exclude=ws-l1-010,ws-l2-013
#SBATCH --output=logs/curriculum-risk-teacher-data-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_RISK_PHASE:?Set collect or audit}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-risk-teacher-data-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_RISK_PHASE" in
  collect) timeout --signal=TERM --kill-after=15s 40m .venv/bin/python tools/collect_curriculum_risk_teacher.py --config configs/training/curriculum-risk-mortal-data-v1.json ;;
  audit) timeout --signal=TERM --kill-after=15s 40m .venv/bin/python tools/audit_curriculum_risk_teacher.py --source runs/curriculum-risk-teacher-data-v1/data --out runs/curriculum-risk-teacher-data-v1/audit-v1 ;;
  *) exit 2 ;;
esac
