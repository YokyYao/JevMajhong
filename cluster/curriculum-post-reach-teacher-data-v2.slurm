#!/bin/bash
# Set JEV_POST_REACH_PHASE=collect or audit; outputs must be fresh.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-post-reach-teacher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/curriculum-post-reach-teacher-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_POST_REACH_PHASE:?Set collect or audit}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-post-reach-teacher-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_POST_REACH_PHASE" in
  collect) timeout --signal=TERM --kill-after=15s 40m .venv/bin/python tools/collect_curriculum_post_reach_teacher_v2.py --config configs/training/curriculum-post-reach-mortal-data-v2.json ;;
  audit) timeout --signal=TERM --kill-after=15s 40m .venv/bin/python tools/audit_curriculum_post_reach_teacher_v2.py --source runs/curriculum-post-reach-teacher-data-v2/data --out runs/curriculum-post-reach-teacher-data-v2/audit-v1 ;;
  *) exit 2 ;;
esac
