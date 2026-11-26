#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-yaku-course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-yaku-course-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_YAKU_PHASE:?Set preflight, prepare, fit or audit}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_YAKU_PHASE" "$task_rc" > "logs/curriculum-yaku-course-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_YAKU_PHASE" in
  preflight)
    timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_yaku_runtime_v1.py -v
    timeout --signal=TERM --kill-after=30s 8m .venv/bin/python tools/preflight_curriculum_yaku_launch_v1.py ;;
  prepare)
    timeout --signal=TERM --kill-after=30s 25m .venv/bin/python tools/prepare_curriculum_yaku_course_v1.py ;;
  fit)
    timeout --signal=TERM --kill-after=30s 1190m .venv/bin/python tools/train_curriculum_yaku_course_v1.py --registered runs/curriculum-yaku-course-v1/fit-plan/registered.json ;;
  audit)
    test -n "$JEV_YAKU_FIT_JOB"
    timeout --signal=TERM --kill-after=30s 85m .venv/bin/python tools/finish_curriculum_yaku_audit_v1.py --fit-job "$JEV_YAKU_FIT_JOB" ;;
  *) exit 2 ;;
esac
