#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-placement-course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l2-010,ws-l1-018,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-placement-course-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_PLACEMENT_PHASE:?Set run or audit}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_PLACEMENT_PHASE" "$task_rc" > "logs/curriculum-placement-course-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_PLACEMENT_PHASE" in
  run)
    JEV_PLACEMENT_PHASE=prepare
    timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_placement_launch_v1.py -v
    timeout --signal=TERM --kill-after=30s 15m .venv/bin/python tools/prepare_curriculum_placement_launch_v1.py \
      --config configs/training/curriculum-placement-launch-v1.json
    JEV_PLACEMENT_PHASE=fit
    timeout --signal=TERM --kill-after=30s 1170m .venv/bin/python tools/train_curriculum_placement_course_v1.py \
      --registered runs/curriculum-placement-course-v1/fit-plan/registered.json ;;
  audit)
    test -n "$JEV_PLACEMENT_FIT_JOB"
    timeout --signal=TERM --kill-after=30s 100m .venv/bin/python tools/finish_curriculum_placement_audit_v1.py \
      --registered runs/curriculum-placement-course-v1/fit-plan/registered.json --fit-job "$JEV_PLACEMENT_FIT_JOB" ;;
  *) exit 2 ;;
esac
