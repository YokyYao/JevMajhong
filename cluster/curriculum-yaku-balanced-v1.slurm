#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-yaku-balanced-parent
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-yaku-balanced-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_YAKU_PARENT_PHASE:?Set prepare, fit, audit or capture}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_YAKU_PARENT_PHASE" "$task_rc" > "logs/curriculum-yaku-balanced-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_YAKU_PARENT_PHASE" in
  prepare) timeout --signal=TERM --kill-after=30s 25m .venv/bin/python tools/launch_curriculum_yaku_balanced_v1.py --phase prepare ;;
  fit) timeout --signal=TERM --kill-after=30s 1190m .venv/bin/python tools/launch_curriculum_yaku_balanced_v1.py --phase fit ;;
  audit)
    test -n "$JEV_YAKU_PARENT_FIT_JOB"
    timeout --signal=TERM --kill-after=30s 110m .venv/bin/python tools/launch_curriculum_yaku_balanced_v1.py --phase audit --fit-job "$JEV_YAKU_PARENT_FIT_JOB" ;;
  capture) timeout --signal=TERM --kill-after=15s 3m .venv/bin/python tools/launch_curriculum_yaku_balanced_v1.py --phase capture ;;
  bridge) timeout --signal=TERM --kill-after=15s 4m .venv/bin/python tools/launch_curriculum_yaku_balanced_v1.py --phase bridge ;;
  *) exit 2 ;;
esac
