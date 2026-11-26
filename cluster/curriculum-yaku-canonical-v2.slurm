#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-yaku-canonical-parent
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-yaku-canonical-v2-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "$JEV_YAKU_PARENT_PHASE"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_YAKU_PARENT_PHASE" "$task_rc" > "logs/curriculum-yaku-canonical-v2-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_YAKU_PARENT_PHASE" in
  bind-enqueue) phase_limit=14m ;;
  archive-parent|bridge) phase_limit=4m ;;
  prepare) phase_limit=25m ;;
  fit) phase_limit=1190m ;;
  audit) phase_limit=110m ;;
  capture) phase_limit=3m ;;
  *) exit 2 ;;
esac
if [ "$JEV_YAKU_PARENT_PHASE" = audit ]; then
  test -n "$JEV_YAKU_PARENT_FIT_JOB"
  timeout --signal=TERM --kill-after=30s "$phase_limit" .venv/bin/python tools/curriculum_yaku_canonical_v2.py --phase audit --fit-job "$JEV_YAKU_PARENT_FIT_JOB"
else
  timeout --signal=TERM --kill-after=30s "$phase_limit" .venv/bin/python tools/curriculum_yaku_canonical_v2.py --phase "$JEV_YAKU_PARENT_PHASE"
fi
