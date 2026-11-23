#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-win-controls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-rank-win-controls-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "$JEV_WIN_CONTROL_PHASE"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_WIN_CONTROL_PHASE" "$task_rc" > "logs/curriculum-rank-win-controls-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_WIN_CONTROL_PHASE" in
  prepare)
    timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_rank_win_controls_v1.py -v
    timeout --signal=TERM --kill-after=30s 12m .venv/bin/python tools/prepare_curriculum_rank_win_controls_v1.py ;;
  label)
    test -n "$SLURM_ARRAY_TASK_ID"
    timeout --signal=TERM --kill-after=30s 710m .venv/bin/python tools/collect_curriculum_rank_win_controls_v1.py ;;
  merge)
    timeout --signal=TERM --kill-after=30s 40m .venv/bin/python tools/merge_curriculum_rank_win_controls_v1.py ;;
  *) exit 2 ;;
esac


