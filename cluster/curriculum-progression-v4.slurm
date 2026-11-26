#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-progression-v4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:30:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-progression-v4-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "$JEV_PROGRESS_PHASE"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_PROGRESS_PHASE" "$task_rc" > "logs/curriculum-progression-v4-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_PROGRESS_PHASE" in
  run)
    JEV_PROGRESS_PHASE=prepare
    timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_checkpoint_evaluation_fence_v1.py -v
    timeout --signal=TERM --kill-after=30s 14m .venv/bin/python tools/prepare_curriculum_progression_replay_v4.py
    JEV_PROGRESS_PHASE=fit
    timeout --signal=TERM --kill-after=30s 830m .venv/bin/python tools/train_curriculum_progression_v3.py --registered runs/curriculum-progression-v4/fit-plan/registered.json ;;
  audit)
    test -n "$JEV_PROGRESS_FIT_JOB"
    timeout --signal=TERM --kill-after=30s 55m .venv/bin/python tools/finish_curriculum_progression_audit_v4.py --fit-job "$JEV_PROGRESS_FIT_JOB" ;;
  *) exit 2 ;;
esac

