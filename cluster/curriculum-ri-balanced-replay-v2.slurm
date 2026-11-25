#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-ri-canonical-replay
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=15:00:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-ri-balanced-replay-v2-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "$JEV_RI_REPLAY_V2_PHASE"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_RI_REPLAY_V2_PHASE" "$task_rc" > "logs/curriculum-ri-balanced-replay-v2-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_RI_REPLAY_V2_PHASE" in
  prepare) phase_limit=14m ;;
  fit) phase_limit=870m ;;
  audit) phase_limit=110m ;;
  capture) phase_limit=4m ;;
  *) exit 2 ;;
esac
timeout --signal=TERM --kill-after=30s "$phase_limit" .venv/bin/python tools/curriculum_ri_balanced_replay_v2.py --phase "$JEV_RI_REPLAY_V2_PHASE"
