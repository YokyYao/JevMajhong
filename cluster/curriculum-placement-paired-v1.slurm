#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-placement-paired
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --exclude=ws-l1-010,ws-l1-011,ws-l1-018,ws-l2-010,ws-l2-011,ws-l2-013,ws-l3-013,ws-l5-004
#SBATCH --output=logs/curriculum-placement-paired-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_PLACEMENT_PAIRED_PHASE:?Set prepare, integration, fit, audit or capture}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$JEV_PLACEMENT_PAIRED_PHASE" "$task_rc" > "logs/curriculum-placement-paired-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "$JEV_PLACEMENT_PAIRED_PHASE" in
  prepare) limit=55m ;;
  integration) limit=38m ;;
  fit) limit=1190m ;;
  audit) limit=100m ;;
  capture) limit=3m ;;
  *) exit 2 ;;
esac
timeout --signal=TERM --kill-after=30s "$limit" .venv/bin/python tools/curriculum_paired_placement_launch_v1.py --phase "$JEV_PLACEMENT_PAIRED_PHASE"
