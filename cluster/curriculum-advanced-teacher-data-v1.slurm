#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-advanced-teacher-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/curriculum-advanced-teacher-data-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-advanced-teacher-data-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
case "${JEV_ADVANCED_PHASE:-collect}" in
  collect)
    timeout --signal=TERM --kill-after=15s 58m .venv/bin/python tools/collect_curriculum_advanced_teacher.py \
      --config configs/training/curriculum-advanced-mortal-data-v1.json ;;
  audit)
    timeout --signal=TERM --kill-after=15s 58m .venv/bin/python tools/audit_curriculum_advanced_teacher.py \
      --source runs/curriculum-advanced-teacher-data-v1/data --out runs/curriculum-advanced-teacher-data-v1/audit-v1 ;;
  *) echo 'JEV_ADVANCED_PHASE must be collect or audit' >&2; exit 2 ;;
esac
