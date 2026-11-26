#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-riichi-control-audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013
#SBATCH --output=logs/curriculum-riichi-control-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_RIICHI_REGISTERED:?Set the completed registered control}"
test -n "${JEV_RIICHI_LAUNCH_RECEIPT:?Set its launch receipt}"
test -n "${JEV_RIICHI_EVIDENCE:?Set a fresh evidence path}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-riichi-control-audit-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 2m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 55m .venv/bin/python tools/audit_curriculum_riichi_control.py \
  --registered "$JEV_RIICHI_REGISTERED" --launch-receipt "$JEV_RIICHI_LAUNCH_RECEIPT" \
  --out "$JEV_RIICHI_EVIDENCE"
