#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-yaku-v8-cases
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004,ws-l5-002
#SBATCH --output=logs/curriculum-yaku-v8-cases-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-yaku-v8-cases-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=10s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=20s 18m .venv/bin/python tools/evaluate_curriculum_yaku_cases_v1.py --config configs/diagnostics/curriculum-yaku-v8-cases-v1.json
