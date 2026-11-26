#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-placement-input
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-placement-input-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-placement-input-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=15s 2m .venv/bin/python -m unittest discover -s tests/core -p test_curriculum_placement_input_v1.py -v
timeout --signal=TERM --kill-after=30s 26m .venv/bin/python tools/verify_curriculum_placement_input_v2.py

