#!/bin/bash
# Register a qualified calls final before submitting this bounded GPU job.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-curriculum-attack
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=ws-l1-010
#SBATCH --output=logs/curriculum-attack-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_ATTACK_REGISTERED:?Set the calls-final attack registration}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-attack-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 3m "${JEV_PYTHON:-.venv/bin/python}" tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 340m "${JEV_PYTHON:-.venv/bin/python}" tools/train_curriculum_attack.py --registered "$JEV_ATTACK_REGISTERED"
