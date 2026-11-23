#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-ordinary-completion-audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011
#SBATCH --output=logs/curriculum-ordinary-completion-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-ordinary-completion-audit-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 58m .venv/bin/python tools/audit_curriculum_ordinary_completion_v1.py --source runs/curriculum-ordinary-completion-v1/collection-v1 --out runs/curriculum-ordinary-completion-v1/audit-v1 --terminal runs/curriculum-ordinary-completion-v1/collection-terminal-scheduler-private.txt --exit logs/curriculum-ordinary-completion-exit-317875.json
