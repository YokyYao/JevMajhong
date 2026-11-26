#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-complete-legal-native-audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018,ws-l2-010,ws-l2-011,ws-l5-004
#SBATCH --output=logs/curriculum-complete-legal-native-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "$JEV_NATIVE_COMPARE_JOB"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-complete-legal-native-audit-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 13m .venv/bin/python tools/audit_curriculum_complete_legal_class_native_v1.py --source runs/curriculum-complete-legal-class-v1/native64-v2 --out runs/curriculum-complete-legal-class-v1/native64-audit-v1 --terminal runs/curriculum-complete-legal-class-v1/native64-v2-terminal-scheduler-private.txt --exit "logs/curriculum-complete-legal-native64-v2-exit-$JEV_NATIVE_COMPARE_JOB.json"
