#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-defensive-scope-audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --exclude=ws-l1-010,ws-l2-013,ws-l1-018
#SBATCH --output=logs/curriculum-defensive-scope-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-defensive-scope-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
timeout --signal=TERM --kill-after=15s 3m .venv/bin/python -m unittest tests.core.test_curriculum_riichi_risk_scope_policy_v1 -v
timeout --signal=TERM --kill-after=20s 39m .venv/bin/python tools/audit_curriculum_riichi_risk_scope_v1.py --out runs/curriculum-defensive-scope-v1/audit-v1
