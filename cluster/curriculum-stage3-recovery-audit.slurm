#!/bin/bash
# Submit afterok the registered attack fit, with invalid dependencies cancelled.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-recovery-final-audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --exclude=ws-l1-010
#SBATCH --output=logs/curriculum-attack-recovery-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_ATTACK_REGISTERED:?Set the calls-final attack registration}"
test -n "${JEV_ATTACK_AUDIT_BINDING:?Bind the actual audit source and launchers before training}"
test -n "${JEV_ATTACK_EVIDENCE:?Set a fresh final-evidence path}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-attack-recovery-audit-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 3m "${JEV_PYTHON:-.venv/bin/python}" tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 40m "${JEV_PYTHON:-.venv/bin/python}" tools/audit_curriculum_attack_recovery.py --registered "$JEV_ATTACK_REGISTERED" --binding "$JEV_ATTACK_AUDIT_BINDING" --out "$JEV_ATTACK_EVIDENCE"
