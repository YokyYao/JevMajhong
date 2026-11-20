#!/bin/bash
# Submit with afterok on the calls fit and --kill-on-invalid-dep=yes.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-curriculum-calls-audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --output=logs/curriculum-calls-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_COURSE_REGISTERED:?Set the course manifest}"
test -n "${JEV_COURSE_EVIDENCE_OUT:?Set a fresh evidence output path}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-calls-audit-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 3m python tools/check_gpu.py
timeout --signal=TERM --kill-after=30s 30m python tools/audit_curriculum_course.py --registered "$JEV_COURSE_REGISTERED" --out "$JEV_COURSE_EVIDENCE_OUT"
