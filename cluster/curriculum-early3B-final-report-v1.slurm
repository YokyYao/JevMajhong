#!/bin/bash
#SBATCH --partition=gpu,ws-ia
#SBATCH --job-name=jev-3b-final-curves
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/curriculum-early3B-final-report-v1-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-early3B-final-report-v1-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
hostname
timeout --signal=TERM --kill-after=15s 1m .venv/bin/python tools/check_gpu.py
sha256sum --check runs/curriculum-early3B-final-report-queue-v1/source-sha256-private.txt
timeout --signal=TERM --kill-after=15s 15m .venv/bin/python tools/summarize_curriculum_curve16_v2.py \
  --registered runs/curriculum-yaku-early-v2/fit-plan/registered.json \
  --steps 0 625 1250 1875 2500 --out runs/curriculum-early3B2500-summary-v2
timeout --signal=TERM --kill-after=15s 3m .venv/bin/python tools/plot_curriculum_development_report_v1.py \
  --report runs/curriculum-early3B2500-summary-v2 \
  --journal runs/curriculum-yaku-early-v2/stage3B-fit/metrics/train_metrics.jsonl \
  --last-step 2500 --out runs/curriculum-early3B2500-curves-v1
