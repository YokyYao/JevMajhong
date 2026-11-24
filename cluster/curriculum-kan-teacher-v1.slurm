#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-kan-teacher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=logs/curriculum-kan-teacher-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"teacher_labels\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-kan-teacher-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SLURM_SUBMIT_DIR"
hostname
timeout --signal=TERM --kill-after=15s 8m .venv/bin/python tools/label_curriculum_kan_sources_v1.py --config configs/diagnostics/curriculum-kan-teacher-v1.json
