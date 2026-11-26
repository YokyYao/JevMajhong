#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-kan-sources
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/curriculum-kan-sources-%j.log
set -euo pipefail
task_source_dir=${SLURM_SUBMIT_DIR:-$PWD}
cd "$task_source_dir"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"phase\":\"unlabelled_source\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-kan-sources-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$task_source_dir"
hostname
timeout --signal=TERM --kill-after=15s 8m .venv/bin/python tools/generate_curriculum_kan_sources_v1.py --config configs/diagnostics/curriculum-kan-sources-v1.json
