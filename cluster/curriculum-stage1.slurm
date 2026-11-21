#!/bin/bash
# Adapt the partition name to the cluster; submit from the repository root.
#SBATCH --partition=ws-ia
#SBATCH --job-name=jev-curriculum-stage1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/curriculum-stage1-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -n "$SLURM_JOB_ID"
test -n "$SLURM_JOB_GPUS"
test -n "${JEV_BASE_MODEL_DIR:?Set JEV_BASE_MODEL_DIR to the pinned Qwen snapshot}"
trap 'task_rc=$?; printf "{\"job_id\":\"%s\",\"exit_code\":%d}\n" "$SLURM_JOB_ID" "$task_rc" > "logs/curriculum-exit-$SLURM_JOB_ID.json"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
hostname
timeout --signal=TERM --kill-after=15s 15m python tools/bootstrap_curriculum.py
if [ ! -d models/jev-play-v2 ]; then
    timeout --signal=TERM --kill-after=15s 10m python tools/restore_weights.py weights/jev-play-v2/manifest.json --out models/jev-play-v2
fi
timeout --signal=TERM --kill-after=15s 10m python tools/prepare_curriculum.py --out runs/curriculum-reproduction-v1/public-data
timeout --signal=TERM --kill-after=15s 10m python tools/build_curriculum_stage1.py --source runs/curriculum-reproduction-v1/public-data --out runs/curriculum-reproduction-v1/clean-data-v1
timeout --signal=TERM --kill-after=15s 10m python tools/register_curriculum.py --config configs/training/curriculum-stage1-v1.json --base-path "$JEV_BASE_MODEL_DIR" --reference-path models/jev-play-v2
timeout --signal=TERM --kill-after=30s 230m python tools/train_curriculum.py --registered runs/curriculum-reproduction-v1/fit-plan/registered.json
timeout --signal=TERM --kill-after=15s 3m python tools/plot_curriculum_fixed_grid.py --metrics-dir runs/curriculum-reproduction-v1/stage1-fit/metrics
