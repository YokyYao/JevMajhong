#!/bin/bash
#SBATCH --job-name=mahjong-guide-points
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/guide-call-points-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
guide_points_config="${1:-guide-call-points-20261110.json}"
.venv/bin/python scripts/watch_guide_call_points.py --config "$guide_points_config" --job "$SLURM_JOB_ID" &
guide_points_observer_pid=$!
stop_guide_points_observer() {
    local guide_points_exit=$?
    trap - EXIT
    kill "$guide_points_observer_pid" 2>/dev/null || true
    wait "$guide_points_observer_pid" 2>/dev/null || true
    exit "$guide_points_exit"
}
trap stop_guide_points_observer EXIT
.venv/bin/python scripts/collect_guide_call_points.py --config "$guide_points_config" "${@:2}"
.venv/bin/python scripts/fit_guide_call_points.py --config "$guide_points_config"
