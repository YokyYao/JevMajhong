#!/bin/bash
#SBATCH --job-name=mahjong-call-features
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/call-natural-features-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/watch_call_repair.py --config call-natural-features-20261110.json --eval-job "$SLURM_JOB_ID" --max-seconds 9000 &
call_feature_observer_pid=$!
stop_call_feature_observer() {
    local call_feature_exit=$?
    trap - EXIT
    kill "$call_feature_observer_pid" 2>/dev/null || true
    wait "$call_feature_observer_pid" 2>/dev/null || true
    exit "$call_feature_exit"
}
trap stop_call_feature_observer EXIT
.venv/bin/python scripts/cache_call_features.py --config call-natural-features-20261110.json
wait "$call_feature_observer_pid"
