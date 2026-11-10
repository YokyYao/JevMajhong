#!/bin/bash
#SBATCH --job-name=mahjong-feature-continue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clean-policy-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
continuation_config="${1:-feature-unseen-continuation-20261110.json}"
.venv/bin/python scripts/watch_clean_policy.py --config "$continuation_config" --job "$SLURM_JOB_ID" &
continuation_observer_pid=$!
stop_continuation_observer() {
    local continuation_exit=$?
    trap - EXIT
    kill "$continuation_observer_pid" 2>/dev/null || true
    wait "$continuation_observer_pid" 2>/dev/null || true
    exit "$continuation_exit"
}
trap stop_continuation_observer EXIT
.venv/bin/python scripts/train_feature_continuation.py --config "$continuation_config"
