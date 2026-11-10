#!/bin/bash
#SBATCH --job-name=mahjong-call-encoder
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/call-encoder-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
encoder_config="${1:-call-encoder-20261110.json}"
.venv/bin/python scripts/watch_call_encoder.py --config "$encoder_config" --job "$SLURM_JOB_ID" &
encoder_observer_pid=$!
stop_encoder_observer() {
    local encoder_exit=$?
    trap - EXIT
    kill "$encoder_observer_pid" 2>/dev/null || true
    wait "$encoder_observer_pid" 2>/dev/null || true
    exit "$encoder_exit"
}
trap stop_encoder_observer EXIT
.venv/bin/python scripts/train_call_encoder.py --config "$encoder_config"
