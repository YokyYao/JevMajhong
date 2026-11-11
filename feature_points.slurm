#!/bin/bash
#SBATCH --job-name=mahjong-stage2-points
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/feature-points-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
feature_points_config="${1:?Pass registered JSON config}"
.venv/bin/python - "$feature_points_config" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
if cfg.get('smoke_config'):
    subprocess.run([sys.executable, 'scripts/audit_feature_point_smoke.py',
                    '--config', cfg['smoke_config']], check=True)
PY
.venv/bin/python scripts/watch_guide_call_points.py --config "$feature_points_config" --job "$SLURM_JOB_ID" --max-seconds 21500 &
feature_points_observer_pid=$!
stop_feature_points_observer() {
    local feature_points_exit=$?
    trap - EXIT
    kill "$feature_points_observer_pid" 2>/dev/null || true
    wait "$feature_points_observer_pid" 2>/dev/null || true
    exit "$feature_points_exit"
}
trap stop_feature_points_observer EXIT
.venv/bin/python scripts/collect_feature_points.py --config "$feature_points_config" "${@:2}"
.venv/bin/python scripts/fit_feature_points.py --config "$feature_points_config"
