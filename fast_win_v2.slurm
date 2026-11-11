#!/bin/bash
#SBATCH --job-name=mahjong-fast-win-v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fast-win-v2-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/check_gpu.py
fast_win_mode="${1:?Pass collect or evaluate}"
fast_win_config="${2:?Pass registered JSON config}"
if [[ "$fast_win_mode" == "collect" ]]; then
    .venv/bin/python scripts/collect_fast_win_v2.py --config "$fast_win_config" "${@:3}"
    .venv/bin/python - "$fast_win_config" <<'PY'
import json, subprocess, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
if cfg['purpose_role'] != 'mechanism_smoke':
    subprocess.run([sys.executable, 'scripts/fit_fast_win.py', '--config', sys.argv[1]], check=True)
PY
else
    if [[ "$fast_win_mode" != "evaluate" ]]; then exit 2; fi
    .venv/bin/python scripts/evaluate_fast_win.py --config "$fast_win_config" "${@:3}"
fi
