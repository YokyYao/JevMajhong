#!/bin/bash
#SBATCH --job-name=mahjong-preserved-calls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clean-strength-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
eval_config="${1:-preserved-call-development-20261110.json}"
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/evaluate_preserved_calls.py --config "$eval_config" "${@:2}"
.venv/bin/python - "$eval_config" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
cfg=json.loads(Path(sys.argv[1]).read_text())
out=Path(cfg['out'])
if (out/'evaluation-skipped.json').exists():
    print((out/'evaluation-skipped.json').read_text())
else:
    subprocess.run([sys.executable,'scripts/audit_clean_policy.py','--config',sys.argv[1]],check=True)
PY
