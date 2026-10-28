#!/bin/bash
#SBATCH --job-name=mahjong-jevany
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
nvidia-smi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
.venv/bin/jevany train --config mahjong-sft.toml
.venv/bin/python recommend.py runs/jevany-mahjong-pilot data/jevany-pilot-v1/validation.jsonl > "logs/recommend-${SLURM_JOB_ID}.json"
