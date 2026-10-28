#!/bin/bash
#SBATCH --job-name=mahjong-ddp-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddp-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
.venv/bin/python -c 'import torch; assert torch.cuda.device_count() == 2; print("allocated_gpus", torch.cuda.device_count())'
.venv/bin/torchrun --standalone --nproc_per_node=2 --module jevany.train --config semantic-ddp.toml
# Each evaluator receives one of the GPUs assigned by Slurm, never an arbitrary device.
IFS=',' read -r -a allocated_devices <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU mask missing}"
if (( ${#allocated_devices[@]} != 2 )); then
    echo "Expected exactly two Slurm-assigned GPU identifiers" >&2
    exit 1
fi
(
    export CUDA_VISIBLE_DEVICES="${allocated_devices[0]}"
    .venv/bin/python evaluate.py runs/semantic-2026-ddp data/semantic-2026-v1/validation.jsonl --out runs/semantic-2026-ddp/validation-full.json
    .venv/bin/python arena.py --checkpoint runs/semantic-2026-ddp --seeds 32 --out runs/arena-ddp-model
) &
first_evaluator_pid=$!
(
    export CUDA_VISIBLE_DEVICES="${allocated_devices[1]}"
    .venv/bin/python arena.py --checkpoint runs/semantic-2026-ddp --guarded --seeds 32 --out runs/arena-ddp-guarded
) &
second_evaluator_pid=$!
wait "$first_evaluator_pid"
wait "$second_evaluator_pid"
.venv/bin/python promote.py runs/arena-ddp-model runs/arena-ddp-guarded --out runs/champion-ddp.json
