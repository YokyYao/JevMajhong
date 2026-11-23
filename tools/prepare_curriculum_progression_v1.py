"""Finite GPU-only seal/registration, with actual source and launch binding."""
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_progression_parent_v1 import seal
from tools.register_curriculum_progression_v1 import register, PUBLIC_FILES


def prepare():
    gpu_only(); config = ROOT/'configs/training/curriculum-progression-v1.json'; c = read(config)
    run = ROOT/'runs/curriculum-progression-v1'
    if run.exists(): raise ValueError('Never overwrite an earlier prepare/fit directory')
    for name in PUBLIC_FILES:
        if name.endswith('.py'): compile((ROOT/name).read_text(), str(ROOT/name), 'exec')
    seal(config, run/'parent'); register(config, run/'parent/parent-seal-private.json')
    registered = run/'fit-plan/registered.json'; reg = read(registered)
    files = dict(reg['inputs']); files[str(registered)] = sha(registered)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Launch input changed during preparation')
    write(run/'launch-binding-private.json', dict(files=files, registered_sha256=sha(registered),
        candidate_step=2400, actual_registration_before_first_update=True,
        no_new_model_inference_or_optimizer_update=True, job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname()))
    write(run/'prepare-execution-private.json', dict(status='actual_progression_parent_sealed_and2400_recipe_registered',
        registered_sha256=sha(registered), bound_inputs=len(files), job_id=os.environ['SLURM_JOB_ID'],
        optimizer_updates=0, fit_started=False, automatic_release=False))


if __name__ == '__main__': prepare()
