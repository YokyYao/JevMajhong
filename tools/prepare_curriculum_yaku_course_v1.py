"""Admit a fixed 3B run only after the actual qualified final 3A audit."""
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_yaku_parent_v1 import seal, extend
from tools.register_curriculum_yaku_course_v1 import register


def prepare():
    gpu_only(); run = ROOT/'runs/curriculum-yaku-course-v1'
    preflight = run/'launch-preflight-v1/summary-private.json'; result = read(preflight)
    if result['status'] != 'yaku_launch_full_synthetic_validation_integration_passed_not_parent_or_fit_admitted':
        raise ValueError('Actual launch integration required')
    files = {}; extend(files, preflight.parent/'completion-binding-private.json')
    config = ROOT/'configs/training/curriculum-yaku-fit-v1.json'
    if (run/'parent').exists() or (run/'fit-plan').exists() or (run/'stage3B-fit').exists():
        raise ValueError('No overwrite of parent/registration/fit')
    seal(config, run/'parent'); register(config, run/'parent/parent-seal-private.json')
    registered = run/'fit-plan/registered.json'; reg = read(registered)
    files.update(reg['inputs']); files[str(registered)] = sha(registered); files[str(preflight)] = sha(preflight)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Launch input changed')
    write(run/'launch-binding-private.json', dict(files=files, registered_sha256=sha(registered),
        candidate_step=2500, actual_registration_before_first_update=True, optimizer_updates=0,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname()))
    write(run/'prepare-execution-private.json', dict(status='actual_yaku_parent_sealed_and2500_recipe_registered',
        registered_sha256=sha(registered), bound_inputs=len(files), optimizer_updates=0, fit_started=False,
        original_3A_grade_preserved=True, automatic_release=False))


if __name__ == '__main__': prepare()
