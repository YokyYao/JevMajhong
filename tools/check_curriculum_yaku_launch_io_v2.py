"""Check the actual nested conditional-config writer; no repeated model benchmark."""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.train_curriculum_yaku_course_v2 import write_conditional_config


def check():
    gpu_only(); out = ROOT/'runs/curriculum-yaku-course-v1/launch-io-fix-v2'
    out.mkdir(parents=True, exist_ok=False)
    original = read(ROOT/'runs/curriculum-yaku-course-v1/launch-preflight-v1/completion-binding-private.json')
    if any(sha(p) != h for p, h in original['files'].items()): raise ValueError('Original preflight sources changed')
    names = ('tools/train_curriculum_yaku_course_v2.py', 'tools/register_curriculum_yaku_course_v2.py',
        'tools/prepare_curriculum_yaku_course_v2.py', 'tools/finish_curriculum_yaku_audit_v2.py',
        'tools/check_curriculum_yaku_launch_io_v2.py', 'cluster/curriculum-yaku-course-v2.slurm')
    files = {str((ROOT/name).resolve()): sha(ROOT/name) for name in names}
    for name in names:
        if name.endswith('.py'): compile((ROOT/name).read_text(), str(ROOT/name), 'exec')
    with tempfile.TemporaryDirectory(prefix='jev-yaku-config-', dir=out) as temporary:
        path = Path(temporary)/'conditional-yaku-configs-private/step-000000.json'
        value = dict(expected_cases=92, source_partition='validation', checkpoint_identity='io_fixture_only')
        write_conditional_config(path, value)
        if read(path) != value: raise ValueError('Actual nested config did not persist')
        try: write_conditional_config(path, dict(value, expected_cases=0))
        except FileExistsError: pass
        else: raise ValueError('Existing configuration was overwritten')
        if read(path) != value: raise ValueError('Existing config changed after duplicate write')
    if any(sha(p) != h for p, h in files.items()): raise ValueError('I/O implementation changed')
    result = dict(status='actual_yaku_conditional_config_directory_and_exclusive_write_checked',
        original_full24516_interface_binding_unchanged=True, full_validation_or_game_benchmark_repeated=False,
        new_actual_nested_config_write_passed=True, existing_config_overwrite_refused=True,
        student_updates=0, model_loaded=False, implementation_revision=2,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'completion-binding-private.json', dict(files=files))
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': check()
