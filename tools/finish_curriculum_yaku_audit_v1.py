"""Capture actual fit terminal and run the original independent final audit."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit


def finish(fit_job):
    gpu_only(); run = ROOT/'runs/curriculum-yaku-course-v1'
    exit_path = ROOT/'logs'/f'curriculum-yaku-course-exit-{fit_job}.json'; receipt = read(exit_path)
    if receipt['phase'] != 'fit': raise ValueError('Actual successful FIT receipt required')
    archived = run/'fit-terminal-scheduler-private.json'
    if archived.exists(): raw = read(archived)['raw']
    else:
        actual = subprocess.run(['scontrol', '-o', 'show', 'job', str(fit_job)], check=True, text=True, capture_output=True, timeout=20)
        raw = actual.stdout
        scheduler_exit(receipt, raw, ROOT/'cluster/curriculum-yaku-course-v1.slurm')
        write(archived, dict(raw=raw, exit=receipt))
    scheduler_exit(receipt, raw, ROOT/'cluster/curriculum-yaku-course-v1.slurm')
    binding = read(run/'launch-binding-private.json')
    if any(sha(p) != h for p, h in binding['files'].items()): raise ValueError('Bound launch inputs changed')
    from tools.audit_curriculum_yaku_course_v1 import audit
    audit(run/'fit-plan/registered.json', run/'final-evidence-private.json')
    if any(sha(p) != h for p, h in binding['files'].items()): raise ValueError('Bound source changed during audit')
    write(run/'audit-execution-private.json', dict(status='actual_yaku2500_full_evidence_independently_audited',
        actual_fit_job=fit_job, original_launch_binding_unchanged=True, automatic_release=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--fit-job', type=int, required=True)
    finish(p.parse_args().fit_job)
