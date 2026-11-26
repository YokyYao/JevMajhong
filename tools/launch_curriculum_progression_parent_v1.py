"""Bind held finite jobs, release the fit, and preserve real terminal evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_progression_parent_control_v1 import CONTROL, RUN, ENTRY, PREFLIGHT, PUBLIC
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

QUEUE = CONTROL/'parent-retention-queue-v1-private.json'
CAPTURE = ROOT/'cluster/curriculum-parent-retention-terminal-v1.slurm'


def launch():
    allocation = resource()
    if QUEUE.exists() or RUN.exists():
        raise ValueError('Never submit a duplicate or overwrite a registered run')
    pf = read(PREFLIGHT/'summary-private.json'); binding = read(PREFLIGHT/'completion-binding-private.json')
    if pf['status'] != 'full_validation_parent_retention_launch_preflight_passed':
        raise ValueError('Full actual launch preflight required')
    for p, h in binding['files'].items():
        if sha(p) != h: raise ValueError('Preflight-bound source changed: '+p)
    jobs = []
    def submit(*args):
        r = subprocess.run(['sbatch', '--parsable', *args], cwd=ROOT,
            capture_output=True, text=True, timeout=30)
        if r.returncode: raise RuntimeError(r.stderr)
        value = r.stdout.strip().split(';')[0]
        if not value.isdigit(): raise ValueError('Actual numeric scheduler ID required')
        jobs.append(value)
        return value
    try:
        fit = submit('--hold', '--export=ALL,JEV_PARENT_PHASE=run', str(ENTRY))
        audit = submit('--dependency=afterok:'+fit, '--kill-on-invalid-dep=yes',
            '--time=02:00:00', '--job-name=jev-parent-retention-audit',
            '--export=ALL,JEV_PARENT_PHASE=audit,JEV_PARENT_FIT_JOB='+fit, str(ENTRY))
        capture = submit('--dependency=afterany:'+fit+':'+audit, '--kill-on-invalid-dep=yes', str(CAPTURE))
        observed = {name: scheduler(job) for name, job in [('fit', fit), ('audit', audit), ('capture', capture)]}
        if any(r.returncode for r in observed.values()): raise ValueError('Cannot observe all new scheduler jobs')
        fv, av, cv = (fields(observed[k].stdout) for k in ('fit', 'audit', 'capture'))
        if (fv.get('JobState') != 'PENDING' or fv.get('Reason') != 'JobHeldUser'
                or fv.get('Command') != str(ENTRY) or av.get('Command') != str(ENTRY)
                or cv.get('Command') != str(CAPTURE)
                or f'afterok:{fit}(' not in av.get('Dependency', '')
                or any(f'afterany:{j}(' not in cv.get('Dependency', '') for j in (fit, audit))):
            raise ValueError('Actual held fit / audit / capture dependency contract differs')
        files = [ROOT/p for p in PUBLIC]+[PREFLIGHT/'summary-private.json', PREFLIGHT/'completion-binding-private.json']
        write(QUEUE, dict(status='parent_retention_jobs_bound_while_fit_held',
            fit_job=fit, audit_job=audit, capture_job=capture,
            observations={k: v.stdout for k, v in observed.items()}, allocation=allocation,
            files={str(p): sha(p) for p in files}, student_updates_at_binding=0,
            automatic_release=False, automatic_stage4=False))
        subprocess.run(['scontrol', 'release', fit], check=True, capture_output=True, text=True, timeout=20)
        print(json.dumps(dict(status='parent_retention_fit_released', fit_job=fit, audit_job=audit,
                              capture_job=capture)), flush=True)
    except BaseException as error:
        cleanup = []
        for job in jobs:
            result = subprocess.run(['scancel', job], capture_output=True, text=True, timeout=20)
            cleanup.append(dict(job_id=job, returncode=result.returncode, stderr=result.stderr))
        path = CONTROL/'parent-retention-launch-failure-v1-private.json'
        if not path.exists(): write(path, dict(error=repr(error), cleanup=cleanup, allocation=allocation))
        raise


def capture():
    allocation = resource(); queue = read(QUEUE)
    for p, h in queue['files'].items():
        if sha(p) != h: raise ValueError('Before-release terminal-capture source binding changed: '+p)
    out = (RUN if RUN.exists() else CONTROL)/'parent-retention-terminal-v1'
    out.mkdir(exist_ok=False)
    results = {}
    for role in ('fit', 'audit'):
        job = queue[role+'_job']; result = scheduler(job)
        write(out/(role+'-observation-private.json'), dict(stdout=result.stdout,
            stderr=result.stderr, returncode=result.returncode))
        receipt_path = ROOT/'logs'/f'curriculum-parent-retention-exit-{job}.json'
        receipt = read(receipt_path) if receipt_path.exists() else None
        raw = result.stdout; source = 'actual_terminal_scontrol'
        archive = RUN/'fit-terminal-scheduler-private.json'
        if result.returncode and role == 'fit' and archive.exists():
            prior = read(archive)
            if prior['exit'] != receipt: raise ValueError('Archived/actual fit receipts differ')
            raw = prior['raw']; source = 'actual_independent_audit_fit_archive'
        actual = terminal_record(raw, job, ENTRY)
        if actual['successful']:
            if receipt is None or receipt.get('phase') != role:
                raise ValueError('Successful scheduler job requires exact phase exit receipt')
            scheduler_exit(receipt, raw, ENTRY)
        actual.update(exit_receipt=receipt, source=source)
        results[role] = actual
        write(out/(role+'-terminal-private.json'), actual)
    grade_path = RUN/'final-evidence-private.json'
    write(out/'terminal-evidence-private.json', dict(
        status='parent_retention_actual_terminals_preserved_not_child_admitted',
        scheduler_proofs=results, original_strategy_grade=(read(grade_path).get('stage3_advancement')
            if grade_path.exists() else None), strategy_grade_overridden=False,
        child_parent_sealed=False, student_updates=0, allocation=allocation))
    paths = [QUEUE, *out.glob('*-private.json')]
    write(out/'completion-binding-private.json', dict(files={str(p): sha(p) for p in paths}))
    print(json.dumps(dict(status='actual_parent_retention_terminals_preserved')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('launch', 'capture'), required=True)
    a = p.parse_args()
    if a.phase == 'launch': launch()
    else: capture()
