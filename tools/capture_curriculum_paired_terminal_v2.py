"""Preserve the paired retry's real scheduler terminal; never admit a child."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN = ROOT/'runs/curriculum-progression-riichi-paired-v1'
ENTRY = ROOT/'cluster/curriculum-progression-paired-v1.slurm'
BINDING = RUN/'terminal-capture-binding-v2-private.json'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def fields(raw):
    return dict(t.split('=', 1) for t in raw.split() if '=' in t)


def scheduler(job):
    return subprocess.run(['scontrol', '-o', 'show', 'job', str(job)],
        capture_output=True, text=True, timeout=20)


def terminal_record(raw, job, command=ENTRY):
    v = fields(raw)
    if (v.get('JobId') != str(job) or v.get('Command') != str(command.resolve())
            or v.get('JobState') not in TERMINAL
            or v.get('EndTime') in (None, 'Unknown', 'N/A', '(null)')):
        raise ValueError('An exact original terminal scheduler record is required')
    return dict(job_id=str(job), state=v['JobState'], exit_code=v.get('ExitCode'),
        raw=raw, successful=v['JobState'] == 'COMPLETED' and v.get('ExitCode') == '0:0')


def resource():
    job = os.environ.get('SLURM_JOB_ID', '')
    actual = scheduler(job)
    v = fields(actual.stdout)
    tres = dict(t.split('=', 1) for t in v.get('AllocTRES', '').split(',') if '=' in t)
    if (not job.isdigit() or actual.returncode or v.get('JobId') != job
            or v.get('JobState') != 'RUNNING' or int(tres.get('gres/gpu', '0')) < 1
            or socket.gethostname().startswith(('lo-', 'login'))
            or v.get('NodeList') != socket.gethostname().split('.')[0]):
        raise ValueError('Actual running allocated GPU node required')
    return actual.stdout


def bound_paths():
    return [Path(__file__), ROOT/'cluster/curriculum-paired-terminal-capture-v2.slurm',
        ROOT/'tests/core/test_curriculum_paired_terminal_capture_v2.py',
        ROOT/'tools/capture_curriculum_paired_terminal_v1.py',
        ROOT/'docs/experiments/plans/curriculum-paired-terminal-capture-v2-20261124.md',
        RUN/'fit-plan/registered.json', RUN/'launch-binding-private.json', ENTRY,
        ROOT/'tools/seal_curriculum_basic_repair_v2.py']


def bind(fit_job, audit_job):
    allocation = resource()
    if BINDING.exists():
        raise ValueError('Never overwrite an existing terminal-capture binding')
    fit = scheduler(fit_job); audit = scheduler(audit_job)
    fv = fields(fit.stdout); av = fields(audit.stdout)
    metric = RUN/'stage3-fit/metrics/train_metrics.jsonl'
    last = None
    with metric.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if (fit.returncode or audit.returncode or fv.get('JobId') != str(fit_job)
            or fv.get('JobState') != 'RUNNING' or fv.get('Command') != str(ENTRY)
            or av.get('JobId') != str(audit_job) or av.get('Command') != str(ENTRY)
            or av.get('JobState') != 'PENDING'
            or f'afterok:{fit_job}(' not in av.get('Dependency', '')
            or last is None or not 0 < last['local_step'] < 2400):
        raise ValueError('Bind during the actual fit, before its dependent audit starts')
    result = dict(status='paired_retry_terminal_capture_bound_before_fit_completion',
        fit_job=str(fit_job), audit_job=str(audit_job), observed_step=last['local_step'],
        observed_fit_scheduler=fit.stdout, observed_audit_scheduler=audit.stdout,
        capture_binding_allocation=allocation,
        files={str(p.resolve()): sha(p) for p in bound_paths()},
        automatic_fit=False, strategy_grade_overridden=False)
    write(BINDING, result)
    print(json.dumps(dict(status=result['status'], observed_step=result['observed_step'])), flush=True)


def capture():
    allocation = resource(); b = read(BINDING)
    if (b['status'] != 'paired_retry_terminal_capture_bound_before_fit_completion'
            or b['automatic_fit'] or b['strategy_grade_overridden']
            or any(sha(p) != h for p, h in b['files'].items())):
        raise ValueError('Unchanged before-completion capture source binding required')
    out = RUN/'terminal-capture-v2'; out.mkdir(exist_ok=False)
    observations = {}; proofs = {}; successful = True
    for role in ('fit', 'audit'):
        job = b[role+'_job']; result = scheduler(job)
        observations[role] = dict(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        # Preserve the real observation before validating it. Missing records
        # remain missing; they are never replaced by a synthesized scheduler row.
        write(out/(role+'-scheduler-observation-private.json'), observations[role])
        receipt_path = ROOT/'logs'/f'curriculum-progression-paired-exit-{job}.json'
        receipt = read(receipt_path) if receipt_path.exists() else None
        archive = RUN/'fit-terminal-scheduler-private.json'
        raw = result.stdout
        source = 'actual_terminal_scontrol'
        if result.returncode and role == 'fit' and archive.exists():
            captured = read(archive)
            if captured['exit'] != receipt:
                raise ValueError('Original independent-audit fit archive differs from exit receipt')
            raw = captured['raw']; source = 'original_independent_audit_fit_archive'
        proof = terminal_record(raw, job)
        proof.update(source=source, exit_receipt=receipt)
        write(out/(role+'-terminal-private.json'), proof)
        if proof['successful']:
            if receipt is None or receipt.get('phase') != ('fit' if role == 'fit' else 'audit'):
                raise ValueError('Actual successful role exit receipt required')
            scheduler_exit(receipt, raw, ENTRY)
        successful = successful and proof['successful']
        proofs[role] = proof
    evidence = RUN/'final-evidence-private.json'
    grade = read(evidence).get('stage3_advancement') if evidence.exists() else None
    write(out/'terminal-evidence-private.json', dict(
        status='paired_retry_actual_terminals_preserved_not_child_admitted',
        scheduler_proofs=proofs, both_scheduler_jobs_successful=successful,
        original_strategy_grade=grade, strategy_grade_overridden=False,
        child_parent_sealed=False, student_updates=0, automatic_fit=False,
        automatic_release=False, capture_allocation=allocation))
    paths = [BINDING, *out.glob('*-private.json')]
    write(out/'completion-binding-private.json', dict(files={str(p.resolve()): sha(p) for p in paths}))
    print(json.dumps(dict(status='actual_terminals_preserved', both_jobs_successful=successful,
        original_strategy_grade=grade, child_admitted=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['bind', 'capture'], required=True)
    p.add_argument('--fit-job', type=int); p.add_argument('--audit-job', type=int)
    a = p.parse_args()
    if a.phase == 'bind':
        if not a.fit_job or not a.audit_job:
            p.error('Both actual job IDs are required for binding')
        bind(a.fit_job, a.audit_job)
    else:
        capture()
