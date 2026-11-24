"""Preserve the conditional 3B jobs' real terminals without admitting Course4."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import fields, resource, scheduler, terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

PARENT = ROOT/'runs/curriculum-progression-riichi-paired-v1'
RUN = ROOT/'runs/curriculum-yaku-paired-v1'
ENTRY = ROOT/'cluster/curriculum-yaku-paired-v1.slurm'
QUEUE = PARENT/'yaku-continuation-queue-private.json'
BINDING = PARENT/'yaku-terminal-capture-binding-v1-private.json'
STATUS = 'conditional_yaku_terminal_capture_bound_before_fit_start'


def fixed_paths():
    return [Path(__file__), ROOT/'cluster/curriculum-yaku-terminal-capture-v1.slurm',
        ROOT/'tests/core/test_curriculum_yaku_paired_terminal_v1.py',
        ROOT/'docs/experiments/plans/curriculum-yaku-terminal-capture-20261124.md',
        ROOT/'tools/capture_curriculum_paired_terminal_v2.py',
        ROOT/'tools/seal_curriculum_basic_repair_v2.py',
        ROOT/'tools/curriculum_paired_yaku_launch_v1.py', ENTRY, QUEUE,
        ROOT/'configs/training/curriculum-yaku-paired-v1.json',
        PARENT/'yaku-continuation-launch-binding-private.json']


def bind():
    allocation = resource(); queue = read(QUEUE)
    observations = {}
    for role, predecessor in (('prepare', queue['actual_parent_terminal_capture_job']),
                              ('fit', queue['prepare_job']), ('audit', queue['fit_job'])):
        job = str(queue[role+'_job']); result = scheduler(job); v = fields(result.stdout)
        if (result.returncode or v.get('JobId') != job or v.get('JobState') != 'PENDING'
                or v.get('Command') != str(ENTRY)
                or f'afterok:{predecessor}(' not in v.get('Dependency', '')):
            raise ValueError('Bind only the actual pending conditional preparation/fit/audit chain')
        observations[role] = result.stdout
    if RUN.exists():
        raise ValueError('Bind before conditional 3B preparation creates its run')
    write(BINDING, dict(status=STATUS, fit_job=str(queue['fit_job']),
        audit_job=str(queue['audit_job']), observed_scheduler=observations,
        files={str(p.resolve()): sha(p) for p in fixed_paths()}, binding_allocation=allocation,
        automatic_fit=False, strategy_grade_overridden=False))
    print(json.dumps(dict(status=STATUS, student_updates=0, child_admitted=False)), flush=True)


def capture():
    allocation = resource(); b = read(BINDING)
    if (b['status'] != STATUS or b['automatic_fit'] or b['strategy_grade_overridden']
            or any(sha(p) != digest for p, digest in b['files'].items())):
        raise ValueError('Original before-start terminal-capture binding changed')
    out = PARENT/'yaku-terminal-capture-v1'; out.mkdir(exist_ok=False)
    proofs = {}; successful = True
    for role in ('fit', 'audit'):
        job = b[role+'_job']; actual = scheduler(job)
        write(out/(role+'-scheduler-observation-private.json'), dict(
            stdout=actual.stdout, stderr=actual.stderr, returncode=actual.returncode))
        receipt_path = ROOT/'logs'/f'curriculum-yaku-paired-exit-{job}.json'
        receipt = read(receipt_path) if receipt_path.exists() else None
        raw = actual.stdout; source = 'actual_terminal_scontrol'
        archive = RUN/'fit-terminal-scheduler-private.json'
        if actual.returncode and role == 'fit' and archive.exists():
            saved = read(archive)
            if saved['exit'] != receipt:
                raise ValueError('Original independent-audit fit archive differs from receipt')
            raw = saved['raw']; source = 'original_independent_audit_fit_archive'
        proof = terminal_record(raw, job, ENTRY)
        proof.update(source=source, exit_receipt=receipt)
        write(out/(role+'-terminal-private.json'), proof)
        if proof['successful']:
            if receipt is None or receipt.get('phase') != role:
                raise ValueError('Actual successful role receipt required')
            scheduler_exit(receipt, raw, ENTRY)
        successful = successful and proof['successful']; proofs[role] = proof
    evidence = RUN/'final-evidence-private.json'
    original = read(evidence) if evidence.exists() else None
    write(out/'terminal-evidence-private.json', dict(
        status='conditional_yaku_actual_terminals_preserved_not_placement_admitted',
        scheduler_proofs=proofs, both_scheduler_jobs_successful=successful,
        original_final_evidence=original, strategy_grade_overridden=False,
        child_parent_sealed=False, student_updates=0, automatic_fit=False,
        automatic_release=False, capture_allocation=allocation))
    paths = [BINDING, *out.glob('*-private.json')]
    write(out/'completion-binding-private.json', dict(files={str(p.resolve()): sha(p) for p in paths}))
    print(json.dumps(dict(status='actual_yaku_terminals_preserved',
        both_jobs_successful=successful, child_admitted=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['bind', 'capture'], required=True)
    if parser.parse_args().phase == 'bind': bind()
    else: capture()
