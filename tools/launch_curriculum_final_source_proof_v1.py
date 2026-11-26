"""Submit the final source audit after actual placement jobs exist; open no games."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write

OUT = ROOT/'runs/curriculum-final-source-dispatch-v1'
BINDING = OUT/'source-binding-private.json'
QUEUE = OUT/'queue-private.json'
ENTRY = ROOT/'cluster/curriculum-final-source-dispatch-v1.slurm'
PROOF_ENTRY = ROOT/'cluster/curriculum-final-source-proof-v1.slurm'
PLACEMENT_ENTRY = ROOT/'cluster/curriculum-placement-early-course-v1.slurm'
PLACEMENT_QUEUE = ROOT/'runs/curriculum-placement-early-launch-v1/queue-private.json'
UPSTREAM_LAUNCHER = '325118'
SOURCE_JOBS = ('326782', '326787')


def actual_ids(queue):
    roles = ('register', 'integration', 'prepare', 'fit', 'audit', 'capture')
    ids = {role:str(queue[role+'_job']) for role in roles}
    if any(not job.isdigit() or int(job) <= 0 for job in ids.values()) or len(set(ids.values())) != len(roles):
        raise ValueError('Actual distinct numeric placement job IDs required')
    return ids


def require_held(result, job, entry, dependencies):
    row = fields(result.stdout)
    if (result.returncode or row.get('JobId') != job or row.get('JobState') != 'PENDING'
            or row.get('Reason') != 'JobHeldUser' or row.get('Command') != str(entry)
            or not row.get('UserId', '').endswith(f'({os.getuid()})')
            or any(kind+':'+dep+'(' not in row.get('Dependency', '') for kind,dep in dependencies)):
        raise ValueError('Exact actual held job and dependencies required')


def unchanged():
    binding = read(BINDING)
    if any(sha(path) != expected for path,expected in binding['files'].items()):
        raise ValueError('Final source dispatch sources changed')
    return binding


def bind():
    allocation = resource()
    if OUT.exists(): raise ValueError('Never overwrite dispatch evidence')
    prior = ROOT/'runs/curriculum-final-source-catalog-entry-v1/completion-binding-private.json'
    for path,expected in read(prior)['files'].items():
        if sha(path) != expected: raise ValueError('Previously preflighted producer changed')
    result = subprocess.run([sys.executable, '-m', 'unittest',
        'tests.core.test_curriculum_final_source_dispatch_v1'], cwd=ROOT,
        capture_output=True, text=True, timeout=45)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    for path in (ENTRY, PROOF_ENTRY): subprocess.run(['bash','-n',str(path)], check=True, timeout=10)
    paths = [Path(__file__), ENTRY, PROOF_ENTRY, prior,
        ROOT/'tests/core/test_curriculum_final_source_dispatch_v1.py',
        ROOT/'tools/prove_curriculum_final_source_catalog_v1.py',
        ROOT/'tools/launch_curriculum_placement_early_v1.py', PLACEMENT_ENTRY]
    OUT.mkdir(exist_ok=False)
    write(BINDING, dict(files={str(p):sha(p) for p in paths}, allocation=allocation,
        upstream_launcher=UPSTREAM_LAUNCHER, source_jobs=SOURCE_JOBS,
        test_output=result.stdout+result.stderr, final_games_opened=False))
    print(json.dumps(dict(status='final_source_dispatch_bound_not_submitted', final_games_opened=False)), flush=True)


def dispatch():
    allocation = resource(); unchanged()
    if QUEUE.exists(): raise ValueError('Never duplicate the final source job')
    launch = scheduler(UPSTREAM_LAUNCHER)
    if launch.returncode or not terminal_record(launch.stdout, UPSTREAM_LAUNCHER, PLACEMENT_ENTRY)['successful']:
        raise ValueError('Actual successful placement submitter required')
    receipt = read(ROOT/f'logs/curriculum-placement-early-course-exit-{UPSTREAM_LAUNCHER}.json')
    if receipt != dict(job_id=UPSTREAM_LAUNCHER, phase='submit', exit_code=0):
        raise ValueError('Actual placement submission exit differs')
    ids = actual_ids(read(PLACEMENT_QUEUE))
    actual = {role:scheduler(ids[role]) for role in ('audit','capture')}
    for role,observed in actual.items():
        row = fields(observed.stdout)
        if (observed.returncode or row.get('JobId') != ids[role]
                or row.get('Command') != str(PLACEMENT_ENTRY)
                or not row.get('UserId','').endswith(f'({os.getuid()})')):
            raise ValueError('Actual placement audit/capture jobs required')
    dependencies = [('afterok', job) for job in (ids['audit'], ids['capture'], *SOURCE_JOBS)]
    jobs = {}; released = False
    try:
        def submit(entry, deps, phase=None):
            args = ['sbatch','--hold','--parsable','--no-requeue','--nice=1000',
                '--kill-on-invalid-dep=yes','--dependency='+','.join(k+':'+j for k,j in deps)]
            if phase: args += ['--export=ALL,JEV_FINAL_SOURCE_PHASE='+phase]
            result = subprocess.run([*args,str(entry)], cwd=ROOT, capture_output=True,
                text=True, check=True, timeout=25)
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit(): raise ValueError('Actual numeric submitted job ID required')
            jobs[phase or 'proof'] = job
            require_held(scheduler(job), job, entry, deps)
            return job
        proof = submit(PROOF_ENTRY, dependencies)
        capture = submit(ENTRY, [('afterany',proof)], 'capture')
        write(QUEUE, dict(status='final_source_jobs_bound_while_held', proof_job=proof,
            capture_job=capture, placement_jobs=ids, dependencies=dependencies,
            placement_queue_sha256=sha(PLACEMENT_QUEUE), source_binding_sha256=sha(BINDING),
            actual_placement_launcher=launch.stdout,
            actual_placement_jobs={k:v.stdout for k,v in actual.items()}, allocation=allocation,
            new_games=0, final_confirmation_opened=False))
        for job in (capture,proof):
            subprocess.run(['scontrol','release',job],check=True,capture_output=True,text=True,timeout=20)
        released = True
        print(json.dumps(dict(status='actual_final_source_job_queued_after_placement_audit', **jobs,
            final_confirmation_opened=False)), flush=True)
    finally:
        if not released:
            for job in jobs.values(): subprocess.run(['scancel',job],capture_output=True,timeout=20)


def capture():
    allocation = resource(); unchanged(); queue = read(QUEUE); job = queue['proof_job']
    if str(os.environ['SLURM_JOB_ID']) != queue['capture_job']:
        raise ValueError('Only the actual bound capture job may archive this proof')
    observed = scheduler(job)
    write(OUT/'proof-terminal-observation-private.json', dict(stdout=observed.stdout,
        stderr=observed.stderr, returncode=observed.returncode, allocation=allocation))
    if observed.returncode: raise ValueError('Actual final source terminal missing')
    proof = terminal_record(observed.stdout, job, PROOF_ENTRY)
    path = ROOT/f'logs/curriculum-final-source-proof-v1-exit-{job}.json'
    receipt = read(path) if path.exists() else None
    if proof['successful'] and receipt != dict(job_id=job,exit_code=0):
        raise ValueError('Successful proof exit receipt differs')
    write(OUT/'proof-terminal-private.json', dict(terminal=proof, exit_receipt=receipt,
        actual_scheduler_record=True, final_confirmation_opened=False))
    print(json.dumps(dict(status='actual_final_source_terminal_preserved', successful=proof['successful'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('bind','dispatch','capture'),required=True)
    phase = parser.parse_args().phase
    {'bind':bind,'dispatch':dispatch,'capture':capture}[phase]()
