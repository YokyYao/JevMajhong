"""Freeze the source-only teacher job and preserve its real GPU exit evidence."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN = ROOT/'runs/curriculum-daiminkan-context-v1'
ENTRY = ROOT/'cluster/curriculum-daiminkan-context-teacher-v1.slurm'
CAPTURE = ROOT/'cluster/curriculum-daiminkan-context-terminal-v1.slurm'
BINDING = RUN/'teacher-launch-binding-v1-private.json'
QUEUE = RUN/'teacher-queue-v1-private.json'
PUBLIC = (
    'tools/generate_curriculum_daiminkan_context_v1.py',
    'tools/label_curriculum_daiminkan_context_v1.py',
    'tools/launch_curriculum_daiminkan_teacher_v1.py',
    'configs/diagnostics/curriculum-daiminkan-context-v1.json',
    'tests/core/test_curriculum_daiminkan_context_v1.py',
    'cluster/curriculum-daiminkan-context-teacher-v1.slurm',
    'cluster/curriculum-daiminkan-context-terminal-v1.slurm',
    'docs/experiments/plans/curriculum-daiminkan-context-20261124.md',
    'docs/experiments/plans/curriculum-daiminkan-teacher-20261124.md',
)


def launch():
    allocation = resource()
    if BINDING.exists() or QUEUE.exists(): raise ValueError('Never resubmit or overwrite a bound teacher cohort')
    source_path = RUN/'sources-v1/summary-private.json'; source = read(source_path)
    source_binding = RUN/'sources-v1/completion-binding-private.json'
    bound = read(source_binding)['files']
    if (source['status'] != 'native_daiminkan_context_sources_generated_unlabelled_not_fit_admitted'
            or source['planned'] != 576 or source['labels'] or source['student_updates']
            or sum(s['records'] for s in source['files'].values())+source['quarantined'] != 576
            or any(sha(p) != h for p, h in bound.items())):
        raise ValueError('Actual fixed contextual source generation required')
    for name in PUBLIC:
        path = ROOT/name
        if path.suffix == '.py': ast.parse(path.read_text(), filename=str(path))
        if path.suffix == '.slurm': subprocess.run(['bash', '-n', str(path)], check=True, timeout=10)
    test_code = """import runpy, unittest
namespace=runpy.run_path('tests/core/test_curriculum_kan_source_teacher_v1.py')
functions=[unittest.FunctionTestCase(v,description=k) for k,v in sorted(namespace.items()) if k.startswith('test_') and callable(v)]
assert len(functions)==3
suite=unittest.TestSuite(functions)
suite.addTests(unittest.defaultTestLoader.loadTestsFromName('tests.core.test_curriculum_daiminkan_context_v1'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
    test = subprocess.run([sys.executable, '-c', test_code], cwd=ROOT,
        capture_output=True, text=True, timeout=120)
    test_log = RUN/'teacher-preflight-tests-v1.log'
    with test_log.open('x') as stream: stream.write(test.stdout+test.stderr)
    if test.returncode: raise ValueError('Public-history/physical-wall/family tests failed')
    from tools.curriculum_mortal_source import WEIGHT_SHA256, BINARY_SHA256
    teacher = (ROOT/'../jev-baseline-evaluation').resolve()
    weights = teacher/'mortal-298k.pth'
    binary = teacher/'mortal-adapter-reference/libriichi/libriichi-3.12-x86_64-unknown-linux-gnu.so'
    if sha(weights) != WEIGHT_SHA256 or sha(binary) != BINARY_SHA256:
        raise ValueError('Pinned teacher artifacts differ')
    files = dict(bound)
    dependencies = [weights, binary, teacher/'mortal-adapter-reference/_libriichi_loader.py',
        teacher/'mortal-upstream/mortal/model.py', teacher/'mortal-upstream/mortal/engine.py',
        ROOT/'tools/curriculum_mortal_source.py', ROOT/'tools/label_curriculum_kan_sources_v1.py',
        ROOT/'tests/core/test_curriculum_kan_source_teacher_v1.py', source_path, source_binding, test_log]
    for p in [*[ROOT/p for p in PUBLIC], *dependencies]: files[str(p.resolve())] = sha(p)
    write(BINDING, dict(status='contextual_daiminkan_teacher_frozen_before_execution',
        files=files, allocation=allocation, source_cases=sum(s['records'] for s in source['files'].values()),
        source_quarantined=source['quarantined'], tests_returncode=0, positive_quota=False,
        student_updates=0, active_fit_changed=False))
    jobs = []
    def submit(*args):
        result = subprocess.run(['sbatch', '--parsable', *args], cwd=ROOT,
            capture_output=True, text=True, timeout=30)
        value = result.stdout.strip().split(';')[0]
        if result.returncode or not value.isdigit(): raise ValueError(result.stderr or result.stdout)
        jobs.append(value)
        return value
    try:
        job = submit('--hold', str(ENTRY))
        capture_job = submit('--dependency=afterany:'+job, '--kill-on-invalid-dep=yes', str(CAPTURE))
        first, second = scheduler(job), scheduler(capture_job)
        j, c = fields(first.stdout), fields(second.stdout)
        if (first.returncode or second.returncode or j.get('Reason') != 'JobHeldUser'
                or j.get('Command') != str(ENTRY) or c.get('Command') != str(CAPTURE)
                or f'afterany:{job}(' not in c.get('Dependency', '')):
            raise ValueError('Actual held teacher/capture job identities differ')
        write(QUEUE, dict(status='actual_contextual_daiminkan_teacher_jobs_bound_while_held',
            teacher_job=int(job), capture_job=int(capture_job), binding_sha256=sha(BINDING),
            observed_teacher=first.stdout, observed_capture=second.stdout,
            source_only_student_updates=0, source_and_teacher_raw_private=True))
        subprocess.run(['scontrol', 'release', job], check=True, capture_output=True, text=True, timeout=20)
        print(json.dumps(dict(status='actual_contextual_daiminkan_teacher_released',
            teacher_job=int(job), capture_job=int(capture_job), source_cases=read(BINDING)['source_cases'],
            student_updates=0)), flush=True)
    except BaseException as error:
        cleanup = [dict(job=job, returncode=subprocess.run(['scancel', job],
            capture_output=True, text=True, timeout=20).returncode) for job in jobs]
        write(RUN/'teacher-launch-error-v1-private.json', dict(error=repr(error), cleanup=cleanup))
        raise


def capture():
    allocation = resource(); binding = read(BINDING); queue = read(QUEUE)
    if queue['binding_sha256'] != sha(BINDING) or any(sha(p) != h for p, h in binding['files'].items()):
        raise ValueError('Original before-completion source/teacher binding changed')
    out = RUN/'teacher-terminal-v1'; out.mkdir(exist_ok=False)
    job = queue['teacher_job']; actual = scheduler(job)
    write(out/'scheduler-observation-private.json', dict(stdout=actual.stdout,
        stderr=actual.stderr, returncode=actual.returncode, allocation=allocation))
    proof = terminal_record(actual.stdout, job, ENTRY)
    receipt = ROOT/'logs'/f'curriculum-daiminkan-context-teacher-exit-{job}.json'
    exit_value = read(receipt) if receipt.exists() else None
    if proof['successful']:
        if exit_value is None or exit_value.get('phase') != 'teacher_labels':
            raise ValueError('Actual successful teacher receipt required')
        scheduler_exit(exit_value, actual.stdout, ENTRY)
    write(out/'terminal-evidence-private.json', dict(status='actual_contextual_daiminkan_teacher_terminal_preserved',
        proof=proof, exit_receipt=exit_value, labels_rewritten=False, student_updates=0))
    paths = [BINDING, QUEUE, *out.glob('*-private.json')]
    write(out/'completion-binding-private.json', dict(files={str(p): sha(p) for p in paths}))
    print(json.dumps(dict(status='actual_contextual_daiminkan_teacher_terminal_preserved',
        successful=proof['successful'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=('launch', 'capture'), required=True)
    a = p.parse_args()
    if a.phase == 'launch': launch()
    else: capture()
