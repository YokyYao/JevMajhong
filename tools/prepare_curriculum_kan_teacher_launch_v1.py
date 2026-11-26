"""Freeze and submit the fixed source-only teacher job inside a GPU allocation.

The original run used the equivalent private bounded driver. This public
entry accepts the reproducer's actual live parent job and optional GPU node.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource, fields
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parent-fit-job', type=int, required=True)
parser.add_argument('--nodelist')
args = parser.parse_args()
if args.parent_fit_job <= 0:
    raise ValueError('An actual positive scheduler job identity is required')
parent_job = str(args.parent_fit_job)
allocation = resource(); root = ROOT/'runs/curriculum-kan-sources-v1'
binding_path = root/'teacher-launch-binding-v1-private.json'
queue_path = root/'teacher-queue-v1-private.json'
if binding_path.exists() or queue_path.exists():
    raise ValueError('Never overwrite an existing source/teacher launch binding')
test_code = """import runpy, unittest
namespace = runpy.run_path('tests/core/test_curriculum_kan_source_teacher_v1.py')
names = sorted(k for k, value in namespace.items() if k.startswith('test_') and callable(value))
assert len(names) == 3
suite = unittest.TestSuite(unittest.FunctionTestCase(namespace[name], description=name) for name in names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
checked = subprocess.run([str(ROOT/'.venv/bin/python'), '-c', test_code],
    capture_output=True, text=True, timeout=120)
test_log = root/'teacher-preflight-tests-v1.log'
with test_log.open('x') as stream:
    stream.write(checked.stdout+checked.stderr)
if checked.returncode:
    raise ValueError('Source-family/wall/public-event checks failed: '+str(checked.returncode))
for name in ('cluster/curriculum-kan-sources-v1.slurm', 'cluster/curriculum-kan-teacher-v1.slurm'):
    subprocess.run(['bash', '-n', name], check=True, timeout=10)
parent = subprocess.run(['scontrol', '-o', 'show', 'job', parent_job], capture_output=True,
    text=True, timeout=20, check=True).stdout
if fields(parent).get('JobState') != 'RUNNING':
    raise ValueError('Bind this prospective collection during the actual live3A fit')
source_binding = read(root/'sources-v1/completion-binding-private.json')['files']
if any(sha(p) != h for p, h in source_binding.items()):
    raise ValueError('The actual189-case source generation changed')
public = [
    'tools/generate_curriculum_kan_sources_v1.py', 'tools/label_curriculum_kan_sources_v1.py',
    'tools/prepare_curriculum_kan_teacher_launch_v1.py',
    'configs/diagnostics/curriculum-kan-sources-v1.json',
    'configs/diagnostics/curriculum-kan-teacher-v1.json',
    'cluster/curriculum-kan-sources-v1.slurm', 'cluster/curriculum-kan-teacher-v1.slurm',
    'tests/core/test_curriculum_kan_source_teacher_v1.py',
    'docs/experiments/plans/curriculum-kan-sources-20261124.md',
    'docs/experiments/plans/curriculum-kan-teacher-20261124.md',
    'tools/curriculum_mortal_source.py']
files = dict(source_binding)
for path in [*[ROOT/p for p in public], Path(__file__), test_log,
    root/'sources-v1/completion-binding-private.json']:
    files[str(path.resolve())] = sha(path)
write(binding_path, dict(status='kan_teacher_source_and_recipe_frozen_before_execution',
    files=files, parent_fit_job=parent_job, observed_running_parent=parent,
    preflight_test_returncode=checked.returncode, source_only_not_student_training=True,
    source_cases=189, source_quarantine=3, active_fit_changed=False, allocation=allocation))
command = ['sbatch', '--parsable', '--dependency=afterany:'+parent_job, '--kill-on-invalid-dep=yes', '--nice=0']
if args.nodelist:
    command.append('--nodelist='+args.nodelist)
command.append('cluster/curriculum-kan-teacher-v1.slurm')
submitted = subprocess.run(command, capture_output=True, text=True, timeout=30)
write(root/'teacher-submission-observation-v1-private.json', dict(command=command,
    stdout=submitted.stdout, stderr=submitted.stderr, returncode=submitted.returncode))
if submitted.returncode or not submitted.stdout.strip().split(';')[0].isdigit():
    raise ValueError('Actual scheduler submission did not return a job identity')
job = submitted.stdout.strip().split(';')[0]
observed = subprocess.run(['scontrol', '-o', 'show', 'job', job], capture_output=True,
    text=True, timeout=20, check=True).stdout
value = fields(observed)
if (value.get('JobId') != job or value.get('JobState') != 'PENDING'
        or 'afterany:'+parent_job+'(' not in value.get('Dependency', '')
        or value.get('Command') != str(ROOT/'cluster/curriculum-kan-teacher-v1.slurm')):
    raise ValueError('Actual queued job differs from its requested source-teacher entry')
write(queue_path, dict(status='actual_fixed_kan_teacher_gpu_job_submitted', teacher_job=int(job),
    binding_sha256=sha(binding_path), dependency='afterany:'+parent_job,
    source_cases=189, actual_scheduler=observed, submission_stdout=submitted.stdout,
    source_collection_only_student_updates=0, automatic_fit=False, raw_sources_and_labels_private=True))
print(json.dumps(dict(status='kan_teacher_source_frozen_and_actual_gpu_job_queued',
    teacher_job=int(job), tests_returncode=checked.returncode, source_cases=189,
    labels_generated_now=0, student_updates=0)), flush=True)
