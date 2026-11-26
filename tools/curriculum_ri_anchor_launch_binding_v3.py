"""Supply the missing real launch binding before the unchanged2596 fit.

Preserve the original frozen recipe/activation body. Only registration source
binding and the pre-submit launch-file producer are added in an isolated scope.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import curriculum_ri_anchor_fit_v1 as original
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields

BINDING = original.RUN/'activation-binding-v3-private.json'
PREPARED = original.RUN/'launch-binding-repair-v3/completion-binding-private.json'
ENTRY = ROOT/'cluster/curriculum-ri-anchor-activation-v3.slurm'
PUBLIC = ('tools/curriculum_ri_anchor_launch_binding_v3.py',
    'tests/core/test_curriculum_ri_anchor_launch_binding_v3.py',
    'cluster/curriculum-ri-anchor-activation-v3.slurm',
    'docs/experiments/plans/curriculum-ri-anchor-launch-binding-v3-20261125.md')


def emit_launch_binding(registered, gate, destination, extra_files):
    """Require the original actual gate; create the trainer's input exclusively."""
    registered, gate, destination = map(Path, (registered, gate, destination))
    if destination.exists():
        raise ValueError('Never overwrite an actual launch binding')
    reg = read(registered)
    original.contract.require_registration(reg)
    original.contract.require_launch(registered)
    if read(gate)['registered_sha256'] != sha(registered):
        raise ValueError('Actual original launch gate differs from registration')
    files = dict(reg['inputs'])
    original.merge(files, extra_files)
    for path in (registered, gate):
        files[str(path)] = sha(path)
    original.contract.verify_files(files)
    payload = dict(files=files, registered_sha256=sha(registered),
        actual_registration_before_first_update=True, original_launch_gate_unchanged=True,
        launch_binding_written_before_fit_submission=True, optimizer_updates=0,
        numerical_recipe_changed=False, integration_control_updates_excluded=True)
    write(destination, payload)
    return payload


def activation_function(binding, extra_files):
    """Reuse the original code object; hook only its metadata write/submit."""
    registered = original.REGISTERED
    launch = original.RUN/'launch-binding-private.json'
    gate = original.RUN/'launch-gate-private.json'
    before_write, before_submit = original.write, original.submit

    def bound_write(path, payload):
        if Path(path) == registered:
            payload = copy.deepcopy(payload)
            original.merge(payload['inputs'], extra_files)
        elif Path(path) == original.QUEUE:
            if not launch.is_file():
                raise ValueError('Actual queue must bind the emitted launch file')
            payload = copy.deepcopy(payload)
            original.merge(payload['files'], extra_files)
            payload['files'][str(launch)] = sha(launch)
        return before_write(path, payload)

    def submit(role, dependency, limit, cpus=4, memory='48G'):
        if role == 'fit':
            emit_launch_binding(registered, gate, launch, extra_files)
        elif not launch.is_file():
            raise ValueError('Fit launch binding must exist before audit/capture submission')
        return before_submit(role, dependency, limit, cpus, memory)

    function = original.activate
    return FunctionType(function.__code__, dict(function.__globals__, BRIDGE=Path(binding),
        write=bound_write, submit=submit), function.__name__, function.__defaults__, function.__closure__)


def prepare():
    resource()
    out = PREPARED.parent
    if out.exists():
        raise ValueError('Never overwrite launch-repair preparation evidence')
    if original.REGISTERED.exists() or original.QUEUE.exists():
        raise ValueError('This repair is only for the still-unstarted alternative')
    old = original.verify_binding(original.PREPARED)
    binding = original.verify_binding(original.BRIDGE)
    missing = original.RUN/'launch-binding-private.json'
    if missing.exists():
        raise ValueError('Review an existing launch binding instead of replacing it')
    tests = subprocess.run([sys.executable, '-m', 'unittest',
        'tests.core.test_curriculum_ri_anchor_launch_binding_v3', '-v'],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    out.mkdir()
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=15)
    files = dict(old['files']); original.merge(files, binding['files'])
    for path in [*[ROOT/name for name in PUBLIC], original.PREPARED, original.BRIDGE]:
        files[str(path)] = sha(path)
    summary = dict(status='missing2596_launch_binding_producer_prepared_no_registration_or_fit',
        original_activation_code_object_reused=True, original_numeric_recipe_unchanged=True,
        new_source_hashes_enter_actual_registration=True,
        launch_binding_emitted_after_original_gate_before_fit_submission=True,
        current2472_untouched=True, old_pending_activation_not_yet_replaced=True,
        model_loaded=False, student_updates=0)
    write(out/'summary-private.json', summary)
    for path in (out/'summary-private.json', out/'tests.log'):
        files[str(path)] = sha(path)
    write(PREPARED, dict(files=files))
    print(summary, flush=True)


def enqueue():
    allocation = resource()
    prepared = original.verify_binding(PREPARED)
    old_binding = original.verify_binding(original.BRIDGE)
    if BINDING.exists() or original.REGISTERED.exists() or original.QUEUE.exists():
        raise ValueError('No duplicate replacement or already-started alternative')
    old_job = str(old_binding['activation_job']); old = scheduler(old_job); observed = fields(old.stdout)
    if (old.returncode or observed.get('JobId') != old_job or observed.get('JobState') != 'PENDING'
            or observed.get('Command') != str(original.ENTRY)
            or 'afterany:'+str(old_binding['current_capture_job'])+'(' not in observed.get('Dependency', '')):
        raise ValueError('Replace only the verified own still-pending original activation')
    result = subprocess.run(['sbatch', '--parsable', '--partition=ws-ia', '--time=00:10:00',
        '--cpus-per-task=1', '--mem=6G', '--dependency=afterany:'+str(old_binding['current_capture_job']),
        '--kill-on-invalid-dep=yes', str(ENTRY)], cwd=ROOT,
        capture_output=True, text=True, timeout=30)
    new_job = result.stdout.strip().split(';')[0]
    if result.returncode or not new_job.isdigit():
        raise RuntimeError(result.stderr or result.stdout)
    new_record = scheduler(new_job); new_fields = fields(new_record.stdout)
    try:
        if (new_record.returncode or new_fields.get('JobState') != 'PENDING'
                or new_fields.get('Command') != str(ENTRY)
                or 'afterany:'+str(old_binding['current_capture_job'])+'(' not in new_fields.get('Dependency', '')):
            raise ValueError('Actual finite replacement activation dependency differs')
        files = dict(prepared['files'])
        replacement = dict(old_binding, files=files, activation_job=new_job,
            original_activation_job=old_job, original_activation_binding=str(original.BRIDGE),
            original_activation_source_preserved=True, launch_binding_repair_only=True,
            actual_old_pending=old.stdout, actual_new_pending=new_record.stdout, allocation=allocation)
        write(BINDING, replacement)
        cancelled = subprocess.run(['scancel', old_job], capture_output=True, text=True, timeout=20)
        if cancelled.returncode:
            raise RuntimeError('Original own pending activation could not be cancelled: '+cancelled.stderr)
        terminal = scheduler(old_job)
        if terminal.returncode or fields(terminal.stdout).get('JobState') != 'CANCELLED':
            raise ValueError('Actual cancellation of the original pending activation must be observable')
        write(original.RUN/'activation-v1-superseded-by-v3-private.json', dict(
            original_job=old_job, replacement_job=new_job, old_before=old.stdout,
            old_after=terminal.stdout, new_pending=new_record.stdout,
            original_source_binding_unchanged=True, student_updates=0))
    except BaseException:
        actual = scheduler(new_job)
        if fields(actual.stdout).get('Command') == str(ENTRY):
            subprocess.run(['scancel', new_job], check=True, capture_output=True, text=True, timeout=20)
        raise
    print(dict(status='pending_activation_replaced_for_launch_binding_only',
        old_job=old_job, new_job=new_job, original_capture_dependency_unchanged=True), flush=True)


def activate():
    resource(); binding = original.verify_binding(BINDING)
    if str(binding['activation_job']) != os.environ['SLURM_JOB_ID']:
        raise ValueError('Exact actual replacement activation job required')
    extra = dict(read(PREPARED)['files'])
    extra[str(BINDING)] = sha(BINDING)
    activation_function(BINDING, extra)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'enqueue', 'activate'), required=True)
    arguments = parser.parse_args(); globals()[arguments.phase]()
