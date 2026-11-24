"""Freeze retained placement registration without admitting incomplete work."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import register_curriculum_placement_parent_v1 as registration
from tools import seal_curriculum_placement_retained_parent_v1 as producer
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource


def main():
    allocation = resource()
    if registration.BINDING.exists(): raise ValueError('Never replace a registration entry binding')
    prior = ROOT/'runs/curriculum-placement-retained-seal-v1/entry-preflight-v1/completion-binding-private.json'
    files = dict(read(prior)['files']); files[str(prior)] = sha(prior)
    log_path = ROOT/'logs/curriculum-placement-parent-registration-tests-v1.log'; log = log_path.read_text()
    if 'Ran 7 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Seven registration budget/source/runtime tests required')
    for path in (log_path, *[ROOT/p for p in (*registration.original.PUBLIC_FILES, *registration.PUBLIC)]):
        files[str(path.resolve())] = sha(path)
    for name in registration.PUBLIC:
        if name.endswith('.py'): compile(ast.parse((ROOT/name).read_text()), name, 'exec')
    if registration.registration_function().__code__ is not registration.original.register.__code__:
        raise ValueError('Actual original full registration code object must be retained')
    try: producer.parent_artifacts(producer.PARENT)
    except (FileNotFoundError, ValueError) as error: refusal = dict(type=type(error).__name__, detail=str(error))
    else: raise ValueError('This before-parent-completion binding expects the actual unavailable future parent')
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Inherited interface or registration source changed')
    out = registration.BINDING.parent; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, allocation=allocation, actual_parent_refusal=refusal))
    summary = dict(status='retained_placement_registration_entry_frozen_not_actual_registration',
        unit_tests_passed=7, actual_original_registration_code_object_preserved=True,
        parent_loss_method=registration.runtime.PARENT_LOSS, child_loss_method=registration.runtime.LOSS,
        actual_unavailable_parent_refused=True, actual_parent_sealed=False, child_registered=False,
        full57_sources_and_count_derived_budget_required=True, real_model_loaded=False,
        model_forwards=0, new_games=0, student_updates=0, child_jobs_submitted=False,
        real_parent_data_registration_and_model_integration_still_pending=True)
    write(out/'summary-private.json', summary)
    for path in (out/'registered-private.json', out/'summary-private.json'): files[str(path)] = sha(path)
    write(registration.BINDING, dict(files=files))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
