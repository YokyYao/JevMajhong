"""Freeze the seal producer; do not generate a seal or admit future weights."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import seal_curriculum_placement_retained_parent_v1 as producer
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource


def main():
    allocation = resource()
    prior = ROOT/'runs/curriculum-placement-parent-interface-v1/preflight-v1/completion-binding-private.json'
    files = dict(read(prior)['files']); files[str(prior)] = sha(prior)
    log_path = ROOT/'logs/curriculum-placement-retained-seal-tests-v1.log'; log = log_path.read_text()
    if 'Ran 7 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Seven retained-seal phase/data/refusal tests required')
    for path in (log_path, *[ROOT/p for p in producer.PUBLIC]): files[str(path.resolve())] = sha(path)
    for path in producer.PUBLIC:
        if path.endswith('.py'): compile(ast.parse((ROOT/path).read_text()), path, 'exec')
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Prior interface or new producer source changed')
    try:
        producer.parent_artifacts(producer.PARENT)
    except (FileNotFoundError, ValueError) as error:
        refusal = dict(error_type=type(error).__name__, detail=str(error))
    else:
        raise ValueError('Before-parent-completion preflight expects the actual unavailable/unqualified parent')
    out = ROOT/'runs/curriculum-placement-retained-seal-v1/entry-preflight-v1'
    out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, allocation=allocation,
        actual_missing_parent_refusal=refusal, no_seal_produced=True))
    summary = dict(status='retained_placement_seal_producer_frozen_not_actual_parent_admitted',
        unit_tests_passed=7, actual_unavailable_parent_refused=True,
        parent_loss_method=producer.retained.PARENT_LOSS, child_loss_method=producer.retained.CHILD_LOSS,
        real_model_loaded=False, model_forwards=0, new_games=0, student_updates=0,
        actual_parent_sealed=False, child_registered=False, child_jobs_submitted=False,
        actual_qualified3B_full57_data_and_exposures_still_required=True,
        registration_and_real_model_integration_still_pending=True)
    write(out/'summary-private.json', summary)
    for path in (out/'registered-private.json', out/'summary-private.json'): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
