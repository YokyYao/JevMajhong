"""Freeze retained placement interfaces, never a parent/model/data admission."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_placement_parent_loss_v2 import METHOD

PUBLIC = ('tools/curriculum_placement_parent_runtime_v1.py',
    'tools/curriculum_placement_retained_parent_v1.py',
    'tools/train_curriculum_placement_parent_v1.py',
    'tools/evaluate_curriculum_placement_parent_v1.py',
    'tools/audit_curriculum_placement_parent_v1.py',
    'tools/preflight_curriculum_placement_parent_interface_v1.py',
    'tests/core/test_curriculum_placement_parent_interface_v1.py',
    'docs/experiments/plans/curriculum-placement-retained-interface-20261124.md')


def main():
    allocation = resource(); run = ROOT/'runs/curriculum-placement-parent-interface-v1'
    out = run/'preflight-v1'; out.mkdir(parents=True, exist_ok=False)
    actual = ROOT/'runs/curriculum-progression-parent-retention-v1/fit-plan/registered.json'
    reg = read(actual)
    objective = ROOT/'runs/curriculum-placement-parent-loss-v1/objective-preflight-v2'
    proof = read(objective/'summary-private.json'); files = dict(read(objective/'completion-binding-private.json')['files'])
    if (proof['method'] != METHOD or proof['actual_prepared_TRAIN_records'] != 91647
            or proof['maximum_all_records_objective_or_CE_difference'] != 0
            or proof['maximum_representative_gradient_difference'] != 0
            or proof['actual_sources_checked'] != 48 or proof['required_final_sources'] != 57):
        raise ValueError('Actual available-data objective proof, with missing pools explicit, required')
    test_log = ROOT/'logs/curriculum-placement-parent-interface-tests-v3.log'
    log = test_log.read_text()
    if 'Ran 7 tests' not in log or not log.endswith('\nOK\nexit_code=0\n') or 'ResourceWarning' in log:
        raise ValueError('Seven interface tests must pass with all test file handles closed')
    metric_log = ROOT/'logs/curriculum-placement-parent-metrics-tests-v2.log'
    if 'Ran 5 tests' not in metric_log.read_text() or not metric_log.read_text().endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Five normalized terminal metric tests must pass')
    local_journal = ROOT/'jevmahjong/training/metric_journal.py'
    frozen_journal = Path(reg['source'])/'jevmahjong/training/metric_journal.py'
    if sha(local_journal) != sha(frozen_journal): raise ValueError('The actual tested journal differs from the pinned journal')
    paths = [actual, objective/'summary-private.json', objective/'completion-binding-private.json',
        test_log, metric_log, local_journal, frozen_journal,
        run/'failed-source-v1/audit_curriculum_placement_parent_v1.py',
        ROOT/'logs/curriculum-placement-parent-interface-tests-v1.log',
        *[ROOT/p for p in PUBLIC], ROOT/'tools/curriculum_placement_parent_metrics_v2.py',
        ROOT/'tools/curriculum_placement_parent_metrics_v1.py',
        ROOT/'tests/core/test_curriculum_placement_parent_metrics_v2.py']
    for path in paths: files[str(path.resolve())] = sha(path)
    for name in PUBLIC:
        if name.endswith('.py'): ast.parse((ROOT/name).read_text(), filename=name)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('An objective/runtime source changed')
    write(out/'registered-private.json', dict(files=files, allocation=allocation, student_updates=0,
        actual_parent_not_sealed=True, full57_data_not_admitted=True))
    summary = dict(status='retained_placement_interfaces_checked_not_parent_or_fit_admitted',
        method=METHOD, interface_tests=7, metric_tests=5,
        actual_tracker_pinned_journal_and_full_auditor_function_checked=True,
        synthetic_software_observations_only=True, real_model_optimizer_updates=0,
        model_loaded=False, model_forwards=0, new_games=0, child_jobs_submitted=False,
        original_numeric_schedule_policy_baselines_and_grade_unchanged=True,
        actual_parent_seal_registration_isolated_model_integration_full_data_and_fit_still_required=True,
        final_confirmation_unopened=True)
    write(out/'summary-private.json', summary)
    for path in (out/'registered-private.json', out/'summary-private.json'): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
