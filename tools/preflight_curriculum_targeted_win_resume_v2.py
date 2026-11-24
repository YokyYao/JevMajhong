"""Freeze recovery software and audit completed cases, without resuming a job."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields
from tools.curriculum_targeted_win_resume_v2 import (
    RUN, RESUME, BINDING, ENTRY, COLLECTOR, completed_from, resumed_function_tree,
)

PUBLIC = ('tools/curriculum_targeted_win_resume_v2.py',
    'tools/preflight_curriculum_targeted_win_resume_v2.py',
    'tests/core/test_curriculum_targeted_win_resume_v2.py',
    'cluster/curriculum-targeted-win-resume-v2.slurm',
    'docs/experiments/plans/curriculum-targeted-win-resume-20261124.md')


def main():
    allocation = resource(); started = time.monotonic()
    reg_path = RUN/'registered-private.json'; reg = read(reg_path)
    if BINDING.exists(): raise ValueError('Never overwrite a recovery entry binding')
    source = read(reg['source_registration'])
    sys.path[:0] = [source['source'], source['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    log_path = ROOT/'logs/curriculum-targeted-win-resume-tests-v2.log'
    log = log_path.read_text()
    if 'Ran 7 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Seven exact-calculation/I-O/terminal tests must pass first')
    files = {**reg['files'], **reg['active_fit_files']}
    for p in (reg_path, log_path, *[ROOT/p for p in PUBLIC]): files[str(p.resolve())] = sha(p)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Frozen source or active input changed')
    for name in PUBLIC:
        if name.endswith('.py'): ast.parse((ROOT/name).read_text())
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=10)
    compile(resumed_function_tree(COLLECTOR.read_text()), str(COLLECTOR), 'exec')
    out = RESUME/'entry-preflight-v1'; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, allocation=allocation,
        live_case_inspection_only=True, no_jobs_submitted=True, student_updates=0))
    try:
        selected = reg['selected'][:27]
        recovered, observed_files = completed_from(RUN/'shards/shard-00', selected, reg)
        cases = [v for v in recovered.values() if v['kind'] == 'case']
        if len(cases) < 2: raise ValueError('At least two actual complete cases required')
        # Only complete immutable case files are bound. Active JSONL files may
        # continue growing and are intentionally not frozen by this preflight.
        fixed_cases = {v['path']: v['sha256'] for v in cases}
        if any(sha(p) != h for p, h in fixed_cases.items()): raise ValueError('A complete case changed')
        actual = scheduler('319759_0'); values = fields(actual.stdout)
        if (actual.returncode or values.get('JobState') != 'RUNNING'
                or values.get('ArrayJobId') != '319759' or values.get('ArrayTaskId') != '0'):
            raise ValueError('Observe the original live writer; never infer failure from a quiet log')
        summary = dict(status='targeted_win_resume_preflight_passed_no_continuation_started',
            unit_tests_passed=7, audited_complete_cases=len(cases),
            audited_complete_worlds=32*len(cases),
            audited_complete_outer_arms=sum(len(p) for v in cases for b in v['case']['blocks'] for p in b['pairs']),
            full_ledgers_and_soft_target_reconstruction=True, original_case_calculations_AST_unchanged=True,
            original_writer_confirmed_live=True, original_writer_not_cancelled_or_restarted=True,
            population_teacher_worlds_and_targets_unchanged=True, partial_worlds_admitted=False,
            model_loaded=False, model_forwards=0, new_games=0, student_updates=0,
            continuation_jobs_submitted=False, actual_terminal_and_full_merge_still_required=True,
            seconds=time.monotonic()-started)
        write(out/'summary-private.json', summary)
        write(out/'live-writer-observation-private.json', dict(raw=actual.stdout,
            completed_case_files=fixed_cases, active_output_files_not_bound=True))
        files.update(fixed_cases)
        for p in (out/'registered-private.json', out/'summary-private.json', out/'live-writer-observation-private.json'):
            files[str(p.resolve())] = sha(p)
        write(BINDING, dict(status='targeted_win_resume_entry_bound_not_started', files=files,
            fixed_array_job='319759', original_writer_live_at_binding=True, no_jobs_submitted=True,
            raw_records_private=True, student_updates=0))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(error=repr(error)))
        raise


if __name__ == '__main__': main()
