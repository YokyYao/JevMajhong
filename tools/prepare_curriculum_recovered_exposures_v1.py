"""Use the full recovered corpus for unchanged placement mix and exposures."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import curriculum_targeted_win_resumed_pipeline_v3 as pipeline
from tools.curriculum_targeted_win_resumed_pipeline_v2 import adapted_function
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource

RUN = ROOT/'runs/curriculum-placement-recovered-exposures-v1'
BINDING = RUN/'entry-binding-private.json'
ENTRY = ROOT/'cluster/curriculum-placement-recovered-exposures-v1.slurm'
DATA = pipeline.PREPARED/'data-ready-private.json'
MIX = RUN/'proposal-v1'
EXPOSURES = RUN/'data-v1'
CONFIG = ROOT/'configs/training/curriculum-placement-recovered-exposures-v1.json'
SOURCES = {
    'mix': ('tools.prepare_curriculum_placement_mix_v1', {
        'parent_path': ("ROOT/'runs/curriculum-placement-data-v3/data-v1/data-ready-private.json'",
            "ROOT/'runs/curriculum-targeted-win-resume-v3/placement-data-v5/data-v1/data-ready-private.json'"),
        'out': ("ROOT/'runs/curriculum-placement-mix-v1/proposal-v1'",
            "ROOT/'runs/curriculum-placement-recovered-exposures-v1/proposal-v1'")}),
    'exposures': ('tools.prepare_curriculum_placement_exposures_v1', {
        'config_path': ("ROOT/'configs/training/curriculum-placement-exposures-v1.json'",
            "ROOT/'configs/training/curriculum-placement-recovered-exposures-v1.json'")}),
}
PUBLIC = ('tools/prepare_curriculum_recovered_exposures_v1.py',
    'tests/core/test_curriculum_recovered_exposures_v1.py',
    'configs/training/curriculum-placement-recovered-exposures-v1.json',
    'cluster/curriculum-placement-recovered-exposures-v1.slurm',
    'docs/experiments/plans/curriculum-recovered-exposures-20261124.md')


def original_source(phase):
    module, replacements = SOURCES[phase]; path = ROOT/(module.replace('.', '/')+'.py')
    return module, path, adapted_function(path.read_text(), 'prepare', replacements)


def combine_files(original, additional):
    result = dict(original)
    for path, digest in additional.items():
        if path in result and result[path] != digest: raise ValueError('Conflicting immutable producer ancestry')
        result[path] = digest
    return result


def completed_input():
    producer_path = pipeline.PREPARED/'resumed-producer-binding-private.json'
    producer = read(producer_path); ready = read(DATA)
    if (not producer['actual_full_original_projection_complete']
            or ready['status'] != 'placement_all_sources_including_targeted_win_special_prepared_not_fit'
            or ready['optimizer_updates'] != 0
            or any(sha(p) != h for p, h in producer['files'].items())):
        raise ValueError('Actual completed full recovered projection and unchanged source required')
    files = dict(producer['files']); files[str(producer_path)] = sha(producer_path)
    files[str(DATA)] = sha(DATA)
    return files


def execute(phase):
    resource(); binding = read(BINDING)
    if (binding['status'] != 'recovered_exposure_entry_bound_not_executed'
            or any(sha(p) != h for p, h in binding['files'].items())):
        raise ValueError('Frozen recovered exposure software required')
    prerequisite = completed_input(); prerequisite = combine_files(prerequisite, binding['files'])
    prerequisite[str(BINDING)] = sha(BINDING)
    out = MIX if phase == 'mix' else EXPOSURES
    if out.exists(): raise ValueError('Never overwrite recovered mix or exposures')
    if phase == 'exposures':
        proposal_path = MIX/'proposal-private.json'; proposal = read(proposal_path)
        if proposal['status'] != 'count_only_placement_mix_proposed_not_parent_or_fit_admitted':
            raise ValueError('Actual completed count-only proposal required')
        completed = MIX/'completion-binding-private.json'
        prerequisite = combine_files(prerequisite, read(completed)['files'])
        prerequisite[str(completed)] = sha(completed); prerequisite[str(proposal_path)] = sha(proposal_path)
        if any(sha(p) != h for p, h in prerequisite.items()): raise ValueError('Count-only proposal ancestry changed')
    module, path, tree = original_source(phase); original = importlib.import_module(module)
    def writer(target, value):
        if Path(target).resolve() == (out/'completion-binding-private.json').resolve():
            value = dict(value, files=combine_files(value['files'], prerequisite))
        write(target, value)
    scope = dict(vars(original), write=writer)
    exec(compile(tree, str(path), 'exec'), scope); scope['prepare']()
    if any(sha(p) != h for p, h in prerequisite.items()): raise ValueError('Producer ancestry changed during preparation')
    print(json.dumps(dict(status='actual_recovered_'+phase+'_prepared_not_fit_admitted',
        original_calculations_unchanged=True, student_updates=0)), flush=True)


def bind():
    allocation = resource(); pipeline_binding, _ = pipeline.verify_entry()
    if BINDING.exists(): raise ValueError('Never replace an exposure entry binding')
    log_path = ROOT/'logs/curriculum-recovered-exposures-tests-v1.log'; log = log_path.read_text()
    if 'Ran 6 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Six path/calculation/provenance/full-input contracts required')
    files = dict(pipeline_binding['files']); files[str(pipeline.BINDING)] = sha(pipeline.BINDING)
    for phase in SOURCES:
        _, path, tree = original_source(phase); compile(tree, str(path), 'exec'); files[str(path)] = sha(path)
    for path in (log_path, *[ROOT/p for p in PUBLIC]): files[str(path.resolve())] = sha(path)
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=10)
    try: completed_input()
    except (FileNotFoundError, ValueError): missing_full_input_rejected = True
    else: raise ValueError('Before-data-completion binding expects the actual missing full corpus')
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Bound data producer or source changed')
    RUN.mkdir(parents=True, exist_ok=True)
    write(BINDING, dict(status='recovered_exposure_entry_bound_not_executed', files=files,
        allocation=allocation, unit_tests_passed=6, actual_incomplete_input_rejected=missing_full_input_rejected,
        original_calculations_changed_only_at_paths=True, actual_exposures_materialized=False,
        all57_sources_required=True, new_jobs_submitted=False, student_updates=0))
    print(json.dumps(dict(status='recovered_exposure_entry_bound_not_executed',
        unit_tests_passed=6, actual_incomplete_input_rejected=True, student_updates=0)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=['bind', 'mix', 'exposures'], required=True)
    args = p.parse_args()
    if args.phase == 'bind': bind()
    else: execute(args.phase)
