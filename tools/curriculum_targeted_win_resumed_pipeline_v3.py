"""Admit complete-world recovery into the unchanged full108 data pipeline."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import curriculum_targeted_win_resumed_pipeline_v2 as previous
from tools import curriculum_targeted_win_resume_v3 as recovery
from tools.curriculum_targeted_world_cache_v1 import METHOD
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, terminal_record, fields

ASSEMBLED = recovery.RUN/'assembled-v1'
PREPARED = recovery.RUN/'placement-data-v5/data-v1'
BINDING = recovery.RUN/'pipeline-entry-binding-private.json'
ENTRY = ROOT/'cluster/curriculum-targeted-win-resumed-pipeline-v3.slurm'
SOURCE_SPECS = {
    'merge': ('tools.merge_curriculum_targeted_win_labels_v1', 'merge', {
        'root': ("ROOT/'runs/curriculum-targeted-win-labels-v1'",
            "ROOT/'runs/curriculum-targeted-win-resume-v3/assembled-v1'")}),
    'prepare': ('tools.prepare_curriculum_placement_data_v3', 'prepare', {
        'control_path': ("ROOT/'runs/curriculum-targeted-win-labels-v1/merged-v1/data-ready-private.json'",
            "ROOT/'runs/curriculum-targeted-win-resume-v3/assembled-v1/merged-v1/data-ready-private.json'"),
        'out': ("ROOT/'runs/curriculum-placement-data-v3/data-v1'",
            "ROOT/'runs/curriculum-targeted-win-resume-v3/placement-data-v5/data-v1'")}),
}
PUBLIC = ('tools/curriculum_targeted_win_resumed_pipeline_v3.py',
    'tests/core/test_curriculum_targeted_win_resumed_pipeline_v3.py',
    'cluster/curriculum-targeted-win-resumed-pipeline-v3.slurm',
    'docs/experiments/plans/curriculum-targeted-world-pipeline-20261124.md')


def original_source(phase):
    module, name, replacements = SOURCE_SPECS[phase]
    path = ROOT/(module.replace('.', '/')+'.py')
    return module, name, path, previous.adapted_function(path.read_text(), name, replacements)


def run_original(phase):
    module, name, path, tree = original_source(phase)
    scope = dict(vars(importlib.import_module(module)))
    exec(compile(tree, str(path), 'exec'), scope); scope[name]()


def frozen(path, status):
    value = read(path)
    if value['status'] != status or any(sha(p) != h for p, h in value['files'].items()):
        raise ValueError('Unchanged actual entry and all bound source files required')
    return value


def verify_entry():
    # Also validate the old pipeline whose full merge/prepare code is reused.
    previous.verify_entry()
    binding = frozen(BINDING, 'complete_world_targeted_pipeline_bound_not_executed')
    actual = frozen(recovery.BINDING, 'targeted_win_resume_entry_bound_not_started')
    if actual.get('world_cache_method') != METHOD: raise ValueError('Explicit complete-world recovery required')
    return binding, read(previous.RUN/'registered-private.json')


def verify_recovery_lineage(folder):
    folder = folder.resolve()
    if recovery.RUN.resolve() not in folder.parents:
        return previous.verify_recovery_lineage(folder)
    complete_path = folder/'resumption-complete-private.json'; complete = read(complete_path)
    if (complete['binding_sha256'] != sha(recovery.BINDING)
            or complete.get('world_cache_method') != METHOD
            or not complete['original_case_loop_unchanged'] or complete['partial_worlds_admitted']
            or any(type(complete[k]) is not int or complete[k] < 0
                for k in ('exact_recovered_views', 'newly_processed_views'))
            or complete['exact_recovered_views']+complete['newly_processed_views'] != 27):
        raise ValueError('Full unchanged27-view complete-world recovery required')
    admission_path = Path(complete['admission']); admission = read(admission_path)
    if (admission['status'] != 'actual_terminal_complete_cases_preserved_missing_cases_only'
            or not admission['same_fixed_population_worlds_teacher_and_targets']
            or admission['completed_views'] != complete['exact_recovered_views']
            or not admission['terminals'] or any(sha(p) != h for p, h in admission['files'].items())):
        raise ValueError('Original complete-case admission or prior writer proofs changed')
    files = dict(admission['files'])
    commands = {
        str((ROOT/'cluster/curriculum-targeted-win-labels-v1.slurm').resolve()): 'curriculum-targeted-win-labels',
        str(previous.ROOT/'cluster/curriculum-targeted-win-resume-v2.slurm'): 'curriculum-targeted-win-resume',
        str(recovery.ENTRY.resolve()): 'curriculum-targeted-win-resume-v3',
    }
    for proof in admission['terminals']:
        command = fields(proof['raw']).get('Command')
        if command not in commands: raise ValueError('Unknown prior recovery producer')
        receipt_path = ROOT/'logs'/f"{commands[command]}-exit-{proof['job_id']}.json"
        if Path(proof['receipt_path']).resolve() != receipt_path.resolve():
            raise ValueError('Exact original exit-receipt path required')
        if read(receipt_path) != proof['exit_receipt']: raise ValueError('Original prior exit changed')
        actual = terminal_record(proof['raw'], proof['job_id'], Path(command))
        if any(proof[k] != actual[k] for k in actual): raise ValueError('Original prior terminal differs')
        files[str(receipt_path.resolve())] = sha(receipt_path)
    manifest_path = folder/'world-cache-binding-private.json'
    if (Path(complete['world_cache_manifest']).resolve() != manifest_path
            or sha(manifest_path) != complete['world_cache_manifest_sha256']):
        raise ValueError('Exact completed attempt world-cache manifest required')
    manifest = read(manifest_path); collection = read(previous.RUN/'registered-private.json')
    expected = dict(collection_sha256=sha(previous.RUN/'registered-private.json'),
        teacher_identity=collection['continuation_model']['identity_sha256'],
        producer_binding_sha256=sha(recovery.BINDING), purpose='actual_fixed_label_collection',
        world_cache_method=METHOD)
    if (manifest['method'] != METHOD or manifest['original_collection_identity'] != expected
            or manifest['partial_world_targets_admitted'] or not manifest['all_current_arms_required']
            or any(sha(p) != h for p, h in manifest['files'].items())):
        raise ValueError('Production identity and unchanged complete-world records required')
    files.update(manifest['files'])
    for path in (admission_path, complete_path, manifest_path): files[str(path.resolve())] = sha(path)
    return files


def pipeline_function(name):
    """Reuse the identical full pipeline code object with an isolated namespace."""
    function = getattr(previous, name)
    scope = dict(vars(previous), ASSEMBLED=ASSEMBLED, PREPARED=PREPARED, BINDING=BINDING,
        verify_entry=verify_entry, run_original=run_original,
        verify_recovery_lineage=verify_recovery_lineage, require_terminal=recovery.require_terminal)
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)


def bind():
    allocation = resource(); old, reg = previous.verify_entry()
    recovered = frozen(recovery.BINDING, 'targeted_win_resume_entry_bound_not_started')
    if recovered.get('world_cache_method') != METHOD: raise ValueError('Complete-world recovery must be bound first')
    if BINDING.exists(): raise ValueError('Never overwrite pipeline admission')
    log_path = ROOT/'logs/curriculum-targeted-world-pipeline-tests-v1.log'; log = log_path.read_text()
    if 'Ran 7 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Seven full-pipeline/lineage tests required')
    files = {**old['files'], **recovered['files'], str(previous.BINDING): sha(previous.BINDING),
        str(recovery.BINDING): sha(recovery.BINDING)}
    for phase in SOURCE_SPECS:
        _, _, path, tree = original_source(phase); compile(tree, str(path), 'exec'); files[str(path)] = sha(path)
    for path in (*[ROOT/p for p in PUBLIC], log_path): files[str(path.resolve())] = sha(path)
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=10)
    try: previous.verify_complete(previous.RUN/'shards/shard-00', 0, reg)
    except ValueError: pass
    else: raise ValueError('Before-completion binding expects the actual incomplete original shard')
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Bound input changed')
    result = dict(status='complete_world_targeted_pipeline_bound_not_executed', files=files,
        allocation=allocation, unit_tests_passed=7, original_full_pipeline_code_objects_unchanged=True,
        actual_incomplete_shard_rejected=True, all108_views_and_original_partitions_required=True,
        full_merge_and_projection_not_yet_run=True, no_jobs_submitted=True, student_updates=0)
    write(BINDING, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'allocation')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=['bind', 'merge', 'prepare'], required=True)
    p.add_argument('--shard-folder', type=Path, action='append'); p.add_argument('--archive', type=Path, action='append')
    args = p.parse_args()
    if args.phase == 'bind': bind()
    elif args.phase == 'merge': pipeline_function('merge')(args.shard_folder or [], args.archive or [None]*4)
    else: pipeline_function('prepare')()
