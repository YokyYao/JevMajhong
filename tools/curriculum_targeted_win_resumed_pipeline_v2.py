"""Feed completed recovery shards through unchanged full merge/projection code."""
import argparse
import ast
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, terminal_record
from tools.curriculum_targeted_win_resume_v2 import (
    RUN, RESUME, BINDING as RECOVERY_BINDING, require_terminal,
)

ASSEMBLED = RESUME/'assembled-v1'
PREPARED = RESUME/'placement-data-v4/data-v1'
BINDING = RESUME/'pipeline-entry-binding-private.json'
ENTRY = ROOT/'cluster/curriculum-targeted-win-resumed-pipeline-v2.slurm'
SOURCE_SPECS = {
    'merge': ('tools.merge_curriculum_targeted_win_labels_v1', 'merge', {
        'root': ("ROOT/'runs/curriculum-targeted-win-labels-v1'",
                 "ROOT/'runs/curriculum-targeted-win-resume-v2/assembled-v1'")}),
    'prepare': ('tools.prepare_curriculum_placement_data_v3', 'prepare', {
        'control_path': ("ROOT/'runs/curriculum-targeted-win-labels-v1/merged-v1/data-ready-private.json'",
            "ROOT/'runs/curriculum-targeted-win-resume-v2/assembled-v1/merged-v1/data-ready-private.json'"),
        'out': ("ROOT/'runs/curriculum-placement-data-v3/data-v1'",
            "ROOT/'runs/curriculum-targeted-win-resume-v2/placement-data-v4/data-v1'")}),
}
PUBLIC = ('tools/curriculum_targeted_win_resumed_pipeline_v2.py',
    'tests/core/test_curriculum_targeted_win_resumed_pipeline_v2.py',
    'cluster/curriculum-targeted-win-resumed-pipeline-v2.slurm',
    'docs/experiments/plans/curriculum-targeted-win-resumed-pipeline-20261124.md')


def adapted_function(source, name, replacements):
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    result = copy.deepcopy(original); changed = set()
    for node in ast.walk(result):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key not in replacements: continue
            before, after = replacements[key]
            if key in changed or ast.dump(node.value) != ast.dump(ast.parse(before, mode='eval').body):
                raise ValueError('Expected one exact original path assignment: '+key)
            node.value = ast.parse(after, mode='eval').body; changed.add(key)
    if changed != set(replacements): raise ValueError('All exact path substitutions must be present')
    # Round-trip the substitutions to prove no other calculation changed.
    restored = copy.deepcopy(result)
    for node in ast.walk(restored):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in replacements):
            node.value = ast.parse(replacements[node.targets[0].id][0], mode='eval').body
    if ast.dump(restored) != ast.dump(original): raise ValueError('Original verification/projection logic changed')
    return ast.fix_missing_locations(ast.Module(body=[result], type_ignores=[]))


def original_source(phase):
    module, name, replacements = SOURCE_SPECS[phase]
    path = ROOT/(module.replace('.', '/')+'.py')
    return module, name, path, adapted_function(path.read_text(), name, replacements)


def verify_complete(folder, shard, reg):
    if not (folder/'summary-private.json').is_file():
        raise ValueError('A full27-view completed shard is required; never merge a live partial writer')
    summary = read(folder/'summary-private.json'); launch = read(folder/'registered-private.json')
    expected = reg['selected'][shard*27:(shard+1)*27]
    if (summary['status'] != 'fixed_targeted_win_label_shard_complete_candidate_targets_not_fit_admitted'
            or summary['selected_views'] != 27 or launch['selected'] != expected
            or launch['parent_registration_sha256'] != sha(RUN/'registered-private.json')
            or summary['teacher_model'] != reg['continuation_model']['identity_sha256']
            or launch['model'] != reg['continuation_model']
            or str(summary['job_id']) != str(launch['job_id']) or summary['node'] != launch['node']
            or summary['counts'].get('labelled_views', 0)+summary['counts'].get('sampling_quarantine', 0) != 27):
        raise ValueError('Exact complete original shard population/teacher/attempt required')
    files = {}
    for name, digest in summary['files'].items():
        path = folder/name
        if sha(path) != digest: raise ValueError('Completed shard output changed')
        files[str(path.resolve())] = digest
    for name in ('summary-private.json', 'registered-private.json'):
        files[str((folder/name).resolve())] = sha(folder/name)
    return files


def verify_entry():
    binding = read(BINDING)
    if (binding['status'] != 'resumed_targeted_merge_and_projection_bound_not_executed'
            or any(sha(p) != h for p, h in binding['files'].items())):
        raise ValueError('Unchanged before-data-completion pipeline binding required')
    reg = read(RUN/'registered-private.json')
    source = read(reg['source_registration'])
    sys.path[:0] = [source['source'], source['training_config']['upstream_source']]
    return binding, reg


def run_original(phase):
    module, name, path, tree = original_source(phase)
    scope = dict(vars(importlib.import_module(module)))
    exec(compile(tree, str(path), 'exec'), scope)
    scope[name]()


def bind():
    allocation = resource(); recovery = read(RECOVERY_BINDING)
    if recovery['status'] != 'targeted_win_resume_entry_bound_not_started':
        raise ValueError('The completed recovery preflight must precede this pipeline')
    if BINDING.exists(): raise ValueError('Never overwrite a pipeline binding')
    log_path = ROOT/'logs/curriculum-targeted-win-resumed-pipeline-tests-v2.log'
    log = log_path.read_text()
    if 'Ran 6 tests' not in log or not log.endswith('\nOK\nexit_code=0\n'):
        raise ValueError('Six exact-path/full-shard/terminal-lineage tests must pass first')
    files = dict(recovery['files']); files[str(RECOVERY_BINDING)] = sha(RECOVERY_BINDING)
    for phase in SOURCE_SPECS:
        _, _, path, tree = original_source(phase)
        compile(tree, str(path), 'exec'); files[str(path)] = sha(path)
    for path in (*[ROOT/p for p in PUBLIC], log_path): files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('A frozen source changed')
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=10)
    reg = read(RUN/'registered-private.json')
    # A live actual partial shard must not be mistaken for full readiness.
    try: verify_complete(RUN/'shards/shard-00', 0, reg)
    except ValueError: incomplete_rejected = True
    else: raise ValueError('This before-completion preflight requires the original partial shard')
    write(BINDING, dict(status='resumed_targeted_merge_and_projection_bound_not_executed',
        files=files, allocation=allocation, unit_tests_passed=6,
        actual_incomplete_shard_rejected=incomplete_rejected, original_functions_changed_only_at_paths=True,
        full108_merge_and_model_input_projection_not_yet_run=True,
        no_jobs_submitted=True, student_updates=0))
    print(json.dumps(dict(status='resumed_targeted_pipeline_bound_not_executed', unit_tests_passed=6,
        actual_incomplete_shard_rejected=True, student_updates=0)), flush=True)


def verify_recovery_lineage(folder):
    complete = read(folder/'resumption-complete-private.json')
    if (complete['binding_sha256'] != sha(RECOVERY_BINDING)
            or not complete['original_case_loop_unchanged'] or complete['partial_worlds_admitted']
            or complete['exact_recovered_views']+complete['newly_processed_views'] != 27):
        raise ValueError('Actual unchanged complete recovery attempt required')
    path = Path(complete['admission']); admission = read(path)
    if (admission['status'] != 'actual_terminal_complete_cases_preserved_missing_cases_only'
            or not admission['same_fixed_population_worlds_teacher_and_targets']
            or admission['completed_views'] != complete['exact_recovered_views']
            or any(sha(p) != h for p, h in admission['files'].items())):
        raise ValueError('Original recovered cases or admission changed')
    for proof in admission['terminals']:
        receipt = read(proof['receipt_path'])
        if receipt != proof['exit_receipt']: raise ValueError('Original prior exit changed')
        command = ROOT/('cluster/curriculum-targeted-win-labels-v1.slurm'
            if receipt.get('phase') == 'label' else 'cluster/curriculum-targeted-win-resume-v2.slurm')
        actual = terminal_record(proof['raw'], proof['job_id'], command)
        if any(proof[k] != actual[k] for k in actual): raise ValueError('Original prior terminal differs')
    return {str(path.resolve()): sha(path), str((folder/'resumption-complete-private.json').resolve()): sha(folder/'resumption-complete-private.json'),
        **admission['files'], **{p['receipt_path']: sha(p['receipt_path']) for p in admission['terminals']}}


def merge(folders, archives):
    allocation = resource(); binding, reg = verify_entry()
    folders = [p.resolve() for p in folders]
    if len(folders) != 4 or len(set(folders)) != 4 or len(archives) != 4:
        raise ValueError('Exactly four distinct complete shard folders in original shard order required')
    files = dict(binding['files']); terminals = []
    for shard, (folder, archive) in enumerate(zip(folders, archives)):
        files.update(verify_complete(folder, shard, reg))
        terminals.append(require_terminal(folder, archive))
        if folder != (RUN/'shards'/f'shard-{shard:02d}').resolve():
            files.update(verify_recovery_lineage(folder))
    ASSEMBLED.mkdir(exist_ok=False)
    (ASSEMBLED/'registered-private.json').symlink_to(RUN/'registered-private.json')
    (ASSEMBLED/'shards').mkdir()
    for shard, folder in enumerate(folders): (ASSEMBLED/'shards'/f'shard-{shard:02d}').symlink_to(folder, target_is_directory=True)
    admission = ASSEMBLED/'merge-admission-private.json'
    write(admission, dict(files=files, terminals=terminals, allocation=allocation, selected_views=108,
        original_merge_function_changed_only_at_root=True, student_updates=0))
    run_original('merge')
    out = ASSEMBLED/'merged-v1'
    for path in (admission, BINDING, out/'completion-binding-private.json', out/'data-ready-private.json'):
        files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('An original source changed during full merge')
    write(out/'resumed-producer-binding-private.json', dict(files=files,
        actual_full108_original_merge_complete=True, student_updates=0))


def prepare():
    allocation = resource(); binding, reg = verify_entry()
    prerequisite = ASSEMBLED/'merged-v1/resumed-producer-binding-private.json'
    proof = read(prerequisite)
    if not proof['actual_full108_original_merge_complete'] or any(sha(p) != h for p, h in proof['files'].items()):
        raise ValueError('Actual complete original108 merge and unchanged producer required')
    run_original('prepare')
    files = {**binding['files'], **proof['files'], str(prerequisite): sha(prerequisite), str(BINDING): sha(BINDING)}
    for path in (PREPARED/'data-ready-private.json', PREPARED/'completion-binding-private.json'):
        files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('An original input changed during projection')
    write(PREPARED/'resumed-producer-binding-private.json', dict(files=files, allocation=allocation,
        actual_full_original_projection_complete=True, loss_mix_and_qualified_parent_fit_still_required=True,
        student_updates=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=['bind', 'merge', 'prepare'], required=True)
    p.add_argument('--shard-folder', type=Path, action='append'); p.add_argument('--archive', type=Path, action='append')
    a = p.parse_args()
    if a.phase == 'merge': merge(a.shard_folder or [], a.archive or [None]*4)
    else: globals()[a.phase]()
