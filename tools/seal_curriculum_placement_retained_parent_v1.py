"""Seal actual qualified retained3B2500 for the full placement corpus.

This never manufactures a qualification, resumes an optimizer, registers a fit
or admits incomplete placement pools. Real future evidence is mandatory.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import curriculum_placement_parent_v1 as common
from tools import curriculum_placement_retained_parent_v1 as retained
from tools.curriculum_transfer import read, sha, canonical_protocol
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_progression_parent_v1 import extend
from tools.capture_curriculum_paired_terminal_v2 import resource, terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.curriculum_parent import rows
from tools.curriculum_placement_partition_v1 import require_partition
from tools.curriculum_placement_mix_v1 import masses

PARENT = ROOT/'runs/curriculum-yaku-parent-retention-v1/fit-plan/registered.json'
QUEUE = ROOT/'runs/curriculum-progression-parent-retention-v1/yaku-parent-queue-v1-private.json'
PUBLIC = ('tools/seal_curriculum_placement_retained_parent_v1.py',
    'tools/preflight_curriculum_placement_retained_seal_v1.py',
    'tests/core/test_curriculum_placement_retained_seal_v1.py',
    'docs/experiments/plans/curriculum-placement-retained-seal-20261124.md')
ROUTE_PATHS = ('tools/curriculum_placement_full_win_policy_v1.py', 'tools/curriculum_placement_input_v1.py',
    'tools/curriculum_rank_win_input_v1.py', 'tools/curriculum_win_context.py',
    'tools/curriculum_complete_legal_class_policy_v1.py', 'tools/curriculum_riichi_class_policy_v1.py',
    'tools/play_curriculum_placement_v1.py')


def phase_proofs(cap, queue, captured, capture_receipt, receipt_reader):
    """Require real producer commands/exits and all four held queue identities."""
    exits = {}; paths = []
    if set(cap['scheduler_proofs']) != {'prepare', 'fit', 'audit'}:
        raise ValueError('All actual retained-parent phases required')
    for role, proof in cap['scheduler_proofs'].items():
        job = str(queue[role+'_job']); receipt = proof['exit_receipt']
        actual = terminal_record(proof['raw'], job, retained.ENTRY)
        if (not actual['successful'] or any(proof[k] != actual[k] for k in actual)
                or str(receipt['job_id']) != job or receipt['phase'] != role):
            raise ValueError('Original successful phase differs from the held queue')
        path = ROOT/'logs'/f'curriculum-yaku-parent-exit-{job}.json'
        if receipt_reader(path) != receipt: raise ValueError('Original phase exit receipt changed')
        exits[role] = scheduler_exit(receipt, proof['raw'], retained.ENTRY); paths.append(path)
    job = str(queue['capture_job']); path = ROOT/'logs'/f'curriculum-yaku-parent-exit-{job}.json'
    if (captured['exit'] != capture_receipt or str(capture_receipt['job_id']) != job
            or capture_receipt['phase'] != 'capture' or receipt_reader(path) != capture_receipt):
        raise ValueError('Actual own capture exit and held capture identity required')
    actual = terminal_record(captured['raw'], job, retained.ENTRY)
    if not actual['successful']: raise ValueError('Original capture must be genuinely successful')
    exits['capture'] = scheduler_exit(capture_receipt, captured['raw'], retained.ENTRY); paths.append(path)
    return exits, paths


def parent_artifacts(registration):
    registration = registration.resolve()
    if registration != PARENT.resolve(): raise ValueError('Only the fixed retained3B lineage is eligible')
    parent = read(registration); run = registration.parent.parent; fit = ROOT/parent['curriculum']['out']
    done_path = fit/'completion.json'; evidence_path = run/'final-evidence-private.json'
    execution_path = run/'audit-execution-private.json'
    cap_path = run/'yaku-parent-terminal-v1/terminal-evidence-private.json'
    done, evidence, execution, cap = map(read, (done_path, evidence_path, execution_path, cap_path))
    retained.require_parent_evidence(parent, done, evidence, execution, cap)
    queue = read(QUEUE); capture_path = run/'capture-terminal-for-placement-private.json'
    captured = read(capture_path)
    capture_receipt = read(ROOT/'logs'/f"curriculum-yaku-parent-exit-{queue['capture_job']}.json")
    exits, paths = phase_proofs(cap, queue, captured, capture_receipt, read)
    paths.extend((registration, done_path, evidence_path, execution_path, cap_path, QUEUE, capture_path))
    files = dict(parent['inputs'])
    for path in (run/'launch-binding-private.json', cap_path.parent/'completion-binding-private.json'):
        extend(files, path)
    for path in paths:
        key = str(path.resolve()); digest = sha(path)
        if key in files and files[key] != digest: raise ValueError('Conflicting parent evidence identity')
        files[key] = digest
    return parent, evidence, fit, files, exits


def require_data(data, original):
    if (data['status'] != 'placement_expanded_corpus_and_two_epoch_recipe_prepared_not_parent_or_fit_registered'
            or data['optimizer_updates'] != 0 or data['student_fit_registered']
            or not data['originals_preserved_object_exact'] or not data['inherited_split_tags_preserved']
            or not data['validation_file_reused_unchanged'] or data['id_overlap'] or data['group_overlap']
            or data['source_loss_mass'] != {s: float(v) for s, v in masses().items()}
            or set(data['source_unique_records']) != set(masses())
            or any(type(n) is not int or n <= 0 for n in data['source_unique_records'].values())
            or original['status'] != 'placement_all_sources_including_targeted_win_special_prepared_not_fit'):
        raise ValueError('Full57-pool fixed-mass placement corpus and exact exposure preparation required')


def seal(registration, data_path, original_path, out):
    allocation = resource(); out = out.resolve()
    if out.exists(): raise ValueError('Never overwrite a parent seal')
    parent, evidence, fit, files, exits = parent_artifacts(registration)
    pc = parent['curriculum']; data = read(data_path); original = read(original_path)
    require_data(data, original)
    sys.path[:0] = [parent['source'], parent['training_config']['upstream_source']]
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import lineage_training_files, check_groups
    from tools.curriculum_placement_input_v1 import training_record, SCHEMA
    from tools.curriculum_placement_full_win_policy_v1 import METHOD as CHILD_ROUTE
    if package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']:
        raise ValueError('Pinned parent runtime changed')
    for path in (data_path, original_path): extend(files, path.parent/'completion-binding-private.json')
    # A recovered full-data producer must retain its additional lineage proof.
    if 'curriculum-targeted-win-resume-v3' in original_path.parts:
        producer = original_path.parent/'resumed-producer-binding-private.json'; proof = read(producer)
        if not proof['actual_full_original_projection_complete']:
            raise ValueError('Actual complete recovered projection required')
        extend(files, producer)
    membership_path = ROOT/data['current_membership_file']; membership = read(membership_path)
    orders_path = data_path.parent/'expected-orders-private.json'; orders = read(orders_path)
    if orders['seed'] != 2026112306 or [v['order_sha256'] for v in orders['epochs']] != data['expected_order_sha256']:
        raise ValueError('Exact two-epoch prepared order identities required')
    parent_args = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text()); replay = {}
    for split, key, size in (('train', 'data', 91247), ('validation', 'eval_data', 24516)):
        raw_path = ROOT/original['files'][split]['path']; expanded_path = ROOT/data['files'][split]['path']
        if sha(raw_path) != original['files'][split]['sha256'] or sha(expanded_path) != data['files'][split]['sha256']:
            raise ValueError('Actual original or materialized child data changed')
        if split == 'validation' and raw_path.resolve() != expanded_path.resolve():
            raise ValueError('Validation cannot be expanded or substituted')
        current = rows(raw_path); old = rows(parent_args[key]); require_partition(current, membership, split)
        if len(old) != size or len(current) != original['files'][split]['records']:
            raise ValueError('Complete parent replay and actual original partition required')
        if set(Counter(r['questions']['action']['src'] for r in current)) != set(masses()):
            raise ValueError('All57 sources must exist in each actual partition')
        replay[split] = common.replay_projection(old, current, training_record)
        for path in (raw_path, expanded_path): files[str(path.resolve())] = sha(path)
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != parent['training_config']: raise ValueError('Parent protocol differs from actual registration')
    lineage = lineage_training_files(protocol)
    groups = check_groups(parent_args['data'], ROOT/data['files']['train']['path'],
        ROOT/data['files']['validation']['path'], lineage[1:])
    model = bundle(fit/'training-run/checkpoints/step-002500')
    contracts_path = fit/'policy-contracts-private.json'; contract = read(contracts_path)['2500']
    if (model['identity_sha256'] != evidence['final_identity'] or contract['model_identity'] != evidence['final_identity']
            or contract['policy_identity'] != evidence['final_policy_identity']):
        raise ValueError('Actual audited final2500 weights and serving policy required')
    paths = [registration, data_path, original_path, membership_path, orders_path, contracts_path,
        fit/'protocol.json', ROOT/'tools/curriculum_placement_retained_parent_v1.py',
        ROOT/'tools/curriculum_placement_parent_loss_v2.py', ROOT/'tools/curriculum_yaku_parent_loss_v1.py']
    paths += [ROOT/name for name in (*PUBLIC, *ROUTE_PATHS)]
    paths += [Path(model['directory'])/name for name in model['files']]
    for path in paths:
        name = str(path.resolve()); digest = sha(path)
        if name in files and files[name] != digest: raise ValueError('Conflicting immutable parent/data contract')
        files[name] = digest
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Sealed prerequisite changed')
    result = dict(status=common.STATUS, method=common.METHOD, selected_step=2500, model=model,
        parent_loss_method=retained.PARENT_LOSS, child_loss_method=retained.CHILD_LOSS,
        exact_retained_3B_replay_objective=True, original_parent_grade=evidence['stage3B_advancement'],
        parent_registration=str(registration.resolve()), parent_policy=contract['policy'],
        parent_policy_identity=contract['policy_identity'],
        child_policy=dict(method=CHILD_ROUTE, input_schema=SCHEMA, explicit_public_event_required=True,
            runtime_source_sha256=parent['source_sha256'], route_files={n: sha(ROOT/n) for n in ROUTE_PATHS}),
        child_data_ready=str(data_path.resolve()), child_original_data_ready=str(original_path.resolve()),
        child_membership_file=str(membership_path.resolve()), child_replay_projection=replay,
        parent_training_file=parent_args['data'], parent_validation_file=parent_args['eval_data'],
        base=pc['base'], base_revision=pc['base_revision'], bound_files=files, group_proof=groups,
        scheduler_exits=exits, raw_grade_overridden=False, optimizer_resumed=False,
        selection_from_minimum_CE=False, same_execution_policy_as_parent=False,
        parent_games_not_admitted_as_child_step0_cache=True, child_fit_registered=False,
        seal_allocation=allocation)
    out.mkdir(parents=True, exist_ok=False); write(out/'parent-seal-private.json', result)
    print(json.dumps(dict(status=result['status'], selected_step=2500,
        exact_retained_3B_replay_objective=True, child_fit_registered=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--parent-registration', type=Path, default=PARENT)
    p.add_argument('--data-ready', type=Path, required=True); p.add_argument('--original-data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    seal(a.parent_registration.resolve(), a.data_ready.resolve(), a.original_data_ready.resolve(), a.out.resolve())
