"""Continue qualified final3B tensors under the declared placement contract.

Only the new optimizer and public-input/execution contract change. A parent
seal is not a student fit, a minimum-CE selection, or permission to release.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha, canonical_protocol
from tools.curriculum_parent import rows
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_progression_parent_v1 import extend
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.capture_curriculum_yaku_terminal_v1 import accounting_proof
from tools.curriculum_placement_partition_v1 import require_partition
from tools.curriculum_placement_mix_v1 import NEW_SOURCES

METHOD = 'audited_final_yaku2500_new_placement_contract_new_optimizer_v1'
STATUS = 'actual_qualified_yaku2500_sealed_for_placement_not_fit_registered'


def require_final(parent, done, evidence, execution, terminal):
    if (parent['registered_final_step'] != 2500 or done['optimizer_steps'] != 2500
            or done['status'] != 'yaku2500_fit_and_real_curves_complete'
            or done['actual_records_seen'] != 182494
            or evidence['status'] != 'yaku2500_final_evidence_audit_passed'
            or evidence['stage3B_advancement']['passed'] is not True
            or evidence['saved_final_reloaded_menus'] != 24516
            or not evidence['source_and_data_hashes_unchanged']
            or not evidence['all376_optimizer_states_match']
            or evidence['final_reserved_grid_unused'] is not True
            or execution['status'] != 'actual_yaku2500_full_evidence_independently_audited'
            or not execution['original_launch_binding_unchanged']
            or execution['actual_fit_job'] != terminal['parent_fit_job']
            or terminal['status'] != 'actual_final3B_fit_and_audit_terminal_evidence_preserved_not_child_admitted'
            or terminal['selected_step'] != 2500 or terminal['strategy_grade_overridden']
            or terminal['original_strategy_grade'] != evidence['stage3B_advancement']
            or terminal['final_identity'] != evidence['final_identity']
            or terminal['final_policy_identity'] != evidence['final_policy_identity']):
        raise ValueError('Original-qualified actual final2500, independent audit and real terminal evidence required')


def replay_projection(parent_rows, child_rows, project):
    current = {r['_meta']['id']: r for r in child_rows}
    if len(current) != len(child_rows):
        raise ValueError('Distinct original child records required')
    seen = set(); changed = 0
    for row in parent_rows:
        rid = row['_meta']['id']; before = copy.deepcopy(row)
        if rid in seen or rid not in current:
            raise ValueError('Every distinct original parent replay row required')
        seen.add(rid); expected = project(row)
        if (row != before or expected['state'] != before['state'] or expected['_meta'] != before['_meta']
                or list(expected['questions']) != list(before['questions'])):
            raise ValueError('Declared placement projection cannot change state, metadata or mutate parent')
        for qid, original in before['questions'].items():
            projected = expected['questions'][qid]
            restored = dict(projected, instructions=original['instructions'])
            if (restored != original or list(projected['criteria']) != list(original['criteria'])):
                raise ValueError('Only the declared instruction may differ; labels/menu/order must remain exact')
            changed += projected['instructions'] != original['instructions']
        if expected != current[rid]:
            raise ValueError('Actual child replay differs from the pinned placement projection')
    if any(r['questions']['action']['src'] not in NEW_SOURCES for rid, r in current.items() if rid not in seen):
        raise ValueError('Additional originals must belong to the declared new placement pools')
    return dict(parent_records=len(seen), projected_instruction_changes=changed,
        parent_states_labels_targets_metadata_and_menu_order_preserved=True,
        whole_parent_prediction_parity_not_claimed=True)


def seal(config):
    gpu_only(); c = read(config)
    if (c['parent_selection'] != 'actual_registered_final2500_only'
            or c['admission_mode'] != 'original_stage3B_grade_pass_required' or c['automatic_fit']):
        raise ValueError('Only the original-qualified registered final3B is eligible')
    out = ROOT/c['out']
    if out.exists(): raise ValueError('Never overwrite a parent seal')
    registered = ROOT/c['parent_registration']; parent = read(registered); pc = parent['curriculum']
    run = registered.parent.parent; fit = ROOT/pc['out']
    evidence_path = run/'final-evidence-private.json'; execution_path = run/'audit-execution-private.json'
    done_path = fit/'completion.json'; terminal_path = ROOT/c['parent_terminal']
    evidence = read(evidence_path); done = read(done_path); execution = read(execution_path); terminal = read(terminal_path)
    require_final(parent, done, evidence, execution, terminal)
    if (terminal['parent_registration'] != str(registered.resolve())
            or terminal['parent_fit_job'] != c['parent_fit_job']
            or terminal['parent_audit_job'] != c['parent_audit_job']):
        raise ValueError('Terminal evidence belongs to another parent run')
    sys.path.insert(0, parent['source']); sys.path.insert(0, parent['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import lineage_training_files, check_groups
    from tools.curriculum_placement_input_v1 import training_record, SCHEMA
    from tools.curriculum_placement_full_win_policy_v1 import METHOD as CHILD_ROUTE
    if package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']:
        raise ValueError('Pinned parent runtime changed')
    files = dict(parent['inputs'])
    for path in (run/'launch-binding-private.json', terminal_path.parent/'completion-binding-private.json'):
        extend(files, path)
    script = ROOT/'cluster/curriculum-yaku-course-v3.slurm'; exits = {}
    if set(terminal['scheduler_proofs']) != {'fit', 'audit'}:
        raise ValueError('Both actual terminal phases required')
    for role, proof in terminal['scheduler_proofs'].items():
        receipt = proof['receipt']; job = c['parent_'+role+'_job']
        if str(receipt['job_id']) != str(job) or receipt['phase'] != role:
            raise ValueError('Wrong terminal role or job')
        if proof['source'] == 'sacct_original_row':
            exits[role] = accounting_proof(proof['raw'], receipt, job, script)
        elif proof['source'] in ('actual_terminal_scontrol', 'original_independent_audit_scontrol_archive'):
            exits[role] = scheduler_exit(receipt, proof['raw'], script)
        else:
            raise ValueError('Unrecognized terminal evidence source')
    data_path = ROOT/c['child_data_ready']; data = read(data_path)
    original_path = ROOT/c['original_child_data_ready']; original = read(original_path)
    if (data['status'] != 'placement_expanded_corpus_and_two_epoch_recipe_prepared_not_parent_or_fit_registered'
            or data['optimizer_updates'] != 0 or data['student_fit_registered']
            or not data['originals_preserved_object_exact'] or not data['inherited_split_tags_preserved']
            or not data['validation_file_reused_unchanged'] or data['id_overlap'] or data['group_overlap']
            or original['status'] != 'placement_all_sources_including_targeted_win_special_prepared_not_fit'):
        raise ValueError('Actual complete placement corpus and explicit exposure preparation required')
    extend(files, data_path.parent/'completion-binding-private.json')
    extend(files, original_path.parent/'completion-binding-private.json')
    membership_path = ROOT/data['current_membership_file']; manifest = read(membership_path)
    orders = data_path.parent/'expected-orders-private.json'
    if [r['order_sha256'] for r in read(orders)['epochs']] != data['expected_order_sha256']:
        raise ValueError('Prepared two-epoch order identities changed')
    parent_args = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    replay = {}
    for split, key, size in (('train', 'data', 91247), ('validation', 'eval_data', 24516)):
        raw_path = ROOT/original['files'][split]['path']
        actual_path = ROOT/data['files'][split]['path']
        if (sha(raw_path) != original['files'][split]['sha256']
                or sha(actual_path) != data['files'][split]['sha256']):
            raise ValueError('Placement original or expanded file changed')
        if split == 'validation' and actual_path.resolve() != raw_path.resolve():
            raise ValueError('Validation must reuse the original current file without expansion')
        raw_rows = rows(raw_path); old_rows = rows(parent_args[key]); require_partition(raw_rows, manifest, split)
        if len(old_rows) != size:
            raise ValueError('Exact complete3B replay required')
        replay[split] = replay_projection(old_rows, raw_rows, training_record)
        files[str(raw_path.resolve())] = sha(raw_path); files[str(actual_path.resolve())] = sha(actual_path)
    protocol = canonical_protocol(read(fit/'protocol.json')); lineage = lineage_training_files(protocol)
    groups = check_groups(parent_args['data'], ROOT/data['files']['train']['path'],
        ROOT/data['files']['validation']['path'], lineage[1:])
    model = bundle(fit/'training-run/checkpoints/step-002500')
    contracts_path = fit/'policy-contracts-private.json'; contract = read(contracts_path)['2500']
    if (model['identity_sha256'] != evidence['final_identity']
            or contract['model_identity'] != model['identity_sha256']
            or contract['policy_identity'] != evidence['final_policy_identity']):
        raise ValueError('Actual audited final3B weights/policy differ')
    route_paths = ('tools/curriculum_placement_full_win_policy_v1.py', 'tools/curriculum_placement_input_v1.py',
        'tools/curriculum_rank_win_input_v1.py', 'tools/curriculum_win_context.py',
        'tools/curriculum_complete_legal_class_policy_v1.py', 'tools/curriculum_riichi_class_policy_v1.py',
        'tools/play_curriculum_placement_v1.py')
    paths = [Path(__file__), config, registered, evidence_path, execution_path, done_path, terminal_path,
        membership_path, orders, data_path, original_path, contracts_path, fit/'protocol.json',
        ROOT/'cluster/curriculum-placement-parent-v1.slurm', ROOT/'configs/training/curriculum-placement-parent-v1.json',
        ROOT/'tests/core/test_curriculum_placement_parent_v1.py',
        ROOT/'docs/experiments/plans/curriculum-placement-parent-20261123.md']
    paths += [Path(model['directory'])/name for name in model['files']]
    paths += [ROOT/name for name in route_paths]
    for path in paths:
        name = str(Path(path).resolve()); digest = sha(path)
        if name in files and files[name] != digest:
            raise ValueError('Conflicting immutable parent/placement contract')
        files[name] = digest
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError('Sealed input changed')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status=STATUS, method=METHOD, selected_step=2500, model=model,
        original_parent_grade=evidence['stage3B_advancement'], parent_registration=str(registered.resolve()),
        parent_policy=contract['policy'], parent_policy_identity=contract['policy_identity'],
        child_policy=dict(method=CHILD_ROUTE, input_schema=SCHEMA, explicit_public_event_required=True,
            runtime_source_sha256=parent['source_sha256'], route_files={n: sha(ROOT/n) for n in route_paths}),
        child_data_ready=str(data_path.resolve()), child_original_data_ready=str(original_path.resolve()),
        child_membership_file=str(membership_path.resolve()), child_replay_projection=replay,
        parent_training_file=parent_args['data'], parent_validation_file=parent_args['eval_data'],
        base=pc['base'], base_revision=pc['base_revision'], bound_files=files, group_proof=groups,
        scheduler_exits=exits, raw_grade_overridden=False, optimizer_resumed=False,
        selection_from_minimum_CE=False, same_execution_policy_as_parent=False,
        parent_games_not_admitted_as_child_step0_cache=True, child_fit_registered=False)
    write(out/'parent-seal-private.json', result)
    print(json.dumps(dict(status=STATUS, selected_step=2500, original_grade_passed=True,
        replay_projection=replay, child_fit_registered=False)), flush=True)


def verify_parent(cfg):
    spec = cfg['parent_initialization']; path = Path(spec['manifest']).resolve(); sealed = read(path)
    if (cfg.get('initialization') != 'continued' or spec['method'] != METHOD
            or sha(path) != spec['sha256'] or sealed['method'] != METHOD or sealed['status'] != STATUS
            or spec['selected_step'] != 2500 or sealed['selected_step'] != 2500
            or sealed['original_parent_grade']['passed'] is not True or sealed['raw_grade_overridden']
            or sealed['optimizer_resumed'] or sealed['selection_from_minimum_CE']
            or sealed['same_execution_policy_as_parent'] or not sealed['parent_games_not_admitted_as_child_step0_cache']
            or sealed['model']['identity_sha256'] != spec['identity_sha256']
            or sealed['parent_policy_identity'] != spec['policy_identity_sha256']
            or cfg['base'] != sealed['base'] or cfg['base_revision'] != sealed['base_revision']):
        raise ValueError('Exact qualified final3B and declared changed placement policy required')
    for name, digest in sealed['bound_files'].items():
        if sha(name) != digest: raise ValueError('Sealed input changed: '+name)
    parent = read(sealed['parent_registration']); fit = (ROOT/parent['curriculum']['out']).resolve()
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if (Path(spec['source']).resolve() != Path(parent['source']).resolve()
            or spec['source_sha256'] != parent['source_sha256']
            or cfg['base_files'] != parent['training_config']['base_files']
            or Path(cfg['base_load_path']).resolve() != Path(parent['training_config']['base_load_path']).resolve()
            or Path(cfg['upstream_source']).resolve() != Path(parent['training_config']['upstream_source']).resolve()
            or cfg['upstream_source_sha256'] != parent['training_config']['upstream_source_sha256']
            or package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']
            or bundle(sealed['model']['directory']) != sealed['model']
            or Path(cfg['out']).resolve() == fit or fit in Path(cfg['out']).resolve().parents
            or Path(spec['run']).resolve() != fit or spec['child_data_ready'] != sealed['child_data_ready']):
        raise ValueError('Pinned runtime/model, isolated child output and sealed data required')
    args = tomllib.loads(Path(cfg['trainer_config']).read_text()); data = read(sealed['child_data_ready'])
    if args.get('resume') or Path(args['init_from']).resolve() != Path(sealed['model']['directory']).resolve():
        raise ValueError('Same adapter/head with a new optimizer required')
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        if Path(args[key]).resolve() != (ROOT/data['files'][split]['path']).resolve():
            raise ValueError('Training file differs from the sealed exposure corpus')
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != parent['training_config']:
        raise ValueError('Original parent protocol differs')
    receipt = dict(sealed['bound_files']); receipt[str(path)] = spec['sha256']
    return dict(model=sealed['model'], receipt=receipt, training_protocol=protocol,
        training_config=parent['training_config'], selected_step=2500)


@contextmanager
def parent_loader(initialization):
    original = initialization.verified_parent; initialization.verified_parent = verify_parent
    try: yield
    finally: initialization.verified_parent = original


if __name__ == '__main__':
    seal(ROOT/'configs/training/curriculum-placement-parent-v1.json')
