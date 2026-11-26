"""Seal the audited calls388 parameters for an explicit execution transfer."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha, canonical_protocol
from tools.curriculum_parent import require_replay, rows
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

METHOD = 'audited_calls388_complete_legal_research_transfer_v1'
STATUS = 'actual_calls388_parameters_and_complete_legal_policy_sealed_for_progression'


def extend(files, path):
    path = Path(path).resolve()
    for name, digest in read(path)['files'].items():
        if name in files and files[name] != digest: raise ValueError('Conflicting bound identity')
        if sha(name) != digest: raise ValueError('Immutable prerequisite changed: '+name)
        files[name] = digest
    files[str(path)] = sha(path)


def seal(config, out):
    gpu_only(); c = read(config)
    if (c['parent_selection'] != 'actual_registered_final388_only'
            or c['admission_mode'] != 'research_execution_transfer_after_successful_original_calls_grade'):
        raise ValueError('Explicit actual-final execution-transfer policy required')
    registration = ROOT/c['parent_registration']; parent = read(registration); pc = parent['curriculum']; fit = ROOT/pc['out']
    evidence_path = ROOT/c['parent_evidence']; evidence = read(evidence_path); completion = read(fit/'completion.json')
    if (parent['registered_final_step'] != 388 or completion['status'] != 'stage2_calls_classroute_fit_and_curves_complete'
            or completion['optimizer_steps'] != 388 or completion['actual_records_seen'] != 98044
            or evidence['status'] != 'stage2_calls_classroute_final_evidence_audit_passed'
            or evidence['stage2_advancement']['passed'] is not True or evidence['saved_final_reloaded_menus'] != 12479):
        raise ValueError('Complete originally qualified calls388 and actual independent audit required')
    sys.path.insert(0, parent['source']); sys.path.insert(0, parent['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.training.initialization import lineage_training_files, check_groups
    if package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']:
        raise ValueError('Pinned runtime changed')
    files = dict(parent['inputs'])
    # Captured real terminal receipts survive scheduler job expiry.
    fit_receipt = read(registration.parent.parent/'fit-terminal-scheduler-private.json')
    exits = dict(fit=scheduler_exit(fit_receipt['exit'], fit_receipt['raw'], registration.parent.parent/'fit.slurm'))
    audit_raw = registration.parent.parent/'audit-final-observed-terminal-private.txt'
    audit_id = dict(t.split('=', 1) for t in audit_raw.read_text().split() if '=' in t)['JobId']
    audit_exit = ROOT/'logs'/f'curriculum-calls388-audit-exit-{audit_id}.json'
    exits['audit'] = scheduler_exit(read(audit_exit), audit_raw.read_text(), registration.parent.parent/'audit.slurm')
    native = ROOT/c['parent_complete_legal_comparison']; native_reg = read(native/'registered-private.json'); native_result = read(native/'summary-private.json')
    native_audit_path = ROOT/c['parent_complete_legal_comparison_audit']; native_audit = read(native_audit_path)
    if (native_result['status'] != 'completed_calls388_same_parameters_complete_legal_route_native64_compared'
            or native_audit['status'] != 'same_calls388_complete_legal64_cached_independent_accounting_audit_passed'
            or native_audit['checked_complete_game_records'] != 128 or native_audit['source_grade_preserved'] != evidence['stage2_advancement']):
        raise ValueError('Actual complete same-parameter route comparison and independent audit required')
    native_root = native.parent; native_raw = native_root/'native64-v2-terminal-scheduler-private.txt'
    native_job = dict(t.split('=', 1) for t in native_raw.read_text().split() if '=' in t)['JobId']
    native_exit = ROOT/'logs'/f'curriculum-complete-legal-native64-v2-exit-{native_job}.json'
    exits['native'] = scheduler_exit(read(native_exit), native_raw.read_text(), ROOT/'cluster/curriculum-complete-legal-class-native-v2.slurm')
    na_raw = native_root/'native64-audit-terminal-scheduler-private.txt'; na_exit = ROOT/'logs'/f"curriculum-complete-legal-native-audit-exit-{native_audit['job_id']}.json"
    exits['native_audit'] = scheduler_exit(read(na_exit), na_raw.read_text(), ROOT/'cluster/curriculum-complete-legal-class-native-audit-v1.slurm')
    for path in (native/'completion-binding-private.json', native_audit_path.parent/'completion-binding-private.json'):
        extend(files, path)
    child_path = ROOT/c['source_data_audit']; data = read(child_path)
    if (data['status'] != 'exact_progression_union_original_rows_encoder_masks_and_split_audit_passed_not_fit_admitted'
            or data['checked_exact_source_row_objects'] != 111283 or not data['original675_reserved_groups_excluded']):
        raise ValueError('Independently audited exact full progression union required')
    extend(files, child_path.parent/'completion-binding-private.json')
    for key in ('objective_preflight', 'fit_adapter_preflight'):
        path = ROOT/c[key]; extend(files, path.parent/'completion-binding-private.json'); files[str(path)] = sha(path)
    model = bundle(fit/'training-run/checkpoints/step-000388')
    if (model != native_reg['parameter_bundle'] or native_result['checkpoint_identity'] != model['identity_sha256']
            or native_audit['parameter_identity'] != model['identity_sha256']
            or native_reg['policy_identity'] != native_audit['complete_legal_policy_identity']):
        raise ValueError('Changed policy must retain the exact actual final388 parameter bundle')
    args = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        path = ROOT/data['files'][split]['path']
        if sha(path) != data['files'][split]['sha256']: raise ValueError('Child data changed')
        require_replay(rows(args[key]), rows(path)); files[str(path.resolve())] = sha(path)
    protocol = canonical_protocol(read(fit/'protocol.json'))
    lineage = lineage_training_files(protocol)
    group_proof = check_groups(args['data'], ROOT/data['files']['train']['path'], ROOT/data['files']['validation']['path'], lineage[1:])
    out.mkdir(parents=True, exist_ok=False); view = out/'native-cache-view'; view.mkdir()
    (view/'matches-private.jsonl').symlink_to((native/'matches-private.jsonl').resolve())
    games = rows(native/'matches-private.jsonl')
    if len(games) != 64: raise ValueError('Full changed-policy parent cache required')
    write(view/'protocol.json', dict(checkpoint=model, policy=native_reg['policy'], policy_identity=native_reg['policy_identity'],
        seeds=list(range(pc['curve_seed_start'], pc['curve_seed_start']+8)), profiles=pc['opponent_profiles'], seats=4, groups=8,
        source_cache=str(native), unchanged_event_metadata_view=True))
    write(view/'summary.json', dict(native_result['proposed_summary'], checkpoint_sha256=model['identity_sha256'],
        policy_identity=native_reg['policy_identity'], selection_method=c['selection_method'],
        curve_grid_subset=aggregate_complete_matches([g for g in games if g['seed'] < pc['curve_seed_start']+2])))
    paths = [Path(__file__), config, registration, evidence_path, fit/'completion.json', fit/'protocol.json',
        fit/'metrics/validation_metrics.jsonl', fit/'metrics/train_metrics.jsonl', fit/'metrics/evaluation_metrics.jsonl',
        registration.parent.parent/'fit-terminal-scheduler-private.json', audit_raw, audit_exit, native_raw, native_exit,
        na_raw, na_exit, child_path, native/'registered-private.json', native/'summary-private.json', native_audit_path]
    paths.extend(view/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'))
    paths.extend(Path(model['directory'])/name for name in model['files'])
    for path in paths: files[str(Path(path).resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Immutable seal prerequisites changed')
    result = dict(status=STATUS, method=METHOD, selected_step=388, model=model, policy=native_reg['policy'],
        policy_identity=native_reg['policy_identity'], original_guarded_policy_identity=native_result['original_policy_identity'],
        original_parent_grade=evidence['stage2_advancement'], changed_route_means=native_result['proposed_summary'],
        original_guarded_means=native_result['original_summary'], changed_route_not_demonstrated_upgrade=True,
        parent_registration=str(registration.resolve()), parent_evidence=str(evidence_path.resolve()),
        parent_training_file=args['data'], parent_validation_file=args['eval_data'], parent_evaluation_cache=str(view.resolve()),
        original_guarded_evaluation_cache=str((fit/'evaluations/step-000388').resolve()),
        child_data_ready=str(child_path.resolve()), base=pc['base'], base_revision=pc['base_revision'],
        bound_files=files, group_proof=group_proof, scheduler_exits=exits,
        raw_grade_overridden=False, selection_from_minimum_CE=False, optimizer_resumed=False, child_fit_started=False)
    write(out/'parent-seal-private.json', result)
    print(json.dumps(dict(status=STATUS, identity=model['identity_sha256'], policy_identity=result['policy_identity'])), flush=True)


def verify_parent(cfg):
    spec = cfg['parent_initialization']; path = Path(spec['manifest']).resolve()
    if cfg.get('initialization') != 'continued' or spec['method'] != METHOD or sha(path) != spec['sha256']:
        raise ValueError('Explicit complete calls388 research-transfer seal required')
    sealed = read(path)
    if (sealed['status'] != STATUS or sealed['method'] != METHOD or sealed['selected_step'] != 388
            or sealed['original_parent_grade']['passed'] is not True or sealed['raw_grade_overridden']
            or sealed['optimizer_resumed'] or sealed['selection_from_minimum_CE']
            or sealed['model']['identity_sha256'] != spec['identity_sha256']
            or sealed['policy_identity'] != spec['policy_identity_sha256']
            or cfg['base'] != sealed['base'] or cfg['base_revision'] != sealed['base_revision']):
        raise ValueError('Sealed parent identity/grade/ancestry differs')
    for p, h in sealed['bound_files'].items():
        if sha(p) != h: raise ValueError('Sealed input changed: '+p)
    parent = read(sealed['parent_registration']); fit = ROOT/parent['curriculum']['out']
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if (Path(spec['source']).resolve() != Path(parent['source']).resolve() or spec['source_sha256'] != parent['source_sha256']
            or package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']
            or bundle(sealed['model']['directory']) != sealed['model'] or Path(cfg['out']).resolve() == fit.resolve()
            or Path(spec['run']).resolve() != fit.resolve() or spec['child_data_ready'] != sealed['child_data_ready']):
        raise ValueError('Pinned complete parent runtime/model differs')
    args = tomllib.loads(Path(cfg['trainer_config']).read_text()); data = read(sealed['child_data_ready'])
    if args.get('resume') or Path(args['init_from']).resolve() != Path(sealed['model']['directory']).resolve():
        raise ValueError('Exact adapter/head continuation with fresh optimizer required')
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        if Path(args[key]).resolve() != (ROOT/data['files'][split]['path']).resolve(): raise ValueError('Child data differs from seal')
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != parent['training_config']: raise ValueError('Parent protocol/config mismatch')
    receipt = dict(sealed['bound_files']); receipt[str(path)] = spec['sha256']
    return dict(model=sealed['model'], receipt=receipt, training_protocol=protocol,
        training_config=parent['training_config'], selected_step=388)


@contextmanager
def parent_loader(initialization):
    old = initialization.verified_parent; initialization.verified_parent = verify_parent
    try: yield
    finally: initialization.verified_parent = old
