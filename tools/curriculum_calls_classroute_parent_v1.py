"""Seal/verify a distinct fixed-RI-policy parent, retaining its raw failure."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha, canonical_protocol
from tools.curriculum_parent import ParentNotReady, require_final_integrity, require_replay, rows
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write, AUDITED
from tools.curriculum_riichi_policy_profile_v1 import profile
from tools.curriculum_riichi_class_policy_v1 import METHOD as ROUTE
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

METHOD = 'curriculum_final_distinct_RI_class_policy_v1'
STATUS = 'completed_basic_distinct_RI_policy_parent_sealed_v1'
PARENT = ROOT/'runs/curriculum-riichi-class-control-v1'
CHILD = ROOT/'runs/curriculum-calls-post-reach-data-v1'
NATIVE = ROOT/'runs/curriculum-riichi-class-route-v2'


def require_policy_grade(raw, selected, native):
    if (raw.get('all_source_quality_passed') is not False
            or selected.get('all_source_quality_passed') is not True
            or native.get('status') != 'fixed_RI_route_native64_independent_cached_audit_passed'
            or native.get('raw_parameters_unchanged') is not True
            or native.get('source_grade_not_overridden') is not True
            or native.get('baseline16_exact') is not True
            or native.get('final_reserved_grid_unused') is not True
            or native.get('checked_complete_match_ledgers') != 192
            or native.get('parent_admitted') is not False
            or native.get('new_optimizer_updates') != 0
            or native.get('policy', {}).get('method') != ROUTE):
        raise ParentNotReady('Distinct actually qualified RI policy and preserved raw failure required')


def extend(files, binding):
    for name, digest in read(binding)['files'].items():
        if name in files and files[name] != digest: raise ParentNotReady('Conflicting lineage binding')
        if sha(name) != digest: raise ParentNotReady('Immutable lineage changed: '+name)
        files[name] = digest
    files[str(Path(binding).resolve())] = sha(binding)


def seal(out):
    gpu_only(); out.mkdir(parents=True, exist_ok=False)
    registration = PARENT/'fit-plan/registered.json'; reg = read(registration)
    fit = ROOT/reg['curriculum']['out']; evidence_path = PARENT/'final-evidence-private.json'
    evidence = read(evidence_path); validation = rows(fit/'metrics/validation_metrics.jsonl')[-1]
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest, target_cross_entropy
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from tools.curriculum_riichi_control import require_registered
    require_registered(reg)
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ParentNotReady('Pinned runtime changed')
    model = bundle(fit/'training-run/checkpoints/step-000608')
    require_final_integrity(reg, read(fit/'completion.json'), evidence, validation, model['identity_sha256'])
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    parent_rows = rows(args['eval_data'])
    prediction_file = fit/'training-run/training_eval/step-000608/development/rows.json'
    measured = profile(parent_rows, read(prediction_file), {'weighted_target_ce': validation['natural_loss']}, target_cross_entropy)
    prior_path = ROOT/'runs/curriculum-riichi-policy-profile-v1/audit-v1/summary-private.json'
    prior = read(prior_path); native_path = NATIVE/'audit-v1/summary-private.json'; native = read(native_path)
    require_policy_grade(measured['raw_source_quality'], measured['selected_source_quality'], native)
    if (prior['raw_source_quality'] != measured['raw_source_quality']
            or prior['selected_source_quality'] != measured['selected_source_quality']
            or prior['parameter_identity'] != model['identity_sha256']
            or prior['policy_identity'] != native['policy_identity']):
        raise ParentNotReady('Distinct policy source qualification differs')
    files = dict(reg['inputs'])
    for binding in (CHILD/'completion-binding-private.json', NATIVE/'audit-v1/binding-private.json'):
        extend(files, binding)
    exits = {}
    for role in ('fit', 'audit'):
        path = PARENT/f'{role}-terminal-scheduler-private.json'; receipt = read(path)
        exits['parent_'+role] = scheduler_exit(receipt['exit'], receipt['raw'], PARENT/(role+'.slurm'))
        files[str(path)] = sha(path)
    trial_receipt_path = NATIVE/'trial-terminal-scheduler-private.json'; trial_receipt = read(trial_receipt_path)
    exits['route_trial'] = scheduler_exit(trial_receipt['exit'], trial_receipt['raw'], NATIVE/'evaluate.slurm')
    audit_raw = NATIVE/'audit-terminal-scheduler-private.txt'
    audit_exit = ROOT/'logs/curriculum-RI-class-route-v2-audit-exit-316514.json'
    exits['route_audit'] = scheduler_exit(read(audit_exit), audit_raw.read_text(), NATIVE/'audit.slurm')
    archive_raw = CHILD/'archive-terminal-scheduler-private.txt'
    archive_exit = ROOT/'logs/curriculum-calls13-archive-exit-316741.json'
    exits['prerequisites_archive'] = scheduler_exit(read(archive_exit), archive_raw.read_text(), CHILD/'archive.slurm')
    data_path = CHILD/'data/data-ready-private.json'; data_audit_path = CHILD/'audit-v1/summary-private.json'
    data = read(data_path); data_audit = read(data_audit_path)
    if data_audit['status'] != AUDITED or data_audit['files'] != data['files']:
        raise ParentNotReady('Independent thirteen-source replay audit required')
    for split, parent_file in (('train', args['data']), ('validation', args['eval_data'])):
        require_replay(rows(parent_file), rows(ROOT/data['files'][split]['path']))
    train_groups = {r['_meta']['group_id'] for r in rows(ROOT/data['files']['train']['path'])}
    held = {r['_meta']['group_id'] for r in rows(ROOT/data['files']['validation']['path'])}
    reserved = ROOT/reg['curriculum']['data_root']/'reserved-public-groups.json'
    if train_groups & (held | set(read(reserved))): raise ParentNotReady('Child held-out/reserved overlap')
    # Metadata view only: the real64 candidate event records are linked unchanged.
    source = NATIVE/'trial-v1'; games = rows(source/'matches-private.jsonl')
    protocol = read(source/'protocol.json'); summary = read(source/'summary.json')
    actual = aggregate_complete_matches(games)
    if (len(games) != 64 or summary['policy_identity'] != native['policy_identity']
            or protocol['policy'] != native['policy'] or any(summary[k] != v for k, v in actual.items())):
        raise ParentNotReady('Actual parent-policy cache differs')
    expected_grid = {(seed, p, seat) for seed in protocol['seeds'] for p in protocol['profiles'] for seat in range(4)}
    if (len(expected_grid) != 64 or {(r['seed'], r['profile'], r['seat']) for r in games} != expected_grid
            or protocol['seeds'] != list(range(reg['curriculum']['curve_seed_start'], reg['curriculum']['curve_seed_start']+8))):
        raise ParentNotReady('Full consumed-development parent cache required')
    view = out/'native-cache-view'; view.mkdir()
    (view/'matches-private.jsonl').symlink_to((source/'matches-private.jsonl').resolve())
    write(view/'protocol.json', dict(checkpoint=model, policy=native['policy'], policy_identity=native['policy_identity'],
        seeds=protocol['seeds'], profiles=protocol['profiles'], seats=4, groups=8,
        source_cache=str(source), unchanged_event_metadata_view=True))
    write(view/'summary.json', dict(summary, checkpoint_sha256=model['identity_sha256'],
        curve_grid_subset=aggregate_complete_matches([r for r in games if r['seed'] < protocol['seeds'][0]+2]),
        unchanged_event_metadata_view=True))
    paths = [registration, evidence_path, prior_path, native_path, data_path, data_audit_path,
        prediction_file, reserved, trial_receipt_path, audit_raw, audit_exit, archive_raw, archive_exit,
        fit/'completion.json', fit/'protocol.json', fit/'metrics/validation_metrics.jsonl',
        fit/'metrics/train_metrics.jsonl', fit/'metrics/evaluation_metrics.jsonl',
        fit/'initialization-audit.json', fit/'actual-update-proof.json', Path(__file__)]
    paths.extend(source/name for name in ('matches-private.jsonl', 'summary.json', 'protocol.json'))
    paths.extend(view/name for name in ('matches-private.jsonl', 'summary.json', 'protocol.json'))
    paths.extend(Path(model['directory'])/name for name in model['files'])
    paths.extend(ROOT/spec['path'] for spec in data['files'].values())
    for path in paths:
        files[str(path.absolute())] = sha(path)
        files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()): raise ParentNotReady('Frozen parent/child inputs changed')
    result = dict(status=STATUS, method=METHOD, model=model, selected_step=608,
        parent_registration=str(registration), parent_evidence=str(evidence_path),
        child_data_ready=str(data_path), child_data_audit=str(data_audit_path), bound_files=files,
        stage_quality=measured['selected_source_quality'], original_raw_source_quality=measured['raw_source_quality'],
        original_stage1_advancement=evidence['stage1_advancement'], policy=native['policy'],
        policy_identity=native['policy_identity'], parent_evaluation_cache=str(view.resolve()),
        parent_training_file=args['data'], parent_validation_file=args['eval_data'],
        base=reg['curriculum']['base'], base_revision=reg['curriculum']['base_revision'],
        selection_from_minimum_CE=False, optimizer_resumed=False, raw_grade_overridden=False,
        research_continuation_not_stable_gain_or_v8_parity=True, scheduler_exits=exits,
        child_fit_started=False, mandatory_parent_replay_verified=True)
    write(out/'parent-seal-private.json', result)
    print(json.dumps(dict(status=STATUS, parameter_identity=model['identity_sha256'],
                         policy_identity=native['policy_identity'])), flush=True)
    return result


def verify_parent(cfg):
    spec = cfg['parent_initialization']; seal_path = Path(spec['manifest']).resolve()
    if cfg.get('initialization') != 'continued' or spec['method'] != METHOD or sha(seal_path) != spec['sha256']:
        raise ParentNotReady('Explicit distinct-policy continuation seal required')
    sealed = read(seal_path)
    if (sealed['status'] != STATUS or sealed['method'] != METHOD or sealed['raw_grade_overridden'] is not False
            or sealed['selection_from_minimum_CE'] or sealed['optimizer_resumed']
            or not sealed['stage_quality']['all_source_quality_passed']
            or sealed['original_raw_source_quality']['all_source_quality_passed']
            or sealed['policy_identity'] != spec['policy_identity_sha256']
            or sealed['selected_step'] != spec['selected_step'] or spec['selected_step'] != 608
            or sealed['model']['identity_sha256'] != spec['identity_sha256']
            or sealed['base'] != cfg['base'] or sealed['base_revision'] != cfg['base_revision']):
        raise ParentNotReady('Actual distinct parent policy identity/quality differs')
    for p, h in sealed['bound_files'].items():
        if sha(p) != h: raise ParentNotReady('Sealed source/model/data changed: '+p)
    reg = read(sealed['parent_registration']); fit = ROOT/reg['curriculum']['out']
    if (Path(spec['source']).resolve() != Path(reg['source']).resolve() or spec['source_sha256'] != reg['source_sha256']
            or Path(spec['run']).resolve() != fit.resolve() or fit.resolve() == (ROOT/cfg['out']).resolve()
            or str(Path(spec['child_data_ready']).resolve()) != sealed['child_data_ready']):
        raise ParentNotReady('Parent runtime/run/child input differs')
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256'] or bundle(sealed['model']['directory']) != sealed['model']:
        raise ParentNotReady('Pinned parent source or parameter bundle changed')
    args = tomllib.loads(Path(cfg['trainer_config']).read_text()); data = read(sealed['child_data_ready'])
    if args.get('resume') or Path(args['init_from']).resolve() != Path(sealed['model']['directory']).resolve():
        raise ParentNotReady('Same complete adapter/head with new optimizer required')
    for split, keyname in (('train', 'data'), ('validation', 'eval_data')):
        if Path(args[keyname]).resolve() != (ROOT/data['files'][split]['path']).resolve():
            raise ParentNotReady('Trainer child dataset differs from seal')
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != reg['training_config'] or protocol['trainer_args']['data'] != sealed['parent_training_file']:
        raise ParentNotReady('Actual parent training protocol differs')
    receipt = dict(sealed['bound_files']); receipt[str(seal_path)] = spec['sha256']
    return dict(model=sealed['model'], receipt=receipt, training_protocol=protocol,
                training_config=reg['training_config'], selected_step=608)


@contextmanager
def parent_loader(initialization):
    old = initialization.verified_parent; initialization.verified_parent = verify_parent
    try: yield
    finally: initialization.verified_parent = old


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    seal(p.parse_args().out.resolve())
