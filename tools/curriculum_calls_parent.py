"""Seal an audited registered calls final for the attack course.

This is separate from the frozen stage1-to-calls bridge. A scheduler record
that has expired can use a previously captured COMPLETED snapshot, never a
PENDING/RUNNING observation or a successful shell exit alone.
"""
import argparse
from contextlib import contextmanager
import datetime
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_parent import ParentNotReady, read, rows, sha, require_replay, require_child_input_contract
from tools.curriculum_transfer import canonical_protocol

METHOD = 'curriculum_registered_calls_final_v1'


def balanced(counts):
    # The frozen calls auditor serializes Counter as a sparse dictionary:
    # omitted entries mean zero, e.g. the initial always-decline kan policy.
    values = {k: counts.get(k, 0) for k in ('tp', 'fn', 'tn', 'fp')}
    if any(type(v) is not int or v < 0 for v in values.values()):
        raise ParentNotReady('Invalid operation counts')
    positive = values['tp']+values['fn']
    negative = values['tn']+values['fp']
    if not positive or not negative:
        raise ParentNotReady('Choosing and declining must both be represented')
    return .5*(values['tp']/positive+values['tn']/negative)


def qualify(reg, completion, audit, curve, final_summary):
    c = reg['curriculum']
    last = reg['registered_final_step']
    if (c.get('stage') != 'stage2-calls-v1' or c.get('riichi_input_schema') is not None
        or c.get('candidate_rule') != 'registered_final_full_two_epochs_324_updates'
        or last != c['epochs']*math.ceil(reg['train_records']/(c['batch']*c['accumulation']))
        or completion.get('status') != 'stage2_fit_and_curves_complete'
        or completion.get('optimizer_steps') != last
        or completion.get('actual_records_seen') != c['epochs']*reg['train_records']):
        raise ParentNotReady('A completed plain-input fixed-final calls course is required; context needs its serving-contract audit')
    required = ('source_and_data_hashes_unchanged', 'heldout_groups_excluded', 'exact_train_and_validation_replay',
        'all376_optimizer_states_match')
    if (audit.get('status') != 'stage2_final_evidence_audit_passed'
        or audit.get('optimizer_steps') != last
        or audit.get('real_records_seen') != completion['actual_records_seen']
        or not all(audit.get(k) is True for k in required)
        or audit.get('saved_final_reloaded_menus') != 4721
        or not isinstance(audit.get('saved_final_max_probability_error'), (int, float))
        or not math.isfinite(audit['saved_final_max_probability_error'])
        or audit['saved_final_max_probability_error'] >= 5e-5
        or audit.get('checked_complete_games') != 224):
        raise ParentNotReady('Full calls audit, reload and exposure evidence is required')
    expected = sorted({0, last, *range(c['validation_every_steps'], last+1, c['validation_every_steps'])})
    if [p['step'] for p in curve] != expected:
        raise ParentNotReady('Calls validation schedule is incomplete')
    first, final = curve[0], curve[-1]
    plain = final['stratum_accuracy']['plain']-first['stratum_accuracy']['plain']
    riichi = balanced(final['riichi_type_counts'])-balanced(first['riichi_type_counts'])
    calls = statistics.mean(balanced(final['operation_type_counts'][op])-balanced(first['operation_type_counts'][op])
        for op in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan'))
    net = audit['comparison_to_parent']['net_points']['gain']
    dealin = audit['comparison_to_parent']['dealin']['gain']
    if not all(math.isfinite(v) for v in (plain, riichi, calls, net, dealin)):
        raise ParentNotReady('Nonfinite calls quality evidence')
    if final_summary.get('matches') != 64 or final_summary.get('seed_groups') != 8:
        raise ParentNotReady('Final64 grid is required, not the16-game curve subset')
    if any(final_summary.get(k) != v for k, v in audit['final_summary'].items()):
        raise ParentNotReady('Audited and saved final game results differ')
    limits = c['development_targets']
    grade = dict(plain_agreement_gain=plain, riichi_balanced_gain=riichi, five_call_type_balanced_gain=calls,
        passed=plain >= -limits['max_plain_agreement_drop_from_step0']
            and riichi >= -limits['max_riichi_balanced_accuracy_drop_from_step0']
            and calls >= limits['min_five_call_type_balanced_accuracy_gain_from_step0']
            and net >= limits['min_final64_net_gain_over_parent']
            and dealin <= limits['max_final64_dealin_increase_over_parent'],
        development_only_not_strength_proof=True)
    if grade != audit.get('stage2_advancement') or not grade['passed']:
        raise ParentNotReady('The original calls-course quality targets did not pass')
    return grade


def scheduler_snapshot(receipt, observation_file):
    if receipt.get('exit_code') != 0:
        raise ParentNotReady('The fit/audit exit receipt is unsuccessful')
    job = str(int(receipt['job_id']))
    query = subprocess.run(['scontrol', '-o', 'show', 'job', job], capture_output=True, text=True, timeout=20)
    if query.returncode == 0:
        f = dict(t.split('=', 1) for t in query.stdout.split() if '=' in t)
        if f.get('JobId') != job or f.get('JobState') != 'COMPLETED' or f.get('ExitCode') != '0:0':
            raise ParentNotReady('Scheduler has not completed this job successfully')
        return dict(job_id=job, source='current_scheduler_query', state='COMPLETED', exit_code='0:0', raw=query.stdout.strip())
    if 'invalid job id' not in query.stderr.lower():
        raise ParentNotReady('Scheduler query failed transiently; no terminal state inferred')
    captures = rows(observation_file)
    for capture in reversed(captures):
        known = capture.get('jobs', {}).get(job, {})
        if (known.get('source') == 'current_scheduler_query' and known.get('state') == 'COMPLETED'
            and known.get('exit_code') == '0:0' and str(known.get('job_id')) == job):
            return dict(job_id=job, source='preserved_verified_completed_scheduler_observation', observation_utc=capture['utc'],
                observation_file=str(Path(observation_file).resolve()), observation_sha256=sha(observation_file), snapshot=known,
                current_query_error=query.stderr.strip(), state='COMPLETED', exit_code='0:0')
    raise ParentNotReady('Expired scheduler history has no captured successful completion')


def seal(registered, evidence, fit_exit, audit_exit, observations, child_data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    reg = read(registered)
    c = reg['curriculum']
    fit = ROOT/c['out']
    completion = read(fit/'completion.json')
    audit = read(evidence)
    curve = read(fit/'validation-curve.json')
    last = reg['registered_final_step']
    summary = read(fit/'evaluations'/f'step-{last:06d}'/'summary.json')
    quality = qualify(reg, completion, audit, curve, summary)
    scheduler = [scheduler_snapshot(read(p), observations) for p in (fit_exit, audit_exit)]
    files = dict(reg['inputs'])
    for p, h in files.items():
        if sha(p) != h:
            raise ParentNotReady('Calls registered input changed: '+p)
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.training.initialization import lineage_training_files, check_groups
    from jevmahjong.evaluation.unified_contract import bundle
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
        or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ParentNotReady('Pinned training package changed')
    checkpoint = fit/'training-run/checkpoints'/f'step-{last:06d}'
    model = bundle(checkpoint)
    if (read(fit/'evaluations'/f'step-{last:06d}'/'protocol.json')['checkpoint'] != model
        or summary['checkpoint_sha256'] != model['identity_sha256']):
        raise ParentNotReady('The evaluated final bundle differs from the selected parent')
    parent_protocol = canonical_protocol(read(fit/'protocol.json'))
    if parent_protocol['config'] != reg['training_config']:
        raise ParentNotReady('Actual calls protocol differs from its registration')
    lineage = lineage_training_files(parent_protocol)
    child = read(child_data_ready)
    require_child_input_contract(reg, child)
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    child_paths = {k: (ROOT/v['path']).resolve() for k, v in child['files'].items()}
    if set(child_paths) != {'train', 'validation'}:
        raise ParentNotReady('Two explicit prepared child splits are required')
    for split, path in child_paths.items():
        if sha(path) != child['files'][split]['sha256']:
            raise ParentNotReady('Prepared attack data changed')
    require_replay(rows(args['data']), rows(child_paths['train']))
    require_replay(rows(args['eval_data']), rows(child_paths['validation']))
    proof = check_groups(lineage[0], child_paths['train'], child_paths['validation'], lineage[1:])
    held = {r['_meta']['group_id'] for r in rows(args['eval_data'])} | set(read(child['reserved_groups']))
    if held & {r['_meta']['group_id'] for r in rows(child_paths['train'])}:
        raise ParentNotReady('Child training intersects parent validation or reserved groups')
    binding = registered.parent.parent/'audit-binding-private.json'
    for p, h in read(binding).items():
        if sha(p) != h:
            raise ParentNotReady('Calls auditor changed since binding')
        files[p] = h
    paths = [registered, evidence, fit_exit, audit_exit, child_data_ready, binding, Path(__file__),
        fit/'completion.json', fit/'validation-curve.json', fit/'protocol.json', fit/'initialization-audit.json',
        fit/'actual-update-proof.json', checkpoint/'trainer_state.pt', Path(child['reserved_groups']),
        fit/'metrics/train_metrics.jsonl', fit/'metrics/validation_metrics.jsonl', fit/'metrics/evaluation_metrics.jsonl', *child_paths.values()]
    # Do not bind an append-only live observation journal. Preserve the exact
    # chosen immutable snapshot in the seal, with the journal's then-hash.
    for step in completion['curve_steps']:
        paths.extend(fit/'evaluations'/f'step-{step:06d}'/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'))
    for name, h in model['files'].items():
        files[str((checkpoint/name).resolve())] = h
    for p in paths:
        files[str(p.resolve())] = sha(p)
    result = dict(status='curriculum_calls_final_parent_sealed', method=METHOD, model=model,
        parent_registration=str(registered), parent_evidence=str(evidence), selected_step=last,
        cumulative_step=c['stage_offset']+last, stage_quality=quality, bound_files=files,
        original_stage1_advancement=audit['original_parent_stage1_advancement'],
        child_data_ready=str(child_data_ready), parent_training_file=str(Path(args['data']).resolve()),
        ancestor_training_files=lineage, parent_validation_files=[str(Path(args['eval_data']).resolve())],
        base=c['base'], base_revision=c['base_revision'], required_input_schema=c.get('riichi_input_schema'),
        selection_from_minimum_CE=False, optimizer_resumed=False, scheduler_exits=scheduler, group_proof=proof,
        sealed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    import json
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], selected_step=last, cumulative_step=result['cumulative_step'], identity=model['identity_sha256'])), flush=True)


def verify_calls_parent(cfg):
    spec = cfg['parent_initialization']
    required = {'method', 'manifest', 'sha256', 'identity_sha256', 'selected_step', 'source', 'source_sha256', 'run', 'child_data_ready'}
    if set(spec) != required or spec.get('method') != METHOD or cfg.get('initialization') != 'continued':
        raise ParentNotReady('Explicit calls-final continuation required')
    manifest = Path(spec['manifest']).resolve()
    if sha(manifest) != spec['sha256']:
        raise ParentNotReady('Calls parent seal changed')
    sealed = read(manifest)
    if (sealed.get('status') != 'curriculum_calls_final_parent_sealed' or sealed.get('method') != METHOD
        or sealed.get('selection_from_minimum_CE') is not False or sealed.get('optimizer_resumed') is not False
        or sealed['selected_step'] != spec['selected_step'] or sealed['model']['identity_sha256'] != spec['identity_sha256']
        or sealed['base'] != cfg['base'] or sealed['base_revision'] != cfg['base_revision']):
        raise ParentNotReady('Wrong registered calls final')
    for p, h in sealed['bound_files'].items():
        if sha(p) != h:
            raise ParentNotReady('Sealed calls evidence changed: '+p)
    reg = read(sealed['parent_registration'])
    fit = (ROOT/reg['curriculum']['out']).resolve()
    for path in (Path(sealed['parent_registration']), Path(sealed['parent_evidence']),
                 fit/'completion.json', fit/'protocol.json', fit/'validation-curve.json'):
        if sealed['bound_files'].get(str(path.resolve())) != sha(path):
            raise ParentNotReady('Required calls audit artifact is not sealed')
    if (Path(spec['run']).resolve() != fit or fit == (ROOT/cfg['out']).resolve()
        or Path(spec['source']).resolve() != Path(reg['source']).resolve()
        or spec['source_sha256'] != reg['source_sha256'] or reg['registered_final_step'] != spec['selected_step']):
        raise ParentNotReady('Calls source, run or selected step differs')
    summary = read(fit/'evaluations'/f"step-{sealed['selected_step']:06d}"/'summary.json')
    grade = qualify(reg, read(fit/'completion.json'), read(sealed['parent_evidence']), read(fit/'validation-curve.json'), summary)
    if grade != sealed['stage_quality']:
        raise ParentNotReady('Recomputed calls quality differs')
    child_path = Path(spec['child_data_ready']).resolve()
    if str(child_path) != sealed['child_data_ready'] or sealed['bound_files'].get(str(child_path)) != sha(child_path):
        raise ParentNotReady('Attack child manifest differs from the seal')
    child = read(child_path)
    require_child_input_contract(reg, child)
    args = tomllib.loads(Path(cfg['trainer_config']).read_text())
    if args.get('resume') or Path(args.get('init_from', '')).resolve() != Path(sealed['model']['directory']).resolve():
        raise ParentNotReady('Use the sealed adapter/head with a new optimizer')
    if cfg.get('riichi_input_schema') != reg['curriculum'].get('riichi_input_schema'):
        raise ParentNotReady('Serving input schema differs')
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        path = (ROOT/child['files'][split]['path']).resolve()
        if Path(args[key]).resolve() != path or sealed['bound_files'].get(str(path)) != sha(path):
            raise ParentNotReady('Actual trainer data differs from the sealed child')
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if package_digest(Path(spec['source'])/'jevmahjong') != spec['source_sha256']:
        raise ParentNotReady('Pinned parent loader package changed')
    model = bundle(sealed['model']['directory'])
    if model != sealed['model'] or summary['checkpoint_sha256'] != model['identity_sha256']:
        raise ParentNotReady('Calls final weights or evaluated identity changed')
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != reg['training_config'] or protocol['trainer_args']['data'] != sealed['parent_training_file']:
        raise ParentNotReady('Calls protocol or lineage changed')
    receipt = dict(sealed['bound_files'])
    receipt[str(manifest)] = spec['sha256']
    return dict(model=model, receipt=receipt, training_protocol=protocol, training_config=reg['training_config'], selected_step=sealed['selected_step'])


@contextmanager
def calls_parent_loader(initialization_module):
    original = initialization_module.verified_parent
    initialization_module.verified_parent = verify_calls_parent
    try:
        yield
    finally:
        initialization_module.verified_parent = original


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('registered', 'evidence', 'fit-exit', 'audit-exit', 'observations', 'child-data-ready', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    seal(a.registered.resolve(), a.evidence.resolve(), a.fit_exit.resolve(), a.audit_exit.resolve(),
        a.observations.resolve(), a.child_data_ready.resolve(), a.out.resolve())
