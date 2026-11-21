"""Admit a completed, quality-checked curriculum final checkpoint as a parent."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


class ParentNotReady(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def require_final_integrity(reg, completion, audit, validation, identity):
    """Shared completed-final integrity; no research/strict quality substitution."""
    c = reg['curriculum']
    last = reg['registered_final_step']
    expected = c['epochs']*math.ceil(reg['train_records']/(c['batch']*c['accumulation']))
    if (last != expected or not c['candidate_rule'].startswith('registered_final_full_')
            or completion.get('status') != 'stage1_fit_and_curves_complete'
            or completion.get('optimizer_steps') != last
            or completion.get('actual_records_seen') != c['epochs']*reg['train_records']):
        raise ParentNotReady('The registered full-budget parent has not completed')
    advancement = audit.get('stage1_advancement')
    if (audit.get('status') != 'stage1_final_evidence_audit_passed'
            or audit.get('optimizer_steps') != last
            or not audit.get('source_and_data_hashes_unchanged')
            or not audit.get('heldout_groups_excluded_from_all_stages')
            or not isinstance(advancement, dict)):
        raise ParentNotReady('A completed evidence audit with the original stage diagnostic is required')
    rc = validation.get('riichi_type_counts', {})
    positives = rc.get('tp', 0)+rc.get('fn', 0)
    negatives = rc.get('tn', 0)+rc.get('fp', 0)
    if (validation.get('local_step') != last or not positives or not negatives
            or positives != validation['stratum_counts']['riichi_yes']
            or negatives != validation['stratum_counts']['riichi_no']
            or any(type(rc.get(k,0)) is not int or rc.get(k,0)<0 for k in ('tp','fn','tn','fp'))
            or any(not math.isfinite(validation[k]['plain']) for k in ('stratum_accuracy','stratum_ce'))):
        raise ParentNotReady('Final validation counts, metrics or registered step differ')
    final_reload = [r for r in audit['saved_reloads'] if r['step'] == last]
    if not isinstance(identity,str) or not identity or len(final_reload) != 1 or final_reload[0]['identity'] != identity:
        raise ParentNotReady('Parent bundle is not the audited registered final checkpoint')
    schema=c.get('riichi_input_schema')
    if schema and (audit.get('input_schema')!=schema
            or audit.get('all_saved_checkpoint_contracts_verified') is not True
            or audit.get('final_raw_public_serving_menus')!=positives+negatives
            or not isinstance(audit.get('final_raw_public_max_probability_error'),(int,float))
            or not math.isfinite(audit['final_raw_public_max_probability_error'])
            or audit['final_raw_public_max_probability_error']>=5e-5
            or not audit.get('final_policy_identity_sha256')):
        raise ParentNotReady('Context parent requires the full raw-public serving audit, not the core audit alone')
    return rc.get('tp',0)/positives,rc.get('tn',0)/negatives


def require_stage1_candidate(reg, completion, audit, validation, identity):
    """The original strict gate remains the default, with actual metric checks."""
    accept,decline=require_final_integrity(reg,completion,audit,validation,identity)
    if (not audit['stage1_advancement'].get('passed') or accept<.6 or decline<.6
            or validation['stratum_accuracy']['plain'] < .4845576407506702
            or validation['stratum_ce']['plain'] > 1.3489726760783527*1.05):
        raise ParentNotReady('The final checkpoint does not satisfy the registered basic/riichi retention gate')


def require_child_input_contract(reg, child):
    schema=reg['curriculum'].get('riichi_input_schema')
    if schema != child.get('required_input_schema'):
        raise ParentNotReady('Parent and child require the same public input schema')
    if schema:
        feature=ROOT/'tools/curriculum_riichi_context.py'
        expected=reg['inputs'].get(str(feature.resolve()))
        if not expected or child.get('feature_code_sha256')!=expected or sha(feature)!=expected:
            raise ParentNotReady('Child public feature implementation differs from its parent')


def require_replay(parent_rows,child_rows):
    """Keep inputs, labels, targets and groups; only loss-only annotations may differ."""
    byid={r['_meta']['id']:r for r in child_rows}
    if len(byid)!=len(child_rows):raise ParentNotReady('Duplicate child replay IDs')
    for row in parent_rows:
        child=byid.get(row['_meta']['id'])
        if child is None or child['_meta']['group_id']!=row['_meta']['group_id']:
            raise ParentNotReady('Mandatory parent replay row/group missing')
        public=lambda r:dict(state=r['state'],questions={name:{k:q[k] for k in ('type','instructions','criteria')}
            for name,q in r['questions'].items()})
        if public(row)!=public(child):raise ParentNotReady('Mandatory replay public input changed')
        for name,q in row['questions'].items():
            if any(q.get(k)!=child['questions'][name].get(k) for k in ('label','target')):
                raise ParentNotReady('Mandatory replay supervision changed')


def seal(registered, evidence, fit_exit, audit_exit, child_data_ready, out, research_policy_binding=None):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    reg = read(registered)
    c = reg['curriculum']
    fit = ROOT/c['out']
    required = [fit/'completion.json', Path(evidence), Path(fit_exit), Path(audit_exit)]
    if any(not p.exists() for p in required):
        raise ParentNotReady('Parent fit/audit completion or exit receipt is missing; no seal written')
    completion, audit = read(required[0]), read(evidence)
    validation_file = fit/'metrics/validation_metrics.jsonl'
    validation = rows(validation_file)[-1]
    reloads = [r for r in audit.get('saved_reloads', []) if r['step'] == reg['registered_final_step']]
    identity = reloads[0]['identity'] if len(reloads) == 1 else None
    research_policy=None;research_files={};research_grade=None
    if research_policy_binding:
        sys.path.insert(0,str(ROOT))
        from tools.curriculum_research_parent import load_bound_policy,require_research_candidate,METHOD
        research_policy,research_files=load_bound_policy(research_policy_binding,registered)
    def qualify(actual_identity):
        if research_policy is not None:
            return require_research_candidate(reg,completion,audit,validation,actual_identity,research_policy)
        require_stage1_candidate(reg,completion,audit,validation,actual_identity)
        return audit['stage1_advancement']
    research_grade=qualify(identity)
    scheduler_receipts = []
    for path in (fit_exit, audit_exit):
        receipt = read(path)
        if receipt['exit_code'] != 0:
            raise ParentNotReady('Parent fit or audit did not exit successfully')
        process = subprocess.run(['scontrol', '-o', 'show', 'job', str(int(receipt['job_id']))],
                                 check=True, text=True, capture_output=True, timeout=20)
        values = dict(token.split('=', 1) for token in process.stdout.split() if '=' in token)
        if values.get('JobState') != 'COMPLETED' or values.get('ExitCode') != '0:0':
            raise ParentNotReady('Parent scheduler exit is not COMPLETED/0:0')
        scheduler_receipts.append(dict(job_id=receipt['job_id'], status=process.stdout.strip()))
    for path, checksum in reg['inputs'].items():
        if sha(path) != checksum:
            raise ParentNotReady('Parent registered input changed: '+path)
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
        or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ParentNotReady('Parent research or upstream package changed')
    checkpoint = fit/'training-run/checkpoints'/f"step-{reg['registered_final_step']:06d}"
    model = bundle(checkpoint)
    research_grade=qualify(model['identity_sha256'])
    contract_path=None
    if c.get('riichi_input_schema'):
        sys.path.insert(0,str(ROOT))
        from tools.curriculum_input_contract import verify_checkpoint,NAME
        contract=verify_checkpoint(checkpoint,reg,bundle)
        if contract['policy_identity_sha256']!=audit['final_policy_identity_sha256']:
            raise ParentNotReady('Context policy contract differs from the completed serving audit')
        contract_path=checkpoint/NAME
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    if args['max_steps'] != reg['registered_final_step'] or args.get('early_stop_patience', 0) != 0:
        raise ParentNotReady('Parent did not use the fixed registered budget')
    child = read(child_data_ready)
    require_child_input_contract(reg,child)
    if child['status'] != 'stage2_data_prepared_pending_parent_and_fit_registration':
        raise ParentNotReady('This seal requires the prepared stage2 replay dataset')
    for spec in child['files'].values():
        if sha(ROOT/spec['path']) != spec['sha256']:
            raise ParentNotReady('Child prepared data changed')
    parent_train = rows(args['data'])
    parent_validation = rows(args['eval_data'])
    child_train = rows(ROOT/child['files']['train']['path'])
    child_validation = rows(ROOT/child['files']['validation']['path'])
    groups = lambda pool: {r['_meta']['group_id'] for r in pool}
    reserved_file = ROOT/c['data_root']/'reserved-public-groups.json'
    held = groups(parent_validation) | groups(child_validation) | set(read(reserved_file))
    if groups(parent_train) & groups(child_validation) or groups(child_train) & held:
        raise ParentNotReady('Child or ancestral training intersects a held-out group')
    require_replay(parent_train,child_train)
    require_replay(parent_validation,child_validation)
    files = dict(reg['inputs'])
    for path,h in research_files.items():
        if path in files and files[path]!=h:raise ParentNotReady('Conflicting research-policy binding')
        files[path]=h
    paths = [Path(registered), *required, validation_file, Path(child_data_ready), reserved_file,
             fit/'protocol.json', fit/'initialization-audit.json', fit/'actual-update-proof.json',
             fit/'metrics/train_metrics.jsonl', fit/'metrics/evaluation_metrics.jsonl', Path(__file__)]
    binding_file = Path(registered).parent.parent/'audit-binding-private.json'
    binding = read(binding_file)
    for path, checksum in binding.items():
        if sha(path) != checksum:
            raise ParentNotReady('Parent audit code binding changed')
        files[path] = checksum
    paths.append(binding_file)
    for point in completion['curve_steps']:
        folder = fit/'evaluations'/f'step-{point:06d}'
        paths.extend(folder/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'))
    paths.extend(ROOT/spec['path'] for spec in child['files'].values())
    for name, checksum in model['files'].items():
        files[str((checkpoint/name).resolve())] = checksum
    if contract_path:files[str(contract_path.resolve())]=sha(contract_path)
    for path in paths:
        files[str(path.resolve())] = sha(path)
    result = dict(status='curriculum_stage1_final_parent_sealed', method=METHOD if research_policy is not None else 'curriculum_registered_final_with_quality_v1',
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], model=model,
        selected_step=reg['registered_final_step'], selection_from_minimum_CE=False,
        parent_registration=str(Path(registered).resolve()), parent_evidence=str(Path(evidence).resolve()),
        child_data_ready=str(Path(child_data_ready).resolve()),
        required_input_schema=c.get('riichi_input_schema'),
        parent_training_file=str(Path(args['data']).resolve()), ancestor_training_files=[str(Path(args['data']).resolve())],
        parent_validation_files=[str(Path(args['eval_data']).resolve())], bound_files=files,
        stage_quality=research_grade,original_stage1_advancement=audit['stage1_advancement'],scheduler_exits=scheduler_receipts,
        research_parent_policy_binding=str(Path(research_policy_binding).resolve()) if research_policy_binding else None,
        group_proof=dict(parent_train_groups=len(groups(parent_train)), child_train_groups=len(groups(child_train)),
            child_validation_groups=len(groups(child_validation)), overlap_with_any_held_group=0,
            mandatory_stage1_replay_present=True,mandatory_stage1_validation_replay_present=True,
            replay_public_inputs_labels_targets_and_groups_equal=True),
        base=c['base'], base_revision=c['base_revision'], optimizer_resumed=False,
        child_fit_started=False, practical_v8_unchanged=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], selected_step=result['selected_step'],
                         identity=model['identity_sha256'], files=len(files))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('registered', 'evidence', 'fit-exit', 'audit-exit', 'child-data-ready', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--research-policy-binding',type=Path)
    a = p.parse_args()
    seal(a.registered, a.evidence, a.fit_exit, a.audit_exit, a.child_data_ready, a.out,a.research_policy_binding)
