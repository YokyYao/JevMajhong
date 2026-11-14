"""Append missing wrapper metadata to a proven complete warm fixed-budget fit.

Never reruns optimization, changes recorded metrics, or reconstructs random fresh
initialization. A missing initial reference can only be copied from a verified
parent whose exact loaded tensors were audited before the first update.
"""
import json
import os
from pathlib import Path
import shutil

from jevmahjong.core.io import digest
from .base_lora import package_digest,validation_profile
from .finalization import fixed_budget_selection,completed_result
from .initialization import bind_parent,check_initialization


def read(path):return json.loads(Path(path).read_text())


def append_json(path,payload):
    with Path(path).open('x') as stream:
        json.dump(payload,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def verify_recovery(manifest,checksum):
    """Check the unchanged originals and all append-only derived outputs."""
    from jevmahjong.evaluation.unified_contract import verify_files
    manifest=Path(manifest)
    if digest(manifest)!=checksum:raise ValueError('Recovery manifest changed')
    proof=read(manifest)
    if proof['status']!='recovered_completed_warm_fixed_budget_fit':raise ValueError('Recovery incomplete')
    if package_digest(Path(proof['recovery_source'])/'jevmahjong')!=proof['recovery_source_sha256']:
        raise ValueError('Recovery implementation changed')
    verify_files(proof['preserved_files']);verify_files(proof['initial_reference_files'])
    binding={'manifest':str(manifest.resolve()),'sha256':checksum}
    for path,payload in proof['derived_records'].items():
        if read(path)!=dict(payload,finalization_recovery=binding):raise ValueError('Recovered metadata changed')
    return proof


def recover(run,failure_log,scheduler_evidence):
    from jevmahjong.evaluation.unified_contract import bundle,verify_files,MODEL_FILES
    root=Path(run).resolve();protocol=read(root/'protocol.json');cfg=protocol['config'];args=protocol['trainer_args']
    trainer=Path(args['out']).resolve();manifest=root/'finalization-recovery-private.json'
    selection_file=trainer/'selection.json';result_file=root/'result.json';initial=trainer/'checkpoints/step-000000'
    if any(p.exists() for p in (manifest,selection_file,result_file,initial)):
        raise ValueError('Recovery refuses existing outputs; inspect rather than retry')
    if cfg.get('initialization')!='continued':raise ValueError('Missing fresh initialization cannot be reconstructed')
    if trainer!=root/'training-run':raise ValueError('Unexpected training directory')
    verify_files(protocol['input_sha256']);verify_files(protocol['code_sha256'])
    if package_digest(Path(cfg['upstream_source'])/'jevany')!=cfg['upstream_source_sha256']:
        raise ValueError('Original upstream changed')
    metrics=read(trainer/'training_metrics.json');curve=read(root/'validation-curve.json')
    selection=fixed_budget_selection(cfg,args,metrics,curve)
    receipt=read(cfg['data_receipt']);admission=read(root/'token-admission.json')
    n=receipt['files']['train']['records']
    if (admission['requested_records']!=n or admission['admitted_records']!=n or admission['rejected_records']
            or metrics['admitted_records']!=n or metrics['rejected_records'] or metrics['truncated_records']
            or metrics['distributed_padding_records'] or metrics['records_seen']!=metrics['optimizer_steps']*metrics['global_effective_batch']):
        raise ValueError('Complete admission and fixed-budget exposure unverified')
    records=[json.loads(line) for line in Path(args['eval_data']).read_text().splitlines() if line.strip()]
    if len(records)!=receipt['files']['validation']['records']:raise ValueError('Validation count changed')
    for item in curve:
        rows=read(trainer/'training_eval'/f"step-{item['step']:06d}"/'development/rows.json')
        actual=validation_profile(records,rows,cfg['validation_phase_weights'])
        if any(item[k]!=v for k,v in actual.items()):raise ValueError('Actual validation predictions differ')
        if item['step']:bundle(trainer/'checkpoints'/f"step-{item['step']:06d}")
    parent=bind_parent(cfg,args)
    if protocol['parent_initialization']!=parent or read(root/'parent-initialization-private.json')!=parent:
        raise ValueError('Original parent binding changed')
    audit=read(root/'initialization-audit.json');updates=read(root/'actual-update-proof.json')
    result=completed_result(cfg,metrics,selection,curve,True)
    check_initialization(cfg,result,audit,updates,parent)
    if (audit['adapter_tensors']<=0 or audit['head_tensors']<=0
            or audit['parent_selected_step']!=parent['selected_step']):raise ValueError('Initial exact-parent audit incomplete')
    source_parent=Path(parent['model']['directory']);parent_bundle=bundle(source_parent)
    if parent_bundle!=parent['model']:raise ValueError('Verified parent bundle changed')
    final=trainer/'checkpoints'/f"step-{metrics['optimizer_steps']:06d}"
    if bundle(final)['files']!=bundle(trainer)['files']:raise ValueError('Saved final bundle differs from root output')
    log=Path(failure_log).resolve();log_text=log.read_text()
    if 'FileNotFoundError' not in log_text or 'selection.json' not in log_text or 'saved ' not in log_text:
        raise ValueError('Original post-save wrapper failure log required')
    if scheduler_evidence.get('state')!='FAILED' or scheduler_evidence.get('exit_code')!='1:0':
        raise ValueError('Recorded actual terminal job evidence required')
    preserved={str(p):digest(p) for p in root.rglob('*') if p.is_file()}
    preserved.update(protocol['input_sha256']);preserved.update(protocol['code_sha256']);preserved[str(log)]=digest(log)
    recovery_source=Path(__file__).resolve().parents[2]
    source_sha=package_digest(recovery_source/'jevmahjong')
    # Only now create the explicitly reconstructed initial *reference*. Its
    # head metadata retains the parent's real optimizer step, not a fake zero.
    initial.mkdir(parents=True,exist_ok=False)
    for name in MODEL_FILES:shutil.copyfile(source_parent/name,initial/name)
    provenance=dict(origin='reconstructed_verified_parent_initial_reference',
        not_an_original_step_zero_serialization=True,parent_bundle=parent_bundle,
        startup_exact_tensor_audit_sha256=digest(root/'initialization-audit.json'),
        optimizer_training_rerun=False)
    append_json(initial/'reconstruction-private.json',provenance)
    if bundle(initial)['files']!=parent_bundle['files']:raise ValueError('Initial reference copy changed')
    selection['record_origin']='recovered_wrapper_fixed_budget_CE_summary'
    result.update(finalization_recovered=True,initial_reference_origin=provenance['origin'])
    proof=dict(status='recovered_completed_warm_fixed_budget_fit',run=str(root),
        recovery_source=str(recovery_source),recovery_source_sha256=source_sha,
        preserved_files=preserved,initial_reference_files={str(p):digest(p) for p in initial.iterdir()},
        derived_records={str(selection_file):selection,str(result_file):result},
        original_scheduler_terminal=scheduler_evidence,original_failure_log=str(log),
        initial_reference_provenance=provenance,optimizer_training_rerun=False,
        original_training_metrics_unchanged=True,validation_profiles_recomputed=len(curve),
        full_match_improvement_proven=False)
    verify_files(preserved);append_json(manifest,proof)
    binding={'manifest':str(manifest),'sha256':digest(manifest)}
    append_json(selection_file,dict(selection,finalization_recovery=binding))
    append_json(result_file,dict(result,finalization_recovery=binding))
    verify_recovery(manifest,binding['sha256'])
    return binding
