"""Audit actual final cached RI-class source metrics as a distinct policy.

This is a reusable prerequisite for a future policy-aware parent contract.
It deliberately does not issue an old-format parent seal or start training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_parent import require_final_integrity, rows
from tools.curriculum_riichi_policy_profile_v1 import profile
from tools.curriculum_riichi_class_policy_v1 import METHOD


def audit(out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Bounded scheduler GPU allocation required')
    out.mkdir(parents=True, exist_ok=False)
    from tools.evaluate_curriculum_riichi_class_route_v2 import repaired_execution, base
    config = ROOT/'configs/diagnostics/curriculum-riichi-class-route-v2.json'
    with repaired_execution():
        c, reg, fit, checkpoint, model, files, refs = base.context(config)
    evidence = read(fit.parent/'final-evidence-private.json')
    validation_path = fit/'metrics/validation_metrics.jsonl'
    validation = rows(validation_path)[-1]
    require_final_integrity(reg, read(fit/'completion.json'), evidence, validation, model['identity_sha256'])
    data = read(ROOT/reg['curriculum']['data_root']/'data-ready.json')
    spec = data['stages']['stage1']['validation']; path = ROOT/spec['path']
    if sha(path) != spec['sha256']:
        raise ValueError('Actual final validation changed')
    records = rows(path)
    predictions_path = fit/'training-run/training_eval/step-000608/development/rows.json'
    predictions = read(predictions_path)
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevmahjong.evaluation.unified_contract import bundle
    result = profile(records, predictions, {'weighted_target_ce': validation['natural_loss']}, target_cross_entropy)
    for field in ('source_stratum_accuracy', 'source_stratum_ce', 'stratum_accuracy',
                  'stratum_ce', 'stratum_counts', 'riichi_by_source', 'riichi_type_counts'):
        if result['raw_profile'][field] != validation[field]:
            raise ValueError('Original raw source metric differs: '+field)
    if result['raw_source_quality']['all_source_quality_passed'] is not False:
        raise ValueError('Do not change the actual failed raw-policy grade')
    if result['selected_source_quality']['all_source_quality_passed'] is not True:
        raise ValueError('The distinct fixed-selector policy must actually meet every source condition')
    if (len(records) != 3266 or result['scope_counts'] != {'RI': 632, 'non_RI': 2634}
            or result['changed_choices'].get('non_RI') != 0):
        raise ValueError('Full actual source-menu coverage required')
    route_root = ROOT/'runs/curriculum-riichi-class-route-v2'
    route_summary_path = route_root/'audit-v1/summary-private.json'
    route = read(route_summary_path)
    if (route['status'] != 'fixed_RI_route_native64_independent_cached_audit_passed'
            or not route['baseline16_exact'] or route['checked_complete_match_ledgers'] != 192
            or not route['raw_parameters_unchanged'] or not route['source_grade_not_overridden']
            or not route['final_reserved_grid_unused'] or route['parent_admitted']
            or route['new_optimizer_updates'] != 0):
        raise ValueError('Actual independent native-policy audit required')
    expected_policy = dict(parameter_identity=model['identity_sha256'],
        runtime_source_sha256=reg['source_sha256'], method=METHOD,
        route_sha256=sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py'), config_sha256=sha(config))
    policy_identity = hashlib.sha256(json.dumps(expected_policy, sort_keys=True).encode()).hexdigest()
    if route['policy'] != expected_policy or route['policy_identity'] != policy_identity:
        raise ValueError('Native result belongs to different parameters or selector')
    for binding_path in (route_root/'trial-v1/binding-private.json', route_root/'audit-v1/binding-private.json'):
        binding = read(binding_path)
        for p, h in binding['files'].items():
            if p in files and files[p] != h:
                raise ValueError('Conflicting native-audit binding')
            files[p] = h
        files[str(binding_path)] = sha(binding_path)
    preflight_path = route_root/'preflight-v1/summary-private.json'
    preflight = read(preflight_path)
    if result['selected_profile']['riichi_by_source'] != preflight['source_confusion']:
        raise ValueError('New source metrics differ from the actual native preflight selector')
    for p in (Path(__file__), ROOT/'tools/curriculum_riichi_policy_profile_v1.py',
              ROOT/'tests/core/test_curriculum_riichi_policy_profile_v1.py',
              ROOT/'docs/experiments/plans/curriculum-riichi-policy-profile-20261123.md',
              ROOT/'cluster/curriculum-riichi-policy-profile-v1.slurm',
              route_summary_path, preflight_path, validation_path, predictions_path, path):
        files[str(p.resolve())] = sha(p)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Bound parent data/source/native artifacts changed')
    if bundle(checkpoint) != model:
        raise ValueError('Frozen parent parameters changed during audit')
    summary = dict(status='RI_class_policy_source_metrics_verified_not_parent_admitted',
        method=METHOD, parameter_identity=model['identity_sha256'], policy_identity=policy_identity,
        policy=expected_policy, raw_source_quality=result['raw_source_quality'],
        selected_source_quality=result['selected_source_quality'],
        selected_riichi_by_source=result['selected_profile']['riichi_by_source'],
        actual_cached_menus=len(records), scope_counts=result['scope_counts'],
        changed_choices=result['changed_choices'], raw_grade_preserved=True,
        original_probability_CE_unchanged=True, native_policy_summary=route['summaries']['candidate'],
        native_paired_comparisons=route['comparisons'], native_audit_reused_not_rerun=True,
        native_guard_not_replayed_in_cached_source_agreement=True,
        parent_admitted=False, child_optimizer_updates=0, raw_private=True,
        remaining_parent_requirements=['child replay and split binding',
            'complete parent scheduler/CI/source lineage receipts',
            'explicit distinct-policy continuation contract and verifier',
            'child route-bound validation and native evaluation',
            'pre-update exact tensors,3266 probabilities/choices,16 native outcomes'])
    for name, value in (('summary-private.json', summary),
                        ('profiles-private.json', result),
                        ('binding-private.json', dict(files=files, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID']))):
        with (out/name).open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit(args.out.resolve())
