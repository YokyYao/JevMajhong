"""Independently reconstruct placement exposures, predictions and native ledgers.

An evidence audit can complete even when the prospective development grade
fails. Neither status establishes independent v8 parity or authorizes release.
"""
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_parent import rows
from tools.curriculum_placement_parent_v1 import verify_parent
from tools.curriculum_placement_runtime_v1 import require_registration, policy_contract, prediction_parity, require_step0, matched_curve
from tools.curriculum_placement_partition_v1 import profile, require_partition
from tools.curriculum_placement_input_v1 import PlacementPolicyClient
from tools.curriculum_placement_objective_v1 import METHOD, COMPONENTS
from tools.curriculum_placement_assessment_v1 import require_criteria, early_profile, paired_development, grade


def journal_evidence(reg, logs, orders):
    c = reg['curriculum']; final = reg['registered_final_step']; exposed = Counter(); cumulative = 0
    if [r['local_step'] for r in logs] != list(range(1, final+1)) or orders != reg['expected_orders']:
        raise ValueError('Every actual optimizer observation and exact two prepared epoch receipts required')
    planned = [counts for order in orders for counts in order['source_exposure_counts_by_update']]
    if [r['source_counts'] for r in logs] != planned:
        raise ValueError('Actual per-update source exposures differ from the fixed permutations')
    for row in logs:
        counts = row['source_counts']; values = row['source_loss_components']; n = sum(counts.values())
        if (not n or set(counts) != set(values) or not set(counts) <= set(reg['course_loss_weights'])
                or any(type(v) is not int or v <= 0 for v in counts.values())
                or row['loss_method'] != METHOD or row['reward'] is not None
                or row['global_step'] != c['stage_offset']+row['local_step']
                or any(not math.isfinite(row[k]) for k in ('loss', 'ce', 'lr', 'grad_norm'))):
            raise ValueError('Finite real SFT observations, explicit counts and cumulative step required')
        exposed.update(counts); cumulative += n
        if row['records_seen'] != cumulative: raise ValueError('Raw exposure counter differs')
        for source, components in values.items():
            if (set(components) != set(COMPONENTS) or any(not math.isfinite(v) for v in components.values())
                    or abs(components['objective']-components['standard_target_ce']-components['extra_RI_group_ce']) > 1e-5):
                raise ValueError('Finite full class2 loss decomposition required')
        for component, field in (('objective', 'loss'), ('standard_target_ce', 'ce'), ('extra_RI_group_ce', 'extra_RI_class_loss')):
            actual = sum(counts[s]*reg['course_loss_weights'][s]*values[s][component] for s in counts)/n
            if abs(actual-row[field]) > 1e-5: raise ValueError('Recorded source-weighted objective differs')
        upstream = row['raw_upstream_metrics']
        if (upstream['optimizer_step'] != row['local_step'] or upstream['train/records_seen'] != cumulative
                or abs(upstream['train/objective']-row['loss']) > 1e-5):
            raise ValueError('Actual upstream optimizer metrics differ from the component journal')
    if cumulative != reg['expected_record_exposures']:
        raise ValueError('Full registered two-epoch exposure budget required')
    return dict(actual_record_exposures=cumulative, actual_source_exposures=dict(exposed),
        actual_optimizer_observations=len(logs))


def audit(registered, out):
    gpu_only(); reg = read(registered); require_registration(reg); c = reg['curriculum']; fit = ROOT/c['out']
    run = registered.parent.parent; launch_path = run/'launch-binding-private.json'; launch = read(launch_path)
    if (launch['status'] != 'placement_actual_upstream_integration_and_launch_bound_v1'
            or launch['registered_sha256'] != sha(registered) or not launch['fit_launch_admitted']
            or launch['automatic_release'] or launch['final_reserved_grid_opened']):
        raise ValueError('Original complete pre-update launch binding required')
    criteria_path = Path(launch['criteria_config']).resolve(); criteria = read(criteria_path); require_criteria(criteria)
    required = [criteria_path, Path(__file__).resolve(), ROOT/'tools/curriculum_placement_assessment_v1.py',
        ROOT/'tools/train_curriculum_placement_course_v1.py', ROOT/'tools/evaluate_curriculum_placement_course_v1.py',
        ROOT/'tools/plot_curriculum.py']
    if any(str(p.resolve()) not in launch['files'] for p in required):
        raise ValueError('Criteria, auditor, native evaluator, trainer and plots must be bound before updating')
    if any(path in reg['inputs'] and reg['inputs'][path] != expected for path,expected in launch['files'].items()):
        raise ValueError('Conflicting immutable budget/launch input identities')
    def verify():
        for path, expected in {**reg['inputs'], **launch['files']}.items():
            if sha(path) != expected: raise ValueError('Frozen training/launch input changed: '+path)
    verify()
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from safetensors.torch import load_file
    from jevmahjong.training.initialization import exact_loaded_tensors
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevany.data import api_request
    parent = verify_parent(reg['training_config']); done = read(fit/'completion.json'); final_step = reg['registered_final_step']
    if (done['status'] != 'placement_full_budget_fit_and_real_curves_complete'
            or done['optimizer_steps'] != final_step or done['actual_records_seen'] != reg['expected_record_exposures']
            or not done['new_child_policy_baseline64'] or done['same_execution_policy_as_parent']
            or done['independent_confirmation_opened'] or done['automatic_release']):
        raise ValueError('Actual completed full placement budget and changed-policy baseline required')
    initial = read(fit/'initialization-audit.json'); update = read(fit/'actual-update-proof.json')
    if (not initial['parent_adapter_exact'] or not initial['parent_head_exact'] or not initial['new_optimizer']
            or initial['pretrained_base_requires_grad'] or initial['external_residual_networks']
            or initial['parent_identity_sha256'] != c['parent_identity']
            or not update['continued_LoRA_parameters_changed'] or not update['head_parameters_changed']
            or update['cached_encoder_features_used']):
        raise ValueError('Actual inherited LoRA/head/new optimizer/update proof required')
    zero = fit/'training-run/checkpoints/step-000000'; warm = read(fit/'warmstart-validation-private.json')
    exact = exact_loaded_tensors(load_file(zero/'adapter_model.safetensors'),
        torch.load(zero/'head.pt', map_location='cpu', weights_only=True)['head'], parent['model']['directory'])
    if exact != warm['exact_tensors']: raise ValueError('Actual exact warm-start proof differs')
    require_step0(bundle(zero), reg, warm)
    data = read(c['data_ready']); train = rows(ROOT/data['files']['train']['path']); validation = rows(ROOT/data['files']['validation']['path'])
    sealed = read(c['parent_seal']); manifest = read(sealed['child_membership_file']); require_partition(validation, manifest, 'validation')
    if len(train) != reg['train_records'] or len(validation) != reg['validation_records']:
        raise ValueError('Actual complete expanded TRAIN/unique VAL required')
    parent_reg = read(sealed['parent_registration']); parent_args = tomllib.loads(Path(parent_reg['training_config']['trainer_config']).read_text())
    parent_predictions = read(ROOT/parent_reg['curriculum']['out']/'training-run/training_eval/step-002500/development/rows.json')
    zero_predictions = read(fit/'training-run/training_eval/step-000000/development/rows.json')
    parity = prediction_parity(rows(parent_args['eval_data']), parent_predictions, validation, zero_predictions)
    if parity != warm['early_prediction_parity']: raise ValueError('Complete early-input inheritance differs')
    orders = [read(fit/'epoch-order-private'/f'epoch-{i:02d}.json') for i in (1,2)]
    exposures = journal_evidence(reg, rows(fit/'metrics/train_metrics.jsonl'), orders)
    if Counter(exposures['actual_source_exposures']) != Counter({s: 2*n for s,n in Counter(r['questions']['action']['src'] for r in train).items()}):
        raise ValueError('Each actual expanded row must have two source exposures')
    final = fit/'training-run/checkpoints'/f'step-{final_step:06d}'
    state = torch.load(final/'trainer_state.pt', map_location='cpu', weights_only=True)
    if (state['optimizer_step'] != final_step or state['epoch'] != 2 or state['next_microbatch'] != 0
            or state['records_seen'] != reg['expected_record_exposures'] or len(state['optimizer']['state']) != 376
            or any(float(v['step']) != final_step for v in state['optimizer']['state'].values())
            or state['world_size'] != 1 or state['dataset_sha256'] != reg['expanded_dataset_fingerprint']
            or sorted(state['order'].tolist()) != list(range(reg['train_records']))
            or hashlib.sha256(json.dumps(state['order'].tolist()).encode()).hexdigest() != orders[-1]['order_sha256']):
        raise ValueError('Actual durable final376-state optimizer/dataset/order differs')
    del state; gc.collect()
    selection = read(fit/'training-run/selection.json')
    if selection['registered_final_step'] != final_step or selection['evaluation_candidate_rule'] != 'final_registered_budget_v1':
        raise ValueError('Minimum-CE diagnostic cannot replace the registered final candidate')
    curve = read(fit/'validation-curve.json'); val_journal = rows(fit/'metrics/validation_metrics.jsonl'); early = {}
    if [p['step'] for p in curve] != c['validation_and_game_steps'] or [p['local_step'] for p in val_journal] != c['validation_and_game_steps']:
        raise ValueError('Complete actual validation trace required')
    for point, event in zip(curve, val_journal):
        predictions = read(fit/'training-run/training_eval'/f"step-{point['step']:06d}"/'development/rows.json')
        actual = profile(validation, predictions, {}, target_cross_entropy, manifest)
        if (any(point[k] != value for k,value in actual.items()) or event['loss'] != actual['weighted_class_objective']
                or event['ce'] != actual['weighted_target_ce']):
            raise ValueError('Unique57-source validation reconstruction/journal differs')
        early[point['step']] = early_profile(validation, predictions)
    identity = bundle(final); client = PlacementPolicyClient(UnifiedPolicyClient(load_model(final)))
    predictions = read(fit/'training-run/training_eval'/f'step-{final_step:06d}'/'development/rows.json')
    saved = {p['id']:p for p in predictions}; error = 0.
    for row in validation:
        actual = client.system_one(**api_request(row))['answers']['action']['probabilities']; previous = saved[row['_meta']['id']]
        if set(actual) != set(previous['keys']): raise ValueError('Actual serving reload menu differs')
        if any(not math.isfinite(v) for v in actual.values()): raise ValueError('Nonfinite serving probability')
        error = max(error, max(abs(actual[key]-p) for key,p in zip(previous['keys'],previous['p'])))
    if error >= 5e-5: raise ValueError('Full unique validation serving reload parity failed')
    del client; gc.collect(); torch.cuda.empty_cache()
    contracts = read(fit/'policy-contracts-private.json'); games = {}; summaries = {}; checked = 0
    game_journal = rows(fit/'metrics/evaluation_metrics.jsonl')
    if [r['local_step'] for r in game_journal] != c['validation_and_game_steps']:
        raise ValueError('Complete five-point actual16-cell evaluation journal required')
    for step, event in zip(c['validation_and_game_steps'], game_journal):
        folder = fit/'evaluations'/f'step-{step:06d}'; records = rows(folder/'matches-private.jsonl'); protocol = read(folder/'protocol.json')
        for row in records:
            seat = row['seat']; log = row['mjai_log']; financial = settlement_ledger(log, row['final_scores'])
            if (financial['hands'] != row['hands'] or row['net_points'] != row['final_scores'][seat]-25000
                    or row['rank'] != sorted(range(4),key=lambda p:(-row['final_scores'][p],p)).index(seat)+1
                    or row['wins'] != sum(e['type']=='hora' and e.get('actor')==seat for e in log)
                    or row['dealin_hands'] != hand_dealin_count(log,seat)):
                raise ValueError('Actual native full-match settlement or counters differ')
            for operation,n in row['choices'].items():
                if n != sum(e['type']==operation and e.get('actor')==seat for e in log):
                    raise ValueError('Actual native operation counts differ')
        summary, subset = matched_curve(records, reg, step, aggregate_complete_matches)
        stored = read(folder/'summary.json'); stored_subset = read(folder/'curve16-summary.json')
        if (any(stored[k] != value for k,value in summary.items())
                or any(stored_subset[k] != value or event[k] != value for k,value in subset.items())):
            raise ValueError('Actual64 endpoints/common16 curve reconstruction differs')
        model = bundle(fit/'training-run/checkpoints'/f'step-{step:06d}'); policy, policy_id = policy_contract(model,reg)
        if (protocol['checkpoint'] != model or protocol['policy'] != policy or protocol['policy_identity'] != policy_id
                or protocol['profiles'] != c['opponent_profiles'] or protocol['seats'] != 4
                or protocol['parent_cache_reused_as_step0']
                or stored['checkpoint_sha256'] != model['identity_sha256'] or event['policy_identity'] != policy_id
                or contracts[str(step)] != dict(model_identity=model['identity_sha256'],policy=policy,policy_identity=policy_id)):
            raise ValueError('Actual fresh child checkpoint/policy contract differs')
        games[step] = records; summaries[step] = stored; checked += len(records)
    comparison = paired_development(games[final_step],games[0],c['curve_seed_start'])
    assessment = grade(early[0],early[final_step],comparison,criteria)
    for ext in ('png','pdf'):
        if (fit/'metrics'/f'curriculum-curves.{ext}').stat().st_size < 1000: raise ValueError('Actual curves missing')
    verify()
    if bundle(final) != identity: raise ValueError('Audit changed actual model parameters')
    result = dict(status='placement_full_budget_native_and_validation_evidence_audited',
        final_identity=identity['identity_sha256'], final_policy_identity=policy_contract(identity,reg)[1],
        optimizer_steps=final_step, **exposures, source_and_launch_hashes_unchanged=True,
        exact_parent_tensors=exact, complete_early_prediction_parity=parity, all376_optimizer_states_match=True,
        saved_final_reloaded_menus=len(validation), saved_final_max_probability_error=error,
        checked_complete_games=checked, zero64=summaries[0], final64=summaries[final_step],
        child_policy_zero_to_final=comparison, early_retention_by_step=early, development_assessment=assessment,
        CE_minimum_is_diagnostic_only=True, candidate_from_registered_final_budget=True,
        independent_confirmation_not_run_by_this_audit=True, automatic_release=False)
    write(out,result); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args(); audit(a.registered.resolve(),a.out.resolve())
