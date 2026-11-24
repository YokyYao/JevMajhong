"""Independent actual yaku-course exposure, reload, validation and native audit."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_parent import rows, require_replay
from tools.curriculum_yaku_balanced_parent_v1 import verify_parent, gpu_only
from tools.curriculum_yaku_parent_runtime_v1 import require_registration, policy_contract, prediction_parity, require_step0
from tools.curriculum_yaku_parent_fit_v1 import profile
from tools.curriculum_yaku_parent_loss_v1 import METHOD, COMPONENTS, MASS, NEW_SOURCES
from tools.curriculum_calls_fit import retention, balanced


def audit(registered, out):
    gpu_only(); reg = read(registered); require_registration(reg); c = reg['curriculum']; fit = ROOT/c['out']
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
    for p, h in reg['inputs'].items():
        if sha(p) != h: raise ValueError('Registered source changed: '+p)
    parent = verify_parent(reg['training_config']); done = read(fit/'completion.json')
    if (done['status'] != 'yaku2500_fit_and_real_curves_complete' or done['optimizer_steps'] != 2500
            or done['actual_records_seen'] != 182494): raise ValueError('Full actual registered fit required')
    initial = read(fit/'initialization-audit.json'); proof = read(fit/'actual-update-proof.json')
    if (not initial['parent_adapter_exact'] or not initial['parent_head_exact'] or not initial['new_optimizer']
            or initial['pretrained_base_requires_grad'] or initial['external_residual_networks']
            or initial['parent_identity_sha256'] != c['parent_identity']
            or not proof['continued_LoRA_parameters_changed'] or not proof['head_parameters_changed'] or proof['cached_encoder_features_used']):
        raise ValueError('Exact new-optimizer initialization and actual LoRA/head updates required')
    zero = fit/'training-run/checkpoints/step-000000'; warm = read(fit/'warmstart-validation-private.json')
    exact = exact_loaded_tensors(load_file(zero/'adapter_model.safetensors'),
        torch.load(zero/'head.pt', map_location='cpu', weights_only=True)['head'], parent['model']['directory'])
    if warm['exact_tensors'] != exact: raise ValueError('Warm-start tensor proof differs')
    require_step0(bundle(zero), reg, warm)
    data = read(c['data_ready']); train = rows(ROOT/data['files']['train']['path']); validation = rows(ROOT/data['files']['validation']['path'])
    sealed = read(c['parent_seal']); parent_reg = read(sealed['parent_registration'])
    pa = tomllib.loads(Path(parent_reg['training_config']['trainer_config']).read_text())
    require_replay(rows(pa['data']), train); require_replay(rows(pa['eval_data']), validation)
    parent_p = read(ROOT/parent_reg['curriculum']['out']/'training-run/training_eval/step-002472/development/rows.json')
    zero_p = read(fit/'training-run/training_eval/step-000000/development/rows.json')
    parity = prediction_parity(rows(pa['eval_data']), parent_p, validation, zero_p)
    if parity != warm['all_parent_prediction_parity']: raise ValueError('Actual full parent parity differs')
    game_proof = read(fit/'warmstart-games-private.json')
    if (game_proof['matches'] != 16 or not game_proof['exact_complete_legal_parent_events']
            or not game_proof['checked_before_first_update'] or not game_proof['original_guarded_policy_not_equated']):
        raise ValueError('Actual changed-policy parent baseline proof required')
    logs = rows(fit/'metrics/train_metrics.jsonl')
    if [r['local_step'] for r in logs] != list(range(1, 2501)): raise ValueError('All actual optimizer steps required')
    orders = [read(fit/'epoch-order-private'/f'epoch-{i:02d}.json') for i in (1, 2)]
    if [x for o in orders for x in o['source_counts_by_update']] != [r['source_counts'] for r in logs]:
        raise ValueError('Actual source exposure differs from its full without-replacement orders')
    for actual, expected in zip(orders, reg['expected_orders']):
        for k, v in expected.items():
            if actual[k] != v: raise ValueError('Actual frozen epoch-order receipt differs: '+k)
    exposed = Counter(); cumulative = 0
    for row in logs:
        if (row['loss_method'] != METHOD or row['reward'] is not None
                or any(not math.isfinite(row[k]) for k in ('loss', 'ce', 'lr', 'grad_norm'))):
            raise ValueError('Finite actual SFT optimizer observations required')
        counts = row['source_counts']; exposed.update(counts); cumulative += sum(counts.values())
        if row['records_seen'] != cumulative or not set(counts) <= set(MASS): raise ValueError('Actual source/exposure counter differs')
        weighted = lambda name: sum(counts[s]*reg['course_loss_weights'][s]*row['source_loss_components'][s][name] for s in counts)/sum(counts.values())
        if (abs(weighted('objective')-row['loss']) > 1e-5 or abs(weighted('standard_target_ce')-row['ce']) > 1e-5
                or abs(weighted('signed_objective_minus_standard_ce')-row['signed_objective_adjustment']) > 1e-5
                or abs(weighted('new_route_objective')-row['new_route_objective']) > 1e-5):
            raise ValueError('Actual objective/CE/signed/new-route journal differs')
        for source, values in row['source_loss_components'].items():
            if set(values) != set(COMPONENTS) or any(not math.isfinite(v) for v in values.values()):
                raise ValueError('Finite complete actual loss decomposition required')
            if abs(values['objective']-values['standard_target_ce']-values['signed_objective_minus_standard_ce']) > 1e-5:
                raise ValueError('Retained objective differs from CE plus signed diagnostic')
            expected_route = values['objective'] if source in NEW_SOURCES else 0.
            if abs(values['new_route_objective']-expected_route) > 1e-5:
                raise ValueError('Actual new-route objective diagnostic differs')
    counts = Counter(r['questions']['action']['src'] for r in train)
    if cumulative != 182494 or exposed != Counter({s: 2*n for s, n in counts.items()}): raise ValueError('Full two-epoch data exposure failed')
    final = fit/'training-run/checkpoints/step-002500'
    state = torch.load(final/'trainer_state.pt', map_location='cpu', weights_only=False)
    if (state['optimizer_step'] != 2500 or state['epoch'] != 2 or state['next_microbatch'] != 0
            or state['records_seen'] != 182494 or len(state['optimizer']['state']) != 376
            or any(float(v['step']) != 2500 for v in state['optimizer']['state'].values())
            or state['world_size'] != 1 or sorted(state['order'].tolist()) != list(range(91247))
            or hashlib.sha256(json.dumps(state['order'].tolist()).encode()).hexdigest() != orders[-1]['order_sha256']):
        raise ValueError('Actual final optimizer/order state differs')
    del state; gc.collect()
    curve = read(fit/'validation-curve.json'); journal = rows(fit/'metrics/validation_metrics.jsonl')
    if [p['step'] for p in curve] != c['validation_and_game_steps'] or [p['local_step'] for p in journal] != c['validation_and_game_steps']:
        raise ValueError('Every actual registered validation point required')
    for point, event in zip(curve, journal):
        predictions = read(fit/'training-run/training_eval'/f"step-{point['step']:06d}"/'development/rows.json')
        actual = profile(validation, predictions, {}, target_cross_entropy)
        if any(point[k] != v for k, v in actual.items()): raise ValueError('Actual cached validation reconstruction differs')
        if event['loss'] != actual['weighted_class_objective'] or event['ce'] != actual['weighted_target_ce']:
            raise ValueError('Real validation curve/journal differs')
    identity = bundle(final); client = UnifiedPolicyClient(load_model(final))
    saved = {r['id']: r for r in read(fit/'training-run/training_eval/step-002500/development/rows.json')}; error = 0.
    for row in validation:
        actual = client.system_one(**api_request(row))['answers']['action']['probabilities']; previous = saved[row['_meta']['id']]
        if set(actual) != set(previous['keys']): raise ValueError('Final reloaded menu differs')
        error = max(error, max(abs(actual[k]-p) for k, p in zip(previous['keys'], previous['p'])))
    if error >= 5e-5: raise ValueError('Final full24516 serving reload parity failed')
    del client; gc.collect(); torch.cuda.empty_cache()
    contracts = read(fit/'policy-contracts-private.json'); games = {}; checked = 0
    labels = ['reference-parent', 'reference-original-guarded', 'reference-v8']+[f'step-{s:06d}' for s in c['validation_and_game_steps']]
    for label in labels:
        folder = fit/'evaluations'/label; protocol = read(folder/'protocol.json'); records = rows(folder/'matches-private.jsonl')
        groups = 8 if label.startswith('reference-') or label == 'step-002500' else 2
        grid = {(s, p, seat) for s in range(c['curve_seed_start'], c['curve_seed_start']+groups) for p in c['opponent_profiles'] for seat in range(4)}
        if (len(records) != len(grid) or {(r['seed'], r['profile'], r['seat']) for r in records} != grid
                or protocol['profiles'] != c['opponent_profiles'] or protocol['seats'] != 4): raise ValueError('Complete exact native development grid differs')
        for row in records:
            seat = row['seat']; log = row['mjai_log']; ledger = settlement_ledger(log, row['final_scores'])
            if (not row['completed'] or not row['ledger_passed'] or ledger['hands'] != row['hands']
                    or row['net_points'] != row['final_scores'][seat]-25000
                    or row['rank'] != sorted(range(4), key=lambda p: (-row['final_scores'][p], p)).index(seat)+1
                    or row['wins'] != sum(e['type'] == 'hora' and e.get('actor') == seat for e in log)
                    or row['dealin_hands'] != hand_dealin_count(log, seat)):
                raise ValueError('Native full-match accounting differs')
            for operation, n in row['choices'].items():
                if n != sum(e['type'] == operation and e.get('actor') == seat for e in log): raise ValueError('Actual operation event counts differ')
        actual = aggregate_complete_matches(records); stored = read(folder/'summary.json')
        if any(stored[k] != v for k, v in actual.items()): raise ValueError('Cached native summary differs')
        if label.startswith('step-'):
            step = int(label[5:]); model = bundle(fit/'training-run/checkpoints'/label); spec, policy_id = policy_contract(model, reg)
            if (protocol['checkpoint'] != model or protocol['policy'] != spec or protocol['policy_identity'] != policy_id
                    or contracts[str(step)] != dict(model_identity=model['identity_sha256'], policy=spec, policy_identity=policy_id)):
                raise ValueError('Actual child checkpoint/policy contract differs')
        if label == 'step-002500' and stored['curve_grid_subset'] != aggregate_complete_matches([r for r in records if r['seed'] < c['curve_seed_start']+2]):
            raise ValueError('Final same16 subset differs')
        games[label] = records; checked += len(records)
    parent_map = {(r['seed'], r['profile'], r['seat']): r for r in games['reference-parent']}
    if any(r != parent_map[r['seed'], r['profile'], r['seat']] for r in games['step-000000']):
        raise ValueError('Actual parent16 baseline events differ')
    comparisons = {}
    for label in ('reference-parent', 'reference-original-guarded', 'reference-v8'):
        reference = {(r['seed'], r['profile'], r['seat']): r for r in games[label]}; values = {}
        for field, fn in [('net_points', lambda r: r['net_points']), ('top2', lambda r: float(r['rank'] <= 2)),
            ('first', lambda r: float(r['rank'] == 1)), ('fourth', lambda r: float(r['rank'] == 4)),
            ('hand_win', lambda r: r['wins']/r['hands']), ('dealin', lambda r: r['dealin_hands']/r['hands'])]:
            clusters = [statistics.mean(fn(r)-fn(reference[r['seed'], r['profile'], r['seat']]) for r in games['step-002500'] if r['seed'] == seed)
                for seed in range(c['curve_seed_start'], c['curve_seed_start']+8)]
            mean = statistics.mean(clusters); radius = 2.3646242510102993*statistics.stdev(clusters)/math.sqrt(8)
            values[field] = dict(gain=mean, ci95=[mean-radius, mean+radius], seed_groups=8)
        comparisons[label] = values
    limits = c['development_targets']; retained = retention(curve[0], curve[-1], dict(
        max_plain_agreement_drop_from_step0=limits['max_plain_agreement_drop_from_step0'],
        max_riichi_balanced_accuracy_drop_from_step0=limits['max_RI_balanced_accuracy_drop_from_step0']))
    call_gain = statistics.mean(balanced(curve[-1]['original_calls_operation_type_counts'][op])-balanced(curve[0]['original_calls_operation_type_counts'][op])
        for op in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan'))
    mc_gain = statistics.mean(curve[0]['stratum_ce']['curriculum3:mc:'+p]-curve[-1]['stratum_ce']['curriculum3:mc:'+p]
        for p in ('closed_discard', 'open_discard', 'call_window', 'kan_window'))
    yaku_ce_gain = statistics.mean(curve[0]['stratum_ce'][s]-curve[-1]['stratum_ce'][s] for s in NEW_SOURCES)
    conditional = audit_conditional(fit, c, reg, bundle)
    passed = (retained['passed'] and call_gain >= -limits['max_original_five_call_balanced_accuracy_drop_from_step0']
        and mc_gain >= -limits['max_equal_phase_MC_CE_regression_from_step0']
        and yaku_ce_gain >= limits['min_six_source_mean_CE_improvement_from_step0']
        and all(comparisons['reference-parent'][field]['gain'] >= 0 for field in ('net_points', 'top2'))
        and comparisons['reference-parent']['dealin']['gain'] <= limits['max_final64_dealin_increase_over_parent'])
    for ext in ('png', 'pdf'):
        if (fit/'metrics'/f'curriculum-curves.{ext}').stat().st_size < 1000: raise ValueError('Actual real curves missing')
    if read(fit/'metrics/plot-data-provenance.json')['curve_steps'] != c['validation_and_game_steps']:
        raise ValueError('Actual matched-grid plot provenance differs')
    for p, h in reg['inputs'].items():
        if sha(p) != h: raise ValueError('Frozen input changed during final audit')
    if bundle(final) != identity: raise ValueError('Audit changed actual weights')
    result = dict(status='yaku2500_final_evidence_audit_passed', loss_method=METHOD, optimizer_steps=2500, actual_records_seen=182494,
        actual_source_exposures=dict(exposed), exact_parent_tensors=exact, full_parent_prediction_parity=parity,
        actual_complete_legal_parent16_events_exact=True, all376_optimizer_states_match=True,
        saved_final_reloaded_menus=24516, saved_final_max_probability_error=error, checked_complete_games=checked,
        final_identity=identity['identity_sha256'], final_policy_identity=policy_contract(identity, reg)[1],
        final_summary=aggregate_complete_matches(games['step-002500']), comparisons=comparisons,
        conditional_yaku=conditional, equal_six_source_CE_gain=yaku_ce_gain,
        basic_retention=retained, original_five_call_balanced_gain=call_gain, equal_phase_masked_MC_CE_gain=mc_gain,
        stage3B_advancement=dict(passed=passed, development_only_not_independent_strength=True),
        source_and_data_hashes_unchanged=True, source_split_replay_and_exposure_exact=True,
        final_reserved_grid_unused=True, automatic_stage4_started=False, automatic_release=False)
    write(out, result); print(json.dumps(result), flush=True)


def audit_conditional(fit, c, reg, bundle):
    from tools.curriculum_endgame_ledger import ledger
    results = {}; cases = {}
    folders = {'step-000000': fit/'conditional-yaku/step-000000',
        'step-002500': fit/'conditional-yaku/step-002500',
        'reference-v8': ROOT/c['conditional_reference_cache']}
    for label, folder in folders.items():
        saved = read(folder/'registered-private.json'); result = read(folder/'summary-private.json')
        binding = read(folder/'completion-binding-private.json')
        if (result['status'] != 'actual_fixed_validation_yaku92_first_hand_benchmark_complete'
                or result['completed_cases'] != 92 or result['source_families'] != 45
                or not result['model_parameters_unchanged'] or result['optimizer_updates'] != 0):
            raise ValueError('Actual complete fixed92 conditional run required')
        if any(sha(p) != h for p, h in binding['files'].items()):
            raise ValueError('Bound conditional benchmark inputs changed')
        if label.startswith('step-'):
            identity = bundle(fit/'training-run/checkpoints'/label)
            if (saved['checkpoint'] != identity or result['checkpoint_identity'] != identity['identity_sha256']
                    or result['policy_method'] != 'complete_legal_fixed_RI_class'):
                raise ValueError('Actual conditional candidate model/route differs')
        elif result['checkpoint_identity'] != c['reference_identity'] or result['policy_method'] != 'original_policy_agent':
            raise ValueError('Frozen conditional v8 changed')
        records = rows(folder/'case-outcomes-private.jsonl')
        ids = {r['source_case_id']: r for r in records}
        if len(ids) != 92 or len(records) != 92 or len({r['group_id'] for r in records}) != 45:
            raise ValueError('Conditional original92/45 coverage differs')
        for row in records:
            log = row['actual_mjai']; financial = ledger(log, row['final_scores'])
            if (not row['completed_first_hand'] or financial != row['ledger']
                    or row['first_hand_net_points'] != row['final_scores'][0]-25000
                    or row['observer_won'] != any(e['type'] == 'hora' and e['actor'] == 0 for e in log)
                    or row['observer_dealt_in'] != any(e['type'] == 'hora' and e['actor'] != 0 and e['target'] == 0 for e in log)):
                raise ValueError('Conditional actual settlement accounting differs')
        for route, stat in result['by_route'].items():
            subset = [r for r in records if r['route'] == route]
            if (stat['cases'] != len(subset) or stat['observer_wins'] != sum(r['observer_won'] for r in subset)
                    or stat['observer_dealins'] != sum(r['observer_dealt_in'] for r in subset)
                    or stat['first_hand_net_points_sum'] != sum(r['first_hand_net_points'] for r in subset)):
                raise ValueError('Conditional route aggregates differ from actual outcomes')
        cases[label] = ids; results[label] = result['by_route']
    parent = cases['step-000000']; child = cases['step-002500']
    if any(set(v) != set(parent) for v in cases.values()):
        raise ValueError('Conditional source cases differ across models')
    if any((parent[k]['group_id'], parent[k]['route']) != (child[k]['group_id'], child[k]['route']) for k in parent):
        raise ValueError('Conditional source family or route changed')
    return dict(by_model=results, source_cases=92, source_families=45,
        child_minus_parent_wins=sum(child[k]['observer_won']-parent[k]['observer_won'] for k in parent),
        child_minus_parent_mean_first_hand_net=statistics.mean(
            child[k]['first_hand_net_points']-parent[k]['first_hand_net_points'] for k in parent),
        biased_first_hand_development_only_not_natural_match_strength=True,
        yaku_counts_not_complete_for_unsupported_scorer_cases=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); audit(a.registered.resolve(), a.out.resolve())
