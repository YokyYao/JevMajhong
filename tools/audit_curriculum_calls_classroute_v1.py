"""Independent full-exposure, selector/metric/reload and native-match audit."""
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
from tools.curriculum_parent import rows, require_replay
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_calls_classroute_parent_v1 import verify_parent
from tools.curriculum_calls_classroute_v1 import METHOD, profile, require_recipe, warmstart_parity, COMPONENTS
from tools.curriculum_calls_fit import balanced, retention
from tools.evaluate_curriculum_calls_classroute_v1 import policy_contract


def audit(registered, out):
    gpu_only(); reg = read(registered); c = reg['curriculum']; fit = ROOT/c['out']
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from safetensors.torch import load_file
    from jevmahjong.training.initialization import exact_loaded_tensors
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevany.data import api_request
    for p, h in reg['inputs'].items(): assert sha(p) == h, p
    parent = verify_parent(reg['training_config']); data = read(c['data_ready']); require_recipe(c, data)
    done = read(fit/'completion.json')
    assert done['status'] == 'stage2_calls_classroute_fit_and_curves_complete'
    assert done['optimizer_steps'] == 388 and done['actual_records_seen'] == 98044
    init = read(fit/'initialization-audit.json'); proof = read(fit/'actual-update-proof.json')
    assert init['parent_adapter_exact'] and init['parent_head_exact'] and init['new_optimizer']
    assert init['old_adapters_loaded'] and not init['pretrained_base_requires_grad'] and init['external_residual_networks'] == 0
    assert init['parent_identity_sha256'] == c['parent_identity']
    assert proof['continued_LoRA_parameters_changed'] and proof['head_parameters_changed'] and not proof['cached_encoder_features_used']
    zero = fit/'training-run/checkpoints/step-000000'
    exact = exact_loaded_tensors(load_file(zero/'adapter_model.safetensors'),
        torch.load(zero/'head.pt', map_location='cpu', weights_only=True)['head'], parent['model']['directory'])
    assert exact['adapter_tensors'] == 372 and exact['head_tensors'] == 4
    warm = read(fit/'warmstart-validation-private.json')
    assert warm['exact_tensors'] == exact and warm['checked_before_first_optimizer_update']
    assert read(fit/'warmstart-games-private.json') == dict(matches=16, exact_parent_event_outcomes=True,
        checked_before_first_update=True, parent_policy_identity=c['parent_policy_identity'])
    train = rows(ROOT/data['files']['train']['path']); validation = rows(ROOT/data['files']['validation']['path'])
    sealed = read(c['parent_seal']); parent_reg = read(sealed['parent_registration'])
    pa = tomllib.loads(Path(parent_reg['training_config']['trainer_config']).read_text())
    require_replay(rows(pa['data']), train); require_replay(rows(pa['eval_data']), validation)
    previous = read(ROOT/parent_reg['curriculum']['out']/'training-run/training_eval/step-000608/development/rows.json')
    initial = read(fit/'training-run/training_eval/step-000000/development/rows.json')
    parity = warmstart_parity(rows(pa['eval_data']), previous, validation, initial)
    assert parity == warm['basic_validation_prediction_and_choice_parity']
    logs = rows(fit/'metrics/train_metrics.jsonl'); assert [r['local_step'] for r in logs] == list(range(1, 389))
    orders = [read(fit/'epoch-order-private'/f'epoch-{i:02d}.json') for i in (1, 2)]
    assert [item for order in orders for item in order['source_counts_by_update']] == [r['source_counts'] for r in logs]
    for order in orders:
        assert order['records'] == 49022 and order['updates'] == 194 and order['copies_added'] == 0
        assert order['each_record_exactly_once'] and order['all_updates_have_all_declared_strata']
    exposed = Counter()
    for row in logs:
        assert row['loss_method'] == METHOD and row['reward'] is None
        assert all(math.isfinite(row[k]) for k in ('loss', 'ce', 'lr', 'grad_norm'))
        counts = row['source_counts']; assert set(counts) == set(reg['course_loss_mass'])
        weighted = lambda field: sum(counts[s]*reg['course_loss_weights'][s]*row[field][s] for s in counts)/sum(counts.values())
        assert abs(weighted('source_objective')-row['loss']) < 1e-5
        assert abs(weighted('source_ce')-row['ce']) < 1e-5
        for source, parts in row['source_loss_components'].items():
            assert set(parts) == set(COMPONENTS) and all(math.isfinite(v) for v in parts.values())
            assert abs(parts['objective']-sum(parts[k] for k in ('riichi_max_class_ce', 'conditional_discard_ce',
                'plain_action_ce', 'advanced_action_ce', 'post_reach_action_ce'))) < 1e-5
        exposed.update(counts)
    assert exposed == Counter({s: 2*n for s, n in data['categories']['train'].items()})
    state = torch.load(fit/'training-run/checkpoints/step-000388/trainer_state.pt', map_location='cpu', weights_only=False)
    assert state['optimizer_step'] == 388 and state['epoch'] == 2 and state['next_microbatch'] == 0
    assert state['records_seen'] == 98044 and len(state['optimizer']['state']) == 376 and state['world_size'] == 1
    assert all(float(v['step']) == 388 for v in state['optimizer']['state'].values())
    assert sorted(state['order'].tolist()) == list(range(49022))
    assert hashlib.sha256(json.dumps(state['order'].tolist()).encode()).hexdigest() == orders[-1]['order_sha256']
    del state; gc.collect()
    curve = read(fit/'validation-curve.json'); assert [p['step'] for p in curve] == [0, 194, 388]
    journal = rows(fit/'metrics/validation_metrics.jsonl'); assert [p['local_step'] for p in journal] == [0, 194, 388]
    for point, event in zip(curve, journal):
        predictions = read(fit/'training-run/training_eval'/f"step-{point['step']:06d}"/'development/rows.json')
        actual = profile(validation, predictions, reg['course_loss_mass'], {'weighted_target_ce': point['phase_weighted_ce']}, target_cross_entropy)
        for name, value in actual.items():
            if name != 'phase_weighted_ce': assert point[name] == value, name
        assert event['loss'] == point['weighted_class_objective'] and event['ce'] == point['weighted_target_ce']
        for name in ('raw_basic_replay', 'basic_replay', 'raw_basic_source_quality', 'basic_replay_source_quality',
                     'raw_operation_type_counts', 'operation_type_counts', 'source_loss_components'):
            assert event[name] == point[name]
    assert curve[0]['basic_replay_source_quality']['all_source_quality_passed']
    checkpoint = fit/'training-run/checkpoints/step-000388'; client = UnifiedPolicyClient(load_model(checkpoint))
    saved = {r['id']: r for r in read(fit/'training-run/training_eval/step-000388/development/rows.json')}; error = 0.
    for row in validation:
        actual = client.system_one(**api_request(row))['answers']['action']['probabilities']; previous = saved[row['_meta']['id']]
        assert set(actual) == set(previous['keys'])
        error = max(error, max(abs(actual[k]-p) for k, p in zip(previous['keys'], previous['p'])))
    assert error < 5e-5; del client; gc.collect(); torch.cuda.empty_cache()
    contracts = read(fit/'policy-contracts-private.json'); games = {}; checked = 0
    for label in ('reference-v8', 'reference-parent', 'step-000000', 'step-000194', 'step-000388'):
        folder = fit/'evaluations'/label; protocol = read(folder/'protocol.json'); match_rows = rows(folder/'matches-private.jsonl')
        groups = 8 if label in ('reference-v8', 'reference-parent', 'step-000388') else 2
        seeds = list(range(c['curve_seed_start'], c['curve_seed_start']+groups))
        assert protocol['seeds'] == seeds and protocol['profiles'] == c['opponent_profiles'] and protocol['seats'] == 4
        grid = {(seed, p, seat) for seed in seeds for p in c['opponent_profiles'] for seat in range(4)}
        assert len(match_rows) == len(grid) and {(r['seed'], r['profile'], r['seat']) for r in match_rows} == grid
        for row in match_rows:
            assert row['completed'] and row['ledger_passed']; seat = row['seat']
            ledger = settlement_ledger(row['mjai_log'], row['final_scores']); assert ledger['hands'] == row['hands']
            assert row['net_points'] == row['final_scores'][seat]-25000
            assert row['rank'] == sorted(range(4), key=lambda p: (-row['final_scores'][p], p)).index(seat)+1
            assert hand_dealin_count(row['mjai_log'], seat) == row['dealin_hands']
            assert sum(e['type'] == 'hora' and e.get('actor') == seat for e in row['mjai_log']) == row['wins']
            for operation, count in row['choices'].items():
                assert sum(e['type'] == operation and e.get('actor') == seat for e in row['mjai_log']) == count
        stored = read(folder/'summary.json'); actual = aggregate_complete_matches(match_rows)
        assert all(stored[k] == v for k, v in actual.items())
        identity = c['reference_identity'] if label == 'reference-v8' else c['parent_identity'] if label == 'reference-parent' else bundle(fit/'training-run/checkpoints'/label)['identity_sha256']
        assert stored['checkpoint_sha256'] == protocol['checkpoint']['identity_sha256'] == identity
        if label != 'reference-v8':
            spec, policy_id = policy_contract(protocol['checkpoint'], reg)
            assert protocol['policy'] == spec and protocol['policy_identity'] == stored['policy_identity'] == policy_id
            if label.startswith('step-'):
                assert contracts[str(int(label.split('-')[1]))] == dict(model_identity=identity, policy=spec, policy_identity=policy_id)
        if groups == 8: assert stored['curve_grid_subset'] == aggregate_complete_matches([r for r in match_rows if r['seed'] < seeds[0]+2])
        games[label] = match_rows; checked += len(match_rows)
    parent_map = {(r['seed'], r['profile'], r['seat']): r for r in games['reference-parent']}
    assert all(r == parent_map[r['seed'], r['profile'], r['seat']] for r in games['step-000000'])
    comparisons = {}
    for name in ('reference-parent', 'reference-v8'):
        reference = {(r['seed'], r['profile'], r['seat']): r for r in games[name]}; values = {}
        for field, fn in [('net_points', lambda r: r['net_points']), ('top2', lambda r: float(r['rank'] <= 2)),
                          ('hand_win', lambda r: r['wins']/r['hands']), ('dealin', lambda r: r['dealin_hands']/r['hands']),
                          ('first', lambda r: float(r['rank'] == 1)), ('fourth', lambda r: float(r['rank'] == 4))]:
            clusters = [statistics.mean(fn(r)-fn(reference[r['seed'], r['profile'], r['seat']])
                for r in games['step-000388'] if r['seed'] == seed) for seed in sorted({r['seed'] for r in games['step-000388']})]
            mean = statistics.mean(clusters); radius = 2.3646242510102993*statistics.stdev(clusters)/math.sqrt(8)
            values[field] = dict(gain=mean, ci95=[mean-radius, mean+radius], seed_groups=8)
        comparisons[name] = values
    retained = retention(curve[0], curve[-1], c['development_targets'])
    call_gain = statistics.mean(balanced(curve[-1]['operation_type_counts'][op])-balanced(curve[0]['operation_type_counts'][op])
                               for op in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan'))
    passed = (retained['passed'] and call_gain >= c['development_targets']['min_five_call_type_balanced_accuracy_gain_from_step0']
        and comparisons['reference-parent']['net_points']['gain'] >= 0
        and comparisons['reference-parent']['dealin']['gain'] <= .02)
    for ext in ('png', 'pdf'): assert (fit/'metrics'/f'curriculum-curves.{ext}').stat().st_size > 1000
    assert read(fit/'metrics/plot-data-provenance.json')['curve_steps'] == [0, 194, 388]
    for p, h in reg['inputs'].items(): assert sha(p) == h, p
    result = dict(status='stage2_calls_classroute_final_evidence_audit_passed', optimizer_steps=388,
        real_records_seen=98044, actual_source_exposures=dict(exposed), exact_parent_tensors=exact,
        actual_parent_probability_and_choice_parity=parity, actual_parent16_full_events_exact=True,
        all376_optimizer_states_match=True, saved_final_reloaded_menus=len(validation), saved_final_max_probability_error=error,
        checked_complete_games=checked, final_summary=aggregate_complete_matches(games['step-000388']),
        comparisons=comparisons, source_specific_basic_retention=retained, five_call_type_balanced_gain=call_gain,
        stage2_advancement=dict(passed=passed, development_only_not_strength_proof=True),
        source_and_data_hashes_unchanged=True, exact_train_validation_replay=True,
        original_parent_raw_source_quality=sealed['original_raw_source_quality'],
        original_parent_selected_source_quality=sealed['stage_quality'],
        independent_han_fu_scoring=False, final_reserved_grid_unused=True,
        automatic_stage3_started=False, automatic_release=False)
    write(out, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); audit(a.registered.resolve(), a.out.resolve())
