"""Independent actual progression exposure, reload, validation and native audit."""
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
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_parent import rows, require_replay
from tools.curriculum_progression_parent_v1 import verify_parent
from tools.curriculum_progression_parent_runtime_v1 import require_registration, policy_contract, prediction_parity, require_step0
from tools.curriculum_progression_parent_fit_v1 import profile
from tools.curriculum_progression_parent_loss_v1 import METHOD, COMPONENTS, MASS
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
    if (done['status'] != 'progression2400_fit_and_real_curves_complete' or done['optimizer_steps'] != 2400
            or done['actual_records_seen'] != 175190): raise ValueError('Full actual registered fit required')
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
    parent_p = read(ROOT/parent_reg['curriculum']['out']/'training-run/training_eval/step-000388/development/rows.json')
    zero_p = read(fit/'training-run/training_eval/step-000000/development/rows.json')
    parity = prediction_parity(rows(pa['eval_data']), parent_p, validation, zero_p)
    if parity != warm['all_parent_prediction_parity']: raise ValueError('Actual full parent parity differs')
    game_proof = read(fit/'warmstart-games-private.json')
    if (game_proof['matches'] != 16 or not game_proof['exact_complete_legal_parent_events']
            or not game_proof['checked_before_first_update'] or not game_proof['original_guarded_policy_not_equated']):
        raise ValueError('Actual changed-policy parent baseline proof required')
    logs = rows(fit/'metrics/train_metrics.jsonl')
    if [r['local_step'] for r in logs] != list(range(1, 2401)): raise ValueError('All actual optimizer steps required')
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
                or abs(weighted('signed_objective_minus_standard_ce')-row['signed_objective_adjustment']) > 1e-5):
            raise ValueError('Actual objective/CE/signed adjustment journal differs')
        for source, values in row['source_loss_components'].items():
            if set(values) != set(COMPONENTS) or any(not math.isfinite(v) for v in values.values()):
                raise ValueError('Finite complete actual loss decomposition required')
            if abs(values['objective']-values['standard_target_ce']-values['signed_objective_minus_standard_ce']) > 1e-5:
                raise ValueError('Retained objective differs from CE plus signed diagnostic')
    counts = Counter(r['questions']['action']['src'] for r in train)
    if cumulative != 175190 or exposed != Counter({s: 2*n for s, n in counts.items()}): raise ValueError('Full two-epoch data exposure failed')
    final = fit/'training-run/checkpoints/step-002400'
    state = torch.load(final/'trainer_state.pt', map_location='cpu', weights_only=False)
    if (state['optimizer_step'] != 2400 or state['epoch'] != 2 or state['next_microbatch'] != 0
            or state['records_seen'] != 175190 or len(state['optimizer']['state']) != 376
            or any(float(v['step']) != 2400 for v in state['optimizer']['state'].values())
            or state['world_size'] != 1 or sorted(state['order'].tolist()) != list(range(87595))
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
    saved = {r['id']: r for r in read(fit/'training-run/training_eval/step-002400/development/rows.json')}; error = 0.
    for row in validation:
        actual = client.system_one(**api_request(row))['answers']['action']['probabilities']; previous = saved[row['_meta']['id']]
        if set(actual) != set(previous['keys']): raise ValueError('Final reloaded menu differs')
        error = max(error, max(abs(actual[k]-p) for k, p in zip(previous['keys'], previous['p'])))
    if error >= 5e-5: raise ValueError('Final full23688 serving reload parity failed')
    del client; gc.collect(); torch.cuda.empty_cache()
    contracts = read(fit/'policy-contracts-private.json'); games = {}; checked = 0
    labels = ['reference-parent', 'reference-original-guarded', 'reference-v8']+[f'step-{s:06d}' for s in c['validation_and_game_steps']]
    for label in labels:
        folder = fit/'evaluations'/label; protocol = read(folder/'protocol.json'); records = rows(folder/'matches-private.jsonl')
        groups = 8 if label.startswith('reference-') or label == 'step-002400' else 2
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
        if label == 'step-002400' and stored['curve_grid_subset'] != aggregate_complete_matches([r for r in records if r['seed'] < c['curve_seed_start']+2]):
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
            clusters = [statistics.mean(fn(r)-fn(reference[r['seed'], r['profile'], r['seat']]) for r in games['step-002400'] if r['seed'] == seed)
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
    passed = (retained['passed'] and call_gain >= -limits['max_original_five_call_balanced_accuracy_drop_from_step0']
        and mc_gain >= limits['min_equal_phase_MC_CE_improvement_from_step0']
        and all(comparisons[label][field]['gain'] >= 0 for label in ('reference-parent', 'reference-original-guarded') for field in ('net_points', 'top2'))
        and comparisons['reference-original-guarded']['dealin']['gain'] <= limits['max_final64_dealin_increase_over_original_guarded_parent'])
    for ext in ('png', 'pdf'):
        if (fit/'metrics'/f'curriculum-curves.{ext}').stat().st_size < 1000: raise ValueError('Actual real curves missing')
    if read(fit/'metrics/plot-data-provenance.json')['curve_steps'] != c['validation_and_game_steps']:
        raise ValueError('Actual matched-grid plot provenance differs')
    for p, h in reg['inputs'].items():
        if sha(p) != h: raise ValueError('Frozen input changed during final audit')
    if bundle(final) != identity: raise ValueError('Audit changed actual weights')
    result = dict(status='progression2400_final_evidence_audit_passed', optimizer_steps=2400, actual_records_seen=175190,
        actual_source_exposures=dict(exposed), exact_parent_tensors=exact, full_parent_prediction_parity=parity,
        actual_complete_legal_parent16_events_exact=True, all376_optimizer_states_match=True,
        saved_final_reloaded_menus=23688, saved_final_max_probability_error=error, checked_complete_games=checked,
        final_identity=identity['identity_sha256'], final_policy_identity=policy_contract(identity, reg)[1],
        final_summary=aggregate_complete_matches(games['step-002400']), comparisons=comparisons,
        basic_retention=retained, original_five_call_balanced_gain=call_gain, equal_phase_masked_MC_CE_gain=mc_gain,
        stage3_advancement=dict(passed=passed, development_only_not_independent_strength=True),
        source_and_data_hashes_unchanged=True, source_split_replay_and_exposure_exact=True,
        final_reserved_grid_unused=True, automatic_stage4_started=False, automatic_release=False)
    write(out, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); audit(a.registered.resolve(), a.out.resolve())
