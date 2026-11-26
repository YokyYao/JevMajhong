"""Independent weighted replay mapping, optimizer, validation and native audit."""
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
from tools.curriculum_ri_anchor_runtime_v1 import verify_parent
from tools.curriculum_ri_anchor_runtime_v1 import require_registration,policy_contract,prediction_parity,require_step0,require_launch,profile
from tools.curriculum_progression_parent_loss_v1 import MASS
from tools.curriculum_ri_parent_anchor_weighted_v1 import COMPONENTS
from tools.curriculum_ri_parent_anchor_weighted_v1 import METHOD
from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay
from tools.curriculum_ri_every_update_replay_v1 import RI
from tools.curriculum_calls_fit import retention, balanced


def audit(registered, out):
    gpu_only(); reg = read(registered); require_registration(reg); require_launch(registered); c = reg['curriculum']; fit = ROOT/c['out']
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
    if (done['status'] != 'ri_anchor2596_fit_and_real_curves_complete' or done['optimizer_steps'] != 2596
            or done['actual_records_seen'] != 189508): raise ValueError('Full actual registered fit required')
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
    if [r['local_step'] for r in logs] != list(range(1, 2597)): raise ValueError('All actual optimizer steps required')
    plans = [read(p) for p in reg['replay_epoch_plans']]
    replay = EpochReplay(train, plans)  # Independently validate every per-row empirical mass.
    expected_hashes = []
    expected_groups = []
    for plan in plans:
        hasher = hashlib.sha256()
        for i in plan['indices']:
            hasher.update(json.dumps([i, plan['weights_by_original_index'][str(i)]], separators=(',', ':')).encode()+b'\n')
        expected_hashes.append(hasher.hexdigest())
        for start in range(0, 94754, 73):
            group = plan['indices'][start:start+73]
            counts = Counter(train[i]['questions']['action']['src'] for i in group)
            mass = {s: sum(plan['weights_by_original_index'][str(i)] for i in group
                if train[i]['questions']['action']['src'] == s)/73 for s in counts}
            expected_groups.append((dict(counts), mass))
    mapping = read(fit/'actual-replay-mapping-private.json')
    if (mapping['epochs'] != 2 or mapping['physical_exposures'] != 189508
            or mapping['unique_rows_each_epoch'] != [92570, 92570]
            or mapping['actual_mapping_weight_hashes'] != expected_hashes
            or mapping['source_counts_by_epoch'] != [dict(Counter(train[i]['questions']['action']['src']
                for i in p['indices'])) for p in plans]):
        raise ValueError('Actual epoch-specific encoded original/weight stream differs')
    admission = read(fit/'token-admission.json')
    if (admission['admitted_records'] != 92570 or admission['rejected_records'] != 0
            or admission['physical_exposure_slots'] != 94754
            or admission['physical_expansion_after_unique_token_admission'] is not True):
        raise ValueError('Real unique token admission and physical expansion required')
    canonical=read(fit/'canonical-request-admission-private.json')
    if (canonical['unique_rows']!=92570 or not canonical['all_nonmetadata_fields_exact']
            or not canonical['every_existing_metadata_field_exact']
            or not canonical['strict_full_canonical_object_equality_passed']
            or not canonical['parent_targets_attached_after_actual_encoding']):
        raise ValueError('Full canonical input identity and post-encoding parent target attachment required')
    exposed = Counter(); cumulative = 0
    for row, (expected_counts, expected_mass) in zip(logs, expected_groups):
        if (row['loss_method'] != METHOD or row['reward'] is not None
                or any(not math.isfinite(row[k]) for k in ('loss', 'ce', 'lr', 'grad_norm'))):
            raise ValueError('Finite actual SFT optimizer observations required')
        counts = row['source_counts']; exposed.update(counts); cumulative += sum(counts.values())
        if (counts != expected_counts or any(counts.get(s,0) < 1 for s in RI)
                or set(row['effective_source_mass']) != set(expected_mass)
                or any(abs(row['effective_source_mass'][s]-v) > 1e-12 for s, v in expected_mass.items())):
            raise ValueError('Actual per-update source counts or inverse-repeat weight mass differs')
        if row['records_seen'] != cumulative or not set(counts) <= set(MASS): raise ValueError('Actual source/exposure counter differs')
        contributions = row['source_weighted_component_contributions']
        if set(contributions) != set(counts): raise ValueError('Missing actual weighted source contributions')
        weighted = lambda name: sum(contributions[s][name] for s in counts)
        if (abs(weighted('objective')-row['loss']) > 1e-5 or abs(weighted('standard_target_ce')-row['ce']) > 1e-5
                or abs(weighted('signed_objective_minus_standard_ce')-row['signed_objective_adjustment']) > 1e-5):
            raise ValueError('Actual objective/CE/signed adjustment journal differs')
        if (abs(row['loss']-row['retained_objective_without_anchor']-row['parent_RI_anchor_adjustment'])>1e-5
                or abs(weighted('parent_RI_anchor_adjustment')-row['parent_RI_anchor_adjustment'])>1e-5
                or abs(row['parent_RI_anchor_adjustment']-4*row['parent_RI_KL'])>1e-5):
            raise ValueError('Actual parent-KL contribution and optimized objective differ')
        for source, values in row['source_loss_components'].items():
            if set(values) != set(COMPONENTS) or any(not math.isfinite(v) for v in values.values()):
                raise ValueError('Finite complete actual loss decomposition required')
            if abs(values['objective']-values['standard_target_ce']-values['signed_objective_minus_standard_ce']) > 1e-5:
                raise ValueError('Retained objective differs from CE plus signed diagnostic')
            if (values['parent_RI_KL'] < -1e-5 or abs(values['parent_RI_anchor_adjustment']-4*values['parent_RI_KL'])>1e-5
                    or source not in RI and (values['parent_RI_KL']!=0 or values['parent_RI_anchor_adjustment']!=0)):
                raise ValueError('Parent retention penalty entered an invalid source or changed strength')
    expected_exposed = Counter(train[i]['questions']['action']['src'] for p in plans for i in p['indices'])
    if cumulative != 189508 or exposed != expected_exposed: raise ValueError('Full two-epoch physical data exposure failed')
    final = fit/'training-run/checkpoints/step-002596'
    state = torch.load(final/'trainer_state.pt', map_location='cpu', weights_only=False)
    if (state['optimizer_step'] != 2596 or state['epoch'] != 2 or state['next_microbatch'] != 0
            or state['records_seen'] != 189508 or len(state['optimizer']['state']) != 376
            or any(float(v['step']) != 2596 for v in state['optimizer']['state'].values())
            or state['world_size'] != 1 or state['order'].tolist() != list(range(94754))):
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
    saved = {r['id']: r for r in read(fit/'training-run/training_eval/step-002596/development/rows.json')}; error = 0.
    for row in validation:
        actual = client.system_one(**api_request(row))['answers']['action']['probabilities']; previous = saved[row['_meta']['id']]
        if set(actual) != set(previous['keys']): raise ValueError('Final reloaded menu differs')
        error = max(error, max(abs(actual[k]-p) for k, p in zip(previous['keys'], previous['p'])))
    if error >= 5e-5: raise ValueError('Final full24998 serving reload parity failed')
    del client; gc.collect(); torch.cuda.empty_cache()
    contracts = read(fit/'policy-contracts-private.json'); games = {}; checked = 0
    labels = ['reference-parent', 'reference-original-guarded', 'reference-v8']+[f'step-{s:06d}' for s in c['validation_and_game_steps']]
    for label in labels:
        folder = fit/'evaluations'/label; protocol = read(folder/'protocol.json'); records = rows(folder/'matches-private.jsonl')
        groups = 8 if label.startswith('reference-') or label == 'step-002596' else 2
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
            if sum(ledger['components'][seat].values())!=row['net_points']:
                raise ValueError('Own score components must close exactly to terminal net')
            for operation, n in row['choices'].items():
                if n != sum(e['type'] == operation and e.get('actor') == seat for e in log): raise ValueError('Actual operation event counts differ')
        actual = aggregate_complete_matches(records); stored = read(folder/'summary.json')
        if any(stored[k] != v for k, v in actual.items()): raise ValueError('Cached native summary differs')
        if label.startswith('step-'):
            step = int(label[5:]); model = bundle(fit/'training-run/checkpoints'/label); spec, policy_id = policy_contract(model, reg)
            if (protocol['checkpoint'] != model or protocol['policy'] != spec or protocol['policy_identity'] != policy_id
                    or contracts[str(step)] != dict(model_identity=model['identity_sha256'], policy=spec, policy_identity=policy_id)):
                raise ValueError('Actual child checkpoint/policy contract differs')
        if label == 'step-002596' and stored['curve_grid_subset'] != aggregate_complete_matches([r for r in records if r['seed'] < c['curve_seed_start']+2]):
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
            clusters = [statistics.mean(fn(r)-fn(reference[r['seed'], r['profile'], r['seat']]) for r in games['step-002596'] if r['seed'] == seed)
                for seed in range(c['curve_seed_start'], c['curve_seed_start']+8)]
            mean = statistics.mean(clusters); radius = 2.3646242510102993*statistics.stdev(clusters)/math.sqrt(8)
            values[field] = dict(gain=mean, ci95=[mean-radius, mean+radius], seed_groups=8)
        comparisons[label] = values
    limits = c['development_targets']; original_curve=[p['original_validation'] for p in curve]
    retained = retention(original_curve[0], original_curve[-1], dict(
        max_plain_agreement_drop_from_step0=limits['max_plain_agreement_drop_from_step0'],
        max_riichi_balanced_accuracy_drop_from_step0=limits['max_RI_balanced_accuracy_drop_from_step0']))
    call_gain = statistics.mean(balanced(original_curve[-1]['original_calls_operation_type_counts'][op])-balanced(original_curve[0]['original_calls_operation_type_counts'][op])
        for op in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan'))
    mc_gain = statistics.mean(original_curve[0]['stratum_ce']['curriculum3:mc:'+p]-original_curve[-1]['stratum_ce']['curriculum3:mc:'+p]
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
    result = dict(status='ri_anchor2596_final_evidence_audit_passed', optimizer_steps=2596, actual_records_seen=189508,
        actual_source_exposures=dict(exposed), exact_parent_tensors=exact, full_parent_prediction_parity=parity,
        actual_complete_legal_parent16_events_exact=True, all376_optimizer_states_match=True,
        saved_final_reloaded_menus=24998, saved_final_max_probability_error=error, checked_complete_games=checked,
        final_identity=identity['identity_sha256'], final_policy_identity=policy_contract(identity, reg)[1],
        final_summary=aggregate_complete_matches(games['step-002596']), comparisons=comparisons,
        basic_retention=retained,retention_quality_cohort='original23688',
        new_teacher_validation=curve[-1]['new_teacher_validation'],
        original_five_call_balanced_gain=call_gain, equal_phase_masked_MC_CE_gain=mc_gain,
        stage3_advancement=dict(passed=passed, development_only_not_independent_strength=True),
        source_and_data_hashes_unchanged=True, source_split_replay_and_exposure_exact=True,
        final_reserved_grid_unused=True, automatic_stage4_started=False, automatic_release=False)
    write(out, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(); audit(a.registered.resolve(), a.out.resolve())
