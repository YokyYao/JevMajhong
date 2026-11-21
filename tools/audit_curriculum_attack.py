"""Audit the full mixed-target attack fit, reload and matched complete games."""
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_parent import read, rows, sha, require_replay
from tools.curriculum_calls_parent import verify_calls_parent, balanced
from tools.curriculum_attack_profile import profile


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def quality(c, first, final, comparison):
    plain = final['stratum_accuracy']['expert:plain']-first['stratum_accuracy']['expert:plain']
    riichi = balanced(final['riichi_type_counts'])-balanced(first['riichi_type_counts'])
    calls = statistics.mean(balanced(final['operation_type_counts'][op])-balanced(first['operation_type_counts'][op])
        for op in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan'))
    mc = statistics.mean(first['stratum_ce']['mc:'+p]-final['stratum_ce']['mc:'+p]
        for p in ('closed_discard', 'open_discard', 'call_window', 'kan_window'))
    net, dealin = [comparison[k]['gain'] for k in ('net_points', 'dealin')]
    if not all(math.isfinite(v) for v in (plain, riichi, calls, mc, net, dealin)):
        raise ValueError('Nonfinite attack quality metric')
    t = c['development_targets']
    return dict(plain_agreement_gain=plain, riichi_balanced_gain=riichi, five_call_balanced_gain=calls,
        MC_mean_CE_improvement=mc, final64_net_gain=net, final64_dealin_gain=dealin,
        passed=plain >= -t['max_plain_agreement_drop_from_step0']
            and riichi >= -t['max_riichi_balanced_drop_from_step0']
            and calls >= -t['max_five_call_balanced_drop_from_step0']
            and mc >= t['min_MC_mean_CE_improvement_from_step0']
            and net >= t['min_final64_net_gain_over_parent']
            and dealin <= t['max_final64_dealin_increase_over_parent'],
        development_only_not_strength_proof=True)


def paired_comparison(final, reference):
    key = lambda r: (r['seed'], r['profile'], r['seat'])
    lookup = {key(r): r for r in reference}
    actual = {key(r) for r in final}
    seeds = sorted({r['seed'] for r in final})
    if len(lookup) != len(reference) or len(actual) != len(final) or actual != set(lookup) or len(seeds) != 8:
        raise ValueError('Whole-seed paired final/reference grids must match')
    fields = dict(top2=lambda r: float(r['rank'] <= 2), first=lambda r: float(r['rank'] == 1),
        fourth=lambda r: float(r['rank'] == 4), net_points=lambda r: r['net_points'],
        hand_win=lambda r: r['wins']/r['hands'], dealin=lambda r: r['dealin_hands']/r['hands'])
    result = {}
    for name, fn in fields.items():
        values = [statistics.mean(fn(r)-fn(lookup[key(r)]) for r in final if r['seed'] == seed) for seed in seeds]
        mean = statistics.mean(values)
        radius = 2.3646242510102993*statistics.stdev(values)/math.sqrt(len(values))
        result[name] = dict(gain=mean, ci95=[mean-radius, mean+radius], seed_groups=len(values))
    return result


def audit(registered, out, binding=None):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-', 'login'))
    reg = read(registered)
    c = reg['curriculum']
    fit = ROOT/c['out']
    last = reg['registered_final_step']
    def verify():
        for p, h in reg['inputs'].items():
            assert sha(p) == h, p
        if binding:
            for p, h in read(binding).items():
                assert sha(p) == h, p
    verify()
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from safetensors.torch import load_file
    from jevmahjong.training.initialization import exact_loaded_tensors
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevany.data import api_request
    parent = verify_calls_parent(reg['training_config'])
    done = read(fit/'completion.json')
    assert done['status'] == 'stage3_fit_and_curves_complete' and done['optimizer_steps'] == last
    assert done['actual_records_seen'] == reg['train_records']*c['epochs']
    assert done['candidate_rule'] == c['candidate_rule'] and done['historical_minimum_CE_result_not_course_candidate']
    init, proof = [read(fit/name) for name in ('initialization-audit.json', 'actual-update-proof.json')]
    assert init['parent_adapter_exact'] and init['parent_head_exact'] and init['new_optimizer']
    assert init['old_adapters_loaded'] and not init['pretrained_base_requires_grad'] and init['external_residual_networks'] == 0
    assert init['parent_identity_sha256'] == c['parent_identity']
    assert proof['continued_LoRA_parameters_changed'] and proof['head_parameters_changed'] and not proof['cached_encoder_features_used']
    zero = fit/'training-run/checkpoints/step-000000'
    exact = exact_loaded_tensors(load_file(zero/'adapter_model.safetensors'),
        torch.load(zero/'head.pt', map_location='cpu', weights_only=True)['head'], parent['model']['directory'])
    logs = rows(fit/'metrics/train_metrics.jsonl')
    assert [r['local_step'] for r in logs] == list(range(1, last+1))
    actual_counts = Counter()
    expected = []
    for epoch in range(1, c['epochs']+1):
        order = read(fit/'epoch-order-private'/f'epoch-{epoch:02d}.json')
        assert order['each_record_exactly_once'] and order['copies_added'] == 0 and order['all_updates_have_all_declared_strata']
        expected.extend(order['source_counts_by_update'])
    assert expected == [r['source_counts'] for r in logs]
    seen = 0
    for r in logs:
        assert all(math.isfinite(r[k]) for k in ('loss', 'ce', 'grad_norm', 'lr')) and r['reward'] is None
        assert set(r['source_counts']) == set(reg['course_loss_mass'])
        reconstructed = sum(r['source_counts'][k]*r['source_ce'][k]*reg['course_loss_weights'][k]
            for k in r['source_counts'])/sum(r['source_counts'].values())
        assert abs(reconstructed-r['loss']) < 1e-5
        actual_counts.update(r['source_counts'])
        seen += sum(r['source_counts'].values())
        assert r['records_seen'] == seen
    data = read(c['data_ready'])
    assert actual_counts == Counter({k: v*c['epochs'] for k, v in data['categories']['train'].items()})
    receipt = read(reg['training_config']['data_receipt'])
    train, held = [rows(ROOT/receipt['files'][split]['path']) for split in ('train', 'validation')]
    seal = read(c['parent_seal'])
    parent_reg = read(seal['parent_registration'])
    parent_data = read(parent_reg['training_config']['data_receipt'])['files']
    require_replay(rows(ROOT/parent_data['train']['path']), train)
    require_replay(rows(ROOT/parent_data['validation']['path']), held)
    forbidden = {r['_meta']['group_id'] for r in held} | set(read(data['reserved_groups']))
    assert not {r['_meta']['group_id'] for r in train} & forbidden
    checkpoint = fit/'training-run/checkpoints'/f'step-{last:06d}'
    identity = bundle(checkpoint)
    state = torch.load(checkpoint/'trainer_state.pt', map_location='cpu', weights_only=False)
    assert state['optimizer_step'] == last and state['epoch'] == c['epochs'] and state['next_microbatch'] == 0
    assert state['records_seen'] == len(train)*c['epochs'] and len(state['optimizer']['state']) == 376
    assert all(float(v['step']) == last for v in state['optimizer']['state'].values())
    del state
    gc.collect()
    curve = read(fit/'validation-curve.json')
    steps = sorted({0, last, *range(c['validation_every_steps'], last+1, c['validation_every_steps'])})
    assert [p['step'] for p in curve] == steps
    for point in curve:
        predictions = read(fit/'training-run/training_eval'/f"step-{point['step']:06d}"/'development/rows.json')
        recomputed = profile(held, predictions, reg['training_config']['validation_phase_weights'], reg['course_loss_mass'])
        assert all(point[k] == v for k, v in recomputed.items())
    val_logs = rows(fit/'metrics/validation_metrics.jsonl')
    assert [r['local_step'] for r in val_logs] == steps
    for logged, point in zip(val_logs, curve):
        assert logged['validation_data_sha256'] == receipt['files']['validation']['sha256']
        assert logged['loss'] == point['weighted_target_ce'] and logged['weighted_target_KL'] == point['weighted_target_KL']
    saved = {r['id']: r for r in read(fit/'training-run/training_eval'/f'step-{last:06d}'/'development/rows.json')}
    client = UnifiedPolicyClient(load_model(checkpoint))
    error = 0.
    for row in held:
        actual = client.system_one(**api_request(row))['answers']['action']['probabilities']
        p = saved[row['_meta']['id']]
        assert set(actual) == set(p['keys'])
        error = max(error, max(abs(actual[k]-v) for k, v in zip(p['keys'], p['p'])))
    assert error < 5e-5
    del client
    gc.collect()
    torch.cuda.empty_cache()
    eval_steps = sorted({0, last, *range(c['game_evaluation_every_steps'], last+1, c['game_evaluation_every_steps'])})
    assert done['curve_steps'] == eval_steps
    games = {}
    checked = 0
    for label in ['reference-v8', 'reference-parent', *[f'step-{s:06d}' for s in eval_steps]]:
        folder = fit/'evaluations'/label
        protocol = read(folder/'protocol.json')
        matches = rows(folder/'matches-private.jsonl')
        count = c['stage_end_seed_groups'] if label.startswith('reference-') or label == f'step-{last:06d}' else c['curve_seed_groups']
        assert protocol['seeds'] == list(range(c['curve_seed_start'], c['curve_seed_start']+count))
        assert protocol['profiles'] == c['opponent_profiles']
        grid = {(seed, p, seat) for seed in protocol['seeds'] for p in c['opponent_profiles'] for seat in range(4)}
        assert len(matches) == len(grid) and {(r['seed'], r['profile'], r['seat']) for r in matches} == grid
        for r in matches:
            ledger = settlement_ledger(r['mjai_log'], r['final_scores'])
            seat = r['seat']
            assert r['completed'] and ledger['hands'] == r['hands'] and r['net_points'] == r['final_scores'][seat]-25000
            assert sorted(range(4), key=lambda p: (-r['final_scores'][p], p)).index(seat)+1 == r['rank']
            assert hand_dealin_count(r['mjai_log'], seat) == r['dealin_hands']
            assert sum(e['type'] == 'hora' and e.get('actor') == seat for e in r['mjai_log']) == r['wins']
        summary = read(folder/'summary.json')
        aggregate = aggregate_complete_matches(matches)
        assert all(summary[k] == v for k, v in aggregate.items())
        expected_identity = c['reference_identity'] if label == 'reference-v8' else c['parent_identity'] if label == 'reference-parent' else bundle(fit/'training-run/checkpoints'/label)['identity_sha256']
        assert summary['checkpoint_sha256'] == expected_identity
        if count == 8:
            subset = [r for r in matches if r['seed'] < c['curve_seed_start']+c['curve_seed_groups']]
            assert summary['curve_grid_subset'] == aggregate_complete_matches(subset)
        games[label] = matches
        checked += len(matches)
    zero_summary = read(fit/'evaluations/step-000000/summary.json')
    assert all(zero_summary[k] == v for k, v in read(fit/'evaluations/reference-parent/summary.json')['curve_grid_subset'].items())
    comparisons = {name: paired_comparison(games[f'step-{last:06d}'], games[name]) for name in ('reference-parent', 'reference-v8')}
    grade = quality(c, curve[0], curve[-1], comparisons['reference-parent'])
    for ext in ('png', 'pdf'):
        assert (fit/'metrics'/f'curriculum-curves.{ext}').stat().st_size > 1000
    plot = read(fit/'metrics/plot-data-provenance.json')
    assert plot['curve_steps'] == eval_steps and plot['final64_steps'] == [last]
    assert bundle(checkpoint) == identity
    verify()
    result = dict(status='stage3_final_evidence_audit_passed', optimizer_steps=last, real_records_seen=len(train)*c['epochs'],
        final_policy_identity_sha256=identity['identity_sha256'], step0_exact_parent_tensors=exact,
        step0_native_subset_exact_parent=True, actual_source_exposure=dict(actual_counts), all376_optimizer_states_match=True,
        saved_final_reloaded_menus=len(held), saved_final_max_probability_error=error,
        saved_reloads=[dict(step=last, identity=identity['identity_sha256'], menus=len(held), maximum_probability_error=error)],
        checked_complete_games=checked, final_summary=aggregate_complete_matches(games[f'step-{last:06d}']),
        comparison_to_parent=comparisons['reference-parent'], comparison_to_v8=comparisons['reference-v8'],
        original_stage1_advancement=seal['original_stage1_advancement'], original_stage2_advancement=seal['stage_quality'],
        stage3_advancement=grade, source_and_data_hashes_unchanged=True, heldout_groups_excluded=True,
        exact_train_and_validation_replay=True, MC_unknown_actions_masked=True,
        MC_labels_are_conditional_not_expert_truth=True, development_grid_not_independent_confirmation=True,
        independent_han_fu_scoring=False, automatic_stage4_started=False, automatic_release=False)
    write(out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binding', type=Path)
    a = p.parse_args()
    try:
        audit(a.registered.resolve(), a.out.resolve(), a.binding.resolve() if a.binding else None)
    except BaseException as error:
        write(a.out.with_name(a.out.stem+'-error.json'), dict(error_type=type(error).__name__, error=str(error)))
        raise
