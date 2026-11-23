"""Bounded outer replanning pilot; reuse exact previously saved value arms."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def projection(result):
    return {key: result[key] for key in ('terminal_scores', 'rank', 'top2', 'net_from_anchor',
        'future_hand_reseeds', 'merged_ledger_hands', 'next_hand_context_preserved', 'merged_mjai_log')}


def native_allocation(saved):
    result = dict(saved)
    for field in ('opponent_hands', 'opponent_temporary_furiten'):
        values = saved[field]
        if any(not (type(p) is int and p in range(4))
               and not (type(p) is str and p in ('0', '1', '2', '3')) for p in values):
            raise ValueError('Invalid serialized allocation seat key')
        result[field] = {int(p): value for p, value in values.items()}
        if len(result[field]) != len(values):
            raise ValueError('Aliased serialized allocation seat keys')
    return result


def audit(registered, out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Finite scheduled GPU resources are required')
    out.mkdir(parents=True, exist_ok=False)
    reg = read(registered)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    import riichienv._riichienv as native
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent, EfficiencyAgent, CallingOpponent
    from tools.audit_curriculum_endgame_cases import restore
    from tools.audit_curriculum_history_worlds import derived_seed
    from tools.curriculum_native_decisions import construct_world
    from tools.curriculum_observer_rollout_v1 import finish, WinReplanner
    from tools.curriculum_outer_fixture_v1 import late_ron_offer, FixtureAgent
    if (version('riichienv') != '0.4.11'
            or package_digest(Path(reg['source']) / 'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source']) / 'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Pinned runtime changed')
    previous = ROOT / 'runs/curriculum-public-win-planner-v1'
    parent = read(previous / 'pilot-v1/registered-private.json')
    completed = read(previous / 'pilot-v1/summary-private.json')
    terminal = (previous / 'pilot-terminal-scheduler-private.txt').read_text()
    if (completed['status'] != 'public_win_choice_and_hidden_world_invariance_pilot_complete'
            or completed['complete_terminal_arms'] != 64
            or read(ROOT / 'logs/curriculum-public-win-planner-exit-317376.json')['exit_code'] != 0
            or 'JobState=COMPLETED' not in terminal or 'ExitCode=0:0' not in terminal
            or not all(sha(p) == h for p, h in parent['files'].items())):
        raise ValueError('Actual completed public-win component or frozen sources differ')
    ready_path = ROOT / 'runs/curriculum-stage4-placement-mortal-data-v4/data/data-ready-private.json'
    ready = read(ready_path); paths = [registered, ready_path, Path(native.__file__),
        previous / 'pilot-v1/registered-private.json', previous / 'pilot-v1/summary-private.json',
        previous / 'pilot-v1/plans-private.jsonl', previous / 'pilot-terminal-scheduler-private.txt',
        ROOT / 'logs/curriculum-public-win-planner-exit-317376.json']
    rows = {}
    for split, spec in ready['files'].items():
        path = ROOT / spec['path']; paths.append(path)
        if sha(path) != spec['sha256']:
            raise ValueError('Canonical source data changed')
        if split == 'train':
            rows.update((row['_meta']['id'], row) for row in
                        (json.loads(line) for line in path.open()))
    plans = [json.loads(line) for line in (previous / 'pilot-v1/plans-private.jsonl').open()]
    selected = sorted(parent['selected'], key=lambda item: tuple(item['bucket']))
    if len(selected) != 4 or len(plans) != 4:
        raise ValueError('The exact earlier four contexts are required')
    caches = {tuple(item['bucket']): item for item in plans}
    for spec in selected:
        cache = caches[tuple(spec['bucket'])]; row = rows[spec['id']]
        if (cache['source_id'] != spec['id'] or row['_meta']['split'] != 'train'
                or cache['original'] != cache['changed_hidden']):
            raise ValueError('Changed public selection or failed prior hidden-world control')
    active_path = ROOT / 'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json'
    active = read(active_path)['files'] if active_path.exists() else {}
    if not all(sha(p) == h for p, h in active.items()):
        raise ValueError('An active fit dependency changed')
    paths += [ROOT / 'tools' / name for name in (
        'curriculum_observer_rollout_v1.py', 'curriculum_outer_fixture_v1.py',
        'audit_curriculum_observer_rollout_v1.py', 'curriculum_public_win_planner_v1.py',
        'curriculum_terminal_scout_stats.py', 'curriculum_observer_decline_fixtures_v3.py',
        'curriculum_native_decisions.py', 'curriculum_native_worlds.py',
        'curriculum_decision_history.py', 'curriculum_hierarchical_prior.py',
        'curriculum_public_inventory.py', 'curriculum_decision_rollout.py',
        'curriculum_endgame_ledger.py', 'audit_curriculum_endgame_cases.py',
        'audit_curriculum_history_worlds.py')]
    paths += [ROOT / 'tests/integration/test_curriculum_observer_rollout_v1.py',
        ROOT / 'docs/experiments/plans/curriculum-observer-rollout-20261123.md',
        ROOT / 'cluster/curriculum-observer-rollout-v1.slurm']
    bindings = {str(p.resolve()): sha(p) for p in paths}
    checkpoint = ROOT / reg['curriculum']['reference_checkpoint']; identity = bundle(checkpoint)
    if identity['identity_sha256'] != completed['actual_teacher_candidate_identity']:
        raise ValueError('Frozen v8 teacher candidate differs')
    write(out / 'registered-private.json', dict(
        method='bounded_depth1_observer_replanning_outer_v1', files=bindings,
        selected=selected, outer_allocation='previous public plan world0',
        baseline_cache='previous public plan paired world0 outcomes',
        initial_arms='both exact saved accept and decline keys, neither replanned',
        baseline_reproductions_before_changed_runs=8, changed_outer_matches=8,
        worlds_per_future_plan=4, max_plans_per_outer_match=4, recursion_depth=1,
        max_changed_inner_terminal_arms=256,
        real_model_fixture_worlds=2, real_model_fixture_max_plans=2,
        total_inner_terminal_arm_upper_bound=264, finite_steps=30000, finite_later_hands=64,
        teacher_candidate_identity=identity['identity_sha256'],
        no_new_context_or_outcome_selection=True, new_training_labels=0, teacher_admitted=False))
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests/integration'),
                                               pattern='test_curriculum_observer_rollout_v1.py')
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful() or tested.testsRun != 6:
        raise RuntimeError('Observer-loop native/contract controls failed')
    client = UnifiedPolicyClient(load_model(checkpoint))
    # Actual model-backed future planning after a forced initial legal PASS.
    fixture_env, _, _, fixture_decline = late_ron_offer()
    fixture_control = WinReplanner(client, seed=202611230600, worlds=2, max_calls=2)
    fixture_result = finish(fixture_env, 0, fixture_decline, 202611230601,
        [FixtureAgent() for _ in range(4)], observer=fixture_control, record_events=True)
    if fixture_control.counts['planned_calls'] < 1:
        raise ValueError('Real inner planning never ran at a later actual win opportunity')
    write(out / 'real-future-plan-fixture-private.json', dict(result=fixture_result,
        counts=dict(fixture_control.counts), plans=fixture_control.records,
        synthetic_fixture_not_population_or_training=True))
    def agents(seat):
        result = [CallingOpponent() if p == (seat + 1) % 4 else EfficiencyAgent() for p in range(4)]
        result[seat] = PolicyAgent(client); return result
    def world(spec):
        row = rows[spec['id']]; env, obs = restore(row)
        cache = caches[tuple(spec['bucket'])]['original']
        fork, _ = construct_world(env, obs, env.mjai_log[-1], native_allocation(cache['allocations'][0]), True)
        return fork, obs.player_id, cache
    # Verify all eight old complete event streams before changed-policy runs.
    with (out / 'baseline-reproductions-private.jsonl').open('x') as stream:
        for spec in selected:
            fork, seat, cache = world(spec)
            for arm in (cache['accept'], cache['decline']):
                baseline = finish(fork.clone(), seat, arm, derived_seed(cache['public_seed'], 0),
                                  agents(seat), record_events=True)
                if projection(baseline) != projection(cache['pairs'][0][arm]):
                    raise ValueError('Observer-disabled runner changed an actual cached baseline')
                stream.write(json.dumps(dict(bucket=spec['bucket'], arm=arm, result=baseline)) + '\n')
                stream.flush()
    cases = []; inner = fixture_control.counts['inner_terminal_arms']
    with (out / 'changed-outer-matches-private.jsonl').open('x') as stream:
        for spec in selected:
            fork, seat, cache = world(spec)
            for arm in (cache['accept'], cache['decline']):
                control = WinReplanner(client, seed=202611230600, worlds=4, max_calls=4)
                result = finish(fork.clone(), seat, arm, derived_seed(cache['public_seed'], 0),
                                agents(seat), observer=control, record_events=True)
                baseline = cache['pairs'][0][arm]
                inner += control.counts['inner_terminal_arms']
                if control.counts['planned_calls'] > 4:
                    raise ValueError('Outer planner exceeded its registered finite budget')
                item = dict(kind=spec['bucket'][0], score_context=spec['bucket'][1],
                    initial_arm='accept' if arm == cache['accept'] else 'decline',
                    observer_hook_calls=result['observer_hook_calls'],
                    planner_counts=dict(control.counts),
                    same_complete_events_as_ordinary=projection(result) == projection(baseline),
                    terminal_rank=result['rank'], baseline_terminal_rank=baseline['rank'],
                    conditional_net_difference=result['net_from_anchor'] - baseline['net_from_anchor'])
                cases.append(item)
                stream.write(json.dumps(dict(aggregate=item, result=result, plans=control.records)) + '\n')
                stream.flush(); print(json.dumps(dict(stage='outer_replanning_match_complete', **item)), flush=True)
    if len(cases) != 8 or inner > 264:
        raise ValueError('Incomplete outer grid or excess total inner work')
    if bundle(checkpoint) != identity or not all(sha(p) == h for p, h in bindings.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound teacher, new source or active fit inputs changed')
    summary = dict(status='bounded_observer_outer_replanning_pilot_complete',
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), integration_tests=6,
        model_backed_future_fixture_counts=dict(fixture_control.counts),
        old_complete_baseline_reproductions=8, changed_outer_matches=8,
        actual_total_inner_terminal_arms=inner, registered_inner_upper_bound=264,
        cases=cases, active_fit_dependencies_unchanged=len(active),
        inner_recursion_depth=1, inner_future_wins_rule_forced=True,
        outer_later_legal_wins_replanned_with_public_worlds=True,
        budget_fallback_is_ordinary_win=True,
        source_families=completed['source_families'], score_views_not_independent=True,
        conditional_forced_initial_arms_not_policy_strength=True,
        teacher_admitted=False, new_training_labels=0, optimizer_updates=0,
        independent_full_han_fu_scoring=False, prior_not_calibrated=True,
        stage4_fit_admitted=False, final_grid_unused=True, raw_records_private=True)
    write(out / 'summary-private.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit(args.registered.resolve(), args.out.resolve())
