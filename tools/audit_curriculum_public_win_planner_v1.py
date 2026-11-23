"""Frozen-v8 public win-choice pilot and hidden-world invariance controls."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


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
    from riichienv import HandEvaluator
    from jevmahjong.core.state import build_request
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from tools.curriculum_endgame import in_endgame_scope
    from tools.audit_curriculum_endgame_cases import restore
    from tools.curriculum_decision_history import public_history
    from tools.curriculum_native_decisions import decision_scope, decision_inventory, construct_world
    from tools.curriculum_hierarchical_prior import sample
    from tools.curriculum_public_win_planner_v1 import plan, METHOD
    if (version('riichienv') != '0.4.11'
            or package_digest(Path(reg['source']) / 'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source']) / 'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Pinned native/project/encoder runtime differs')
    prior = ROOT / 'runs/curriculum-observer-declines-v3'
    proof = read(prior / 'audit-v1/summary-private.json')
    if (proof['status'] != 'known_observer_declines_resampled_native_and_full_match_verified'
            or proof['counts']['complete_terminal_matches'] != 14
            or read(ROOT / 'logs/curriculum-observer-declines-v3-exit-317302.json')['exit_code'] != 0
            or 'JobState=COMPLETED' not in (prior / 'audit-terminal-scheduler-private.txt').read_text()
            or 'ExitCode=0:0' not in (prior / 'audit-terminal-scheduler-private.txt').read_text()):
        raise ValueError('Actual successful known-decline integration is required')
    active_path = ROOT / 'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json'
    active = read(active_path)['files'] if active_path.exists() else {}
    if not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Active fit dependency changed')
    ready_path = ROOT / 'runs/curriculum-stage4-placement-mortal-data-v4/data/data-ready-private.json'
    ready = read(ready_path); paths = [registered, ready_path, Path(native.__file__),
        prior / 'audit-v1/summary-private.json', prior / 'audit-v1/registered-private.json',
        prior / 'audit-terminal-scheduler-private.txt', ROOT / 'logs/curriculum-observer-declines-v3-exit-317302.json']
    wanted = {(kind, context) for kind in ('ron', 'tsumo')
              for context in ('third_gap3000', 'fourth_gap5000')}
    selected = {}
    for split, spec in ready['files'].items():
        path = ROOT / spec['path']; paths.append(path)
        if sha(path) != spec['sha256']:
            raise ValueError('Registered canonical data changed')
        if split != 'train':
            continue
        rows = sorted((json.loads(line) for line in path.open()), key=lambda row: row['_meta']['id'])
        for row in rows:
            meta, state = row['_meta'], row['state']
            if (meta['category'] != 'win' or meta['split'] != 'train'
                    or not in_endgame_scope(state)
                    or sum(p != state['seat'] for p in state['riichi']) > 1):
                continue
            contexts = state.get('win_context', {}).get('candidates', {})
            kinds = {value['kind'] for value in contexts.values()}
            if len(kinds) != 1:
                raise ValueError('Canonical win kind is ambiguous')
            bucket = (next(iter(kinds)), meta['score_context'])
            if bucket in wanted and bucket not in selected:
                selected[bucket] = row
    if set(selected) != wanted:
        raise ValueError('The fixed TRAIN/public-kind/score-context pilot lacks a required bucket')
    paths += [ROOT / 'tools' / name for name in (
        'curriculum_public_win_planner_v1.py', 'audit_curriculum_public_win_planner_v1.py',
        'curriculum_terminal_scout_stats.py', 'curriculum_decision_history.py',
        'curriculum_native_decisions.py', 'curriculum_native_worlds.py', 'curriculum_hierarchical_prior.py',
        'curriculum_public_inventory.py', 'curriculum_decision_rollout.py',
        'curriculum_win_context.py', 'curriculum_endgame.py', 'curriculum_endgame_ledger.py',
        'audit_curriculum_endgame_cases.py', 'audit_curriculum_history_worlds.py')]
    paths += [ROOT / 'tests/integration/test_curriculum_public_win_planner_v1.py',
        ROOT / 'docs/experiments/plans/curriculum-public-win-planner-20261123.md',
        ROOT / 'cluster/curriculum-public-win-planner-v1.slurm']
    bindings = {str(p.resolve()): sha(p) for p in paths}
    checkpoint = ROOT / reg['curriculum']['reference_checkpoint']; identity = bundle(checkpoint)
    if identity['identity_sha256'] != reg['curriculum']['reference_identity']:
        raise ValueError('Frozen v8 teacher candidate changed')
    write(out / 'registered-private.json', dict(method=METHOD, files=bindings,
        selected=[dict(bucket=list(k), id=r['_meta']['id'], group=r['_meta']['group_id'])
                  for k, r in sorted(selected.items())],
        public_outcome_blind_first_ID_selection=True, train_only=True,
        inner_worlds_per_case_per_outer_variant=4, outer_variants=2,
        registered_total_inner_terminal_arms=64, max_prior_attempts_per_plan=250000,
        teacher_candidate_identity=identity['identity_sha256'],
        utility='top2 + .02*(4-rank)/3 + .005*tanh(net_from_anchor/8000)',
        ordinary_future_wins_forced=True, nested_outer_policy_not_implemented=True,
        new_labels=False, fit_admitted=False))
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests/integration'),
                       pattern='test_curriculum_public_win_planner_v1.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 5:
        raise RuntimeError('Paired utility/public-seed integration checks failed')
    started = time.monotonic(); client = UnifiedPolicyClient(load_model(checkpoint))
    load_seconds = time.monotonic() - started; cases = []; total_arms = 0
    with (out / 'plans-private.jsonl').open('x') as stream:
        for number, (bucket, row) in enumerate(sorted(selected.items())):
            env, obs = restore(row); visible = obs.to_dict(); seat = obs.player_id
            waits = lambda p, hand: list(HandEvaluator(hand, list(visible['melds'][p])).get_waits())
            proposal = sample(decision_inventory(obs, decision_scope(env, obs, env.mjai_log[-1])),
                              public_history(env.mjai_log), visible['riichi_declared'],
                              202611230200 + number, 1, 250000, waits)
            variant, _ = construct_world(env, obs, env.mjai_log[-1], proposal['allocations'][0], True)
            changed_obs = variant.get_obs_py([seat])[seat]
            if (build_request(changed_obs)[0] != build_request(obs)[0]
                    or not any(list(variant.hands[p]) != list(env.hands[p]) for p in range(4) if p != seat)):
                raise ValueError('Outer hidden-world control did not change hidden hands while preserving public input')
            original = plan(env, obs, client, seed=202611230300, worlds=4)
            changed = plan(variant, changed_obs, client, seed=202611230300, worlds=4)
            # Full sampled allocations, decisions and terminal event streams,
            # not just equal means or the same chosen action, must reproduce.
            if original != changed:
                raise ValueError('Planning depends on the actual outer hidden world')
            total_arms += original['full_terminal_arms'] + changed['full_terminal_arms']
            stream.write(json.dumps(dict(bucket=list(bucket), source_id=row['_meta']['id'],
                                         original=original, changed_hidden=changed)) + '\n'); stream.flush()
            item = dict(kind=bucket[0], score_context=bucket[1],
                        exact_inner_allocations_decision_and_all_events=True,
                        inner_worlds=4, outer_variants=2, full_terminal_arms=16,
                        selected_decline=original['decision']['selected'] == original['decline'],
                        paired_decline_minus_accept_utility=original['decision']['paired_decline_minus_accept_utility'])
            cases.append(item); print(json.dumps(dict(stage='public_win_plan_case_complete', **item)), flush=True)
    if total_arms != 64 or bundle(checkpoint) != identity:
        raise ValueError('Incomplete paired budget or changed frozen teacher')
    if not all(sha(p) == h for p, h in bindings.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound pilot or active training inputs changed')
    summary = dict(status='public_win_choice_and_hidden_world_invariance_pilot_complete',
                   job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
                   cases=cases, complete_terminal_arms=total_arms, integration_tests=5,
                   model_load_seconds=load_seconds, active_fit_dependencies_unchanged=len(active),
                   source_families=len({row['_meta']['group_id'] for row in selected.values()}),
                   TRAIN_score_views_not_independent_population_games=True,
                   actual_teacher_candidate_identity=identity['identity_sha256'],
                   hidden_world_invariance_scope='four selected current TRAIN win contexts only',
                   ordinary_future_inner_wins_still_rule_forced=True,
                   nested_outer_replanning_not_implemented=True,
                   teacher_admitted=False, new_training_labels=0, optimizer_updates=0,
                   prior_not_calibrated=True, independent_full_han_fu_scoring=False,
                   stage4_fit_admitted=False, untouched_final_grid_unused=True,
                   raw_plans_allocations_and_events_private=True)
    write(out / 'summary-private.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); audit(args.registered.resolve(), args.out.resolve())
