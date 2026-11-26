"""Known observer declines through resampled native worlds and full matches.

No teacher, training targets, model parameters or final-test seeds are used.
"""
import argparse
from collections import Counter
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


def audit(registered, out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('An actual finite scheduled GPU allocation is required')
    out.mkdir(parents=True, exist_ok=False)
    reg = read(registered)
    sys.path.insert(0, reg['source'])
    import riichienv._riichienv as native
    from riichienv import HandEvaluator, ActionType
    from jevmahjong.core.state import build_request
    from jevmahjong.training.base_lora import package_digest
    from tools.audit_curriculum_history_worlds import derived_seed
    from tools.curriculum_decision_history import public_history
    from tools.curriculum_hierarchical_prior import sample, METHOD
    from tools.curriculum_native_decisions import decision_scope, decision_inventory, construct_world
    from tools.curriculum_decision_rollout import finish
    from tools.curriculum_observer_decline_fixtures_v3 import KINDS, SEED, fixture, verify_recovery, verify_sampled_recovery, planned_discard
    if (version('riichienv') != '0.4.11'
            or package_digest(Path(reg['source']) / 'jevmahjong') != reg['source_sha256']):
        raise ValueError('Pinned native/project runtime differs')
    active_path = ROOT / 'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json'
    active = read(active_path)['files'] if active_path.exists() else {}
    if not all(sha(p) == h for p, h in active.items()):
        raise ValueError('An existing active fit dependency changed')
    names = ('curriculum_observer_decline_fixtures_v3.py', 'audit_curriculum_observer_declines_v3.py',
             'curriculum_native_decisions.py', 'curriculum_native_worlds.py',
             'curriculum_decision_history.py', 'curriculum_hierarchical_prior.py',
             'curriculum_public_inventory.py', 'curriculum_decision_rollout.py',
             'curriculum_endgame_ledger.py', 'audit_curriculum_history_worlds.py')
    paths = [ROOT / 'tools' / name for name in names]
    paths += [registered, Path(native.__file__),
              ROOT / 'tests/core/test_curriculum_observer_declines_v3.py',
              ROOT / 'docs/experiments/plans/curriculum-observer-declines-v3-20261123.md',
              ROOT / 'cluster/curriculum-observer-declines-v3.slurm']
    bindings = {str(p.resolve()): sha(p) for p in paths}
    write(out / 'registered-private.json', dict(
        method='actual_observer_ron_declines_native_worlds_v3', files=bindings,
        fixture_kinds=list(KINDS), fixture_seed=SEED, sampled_worlds_per_fixture=2,
        max_prior_attempts_per_fixture=10000, prior=METHOD,
        physical_fixtures_not_population_data=True, known_own_temporary_flag_exact_and_latent_RI_behavior=True,
        opponents_never_decline_legal_wins=True, current_robbed_kan_excluded=True,
        no_teacher_no_labels_no_updates=True, no_reserved_final_seeds=True,
        active_fit_dependencies=len(active)))
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.core.test_curriculum_observer_declines_v3')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 3:
        raise RuntimeError('Actual known-decline regression controls failed')
    counts = Counter(); cases = []
    with (out / 'resampled-full-matches-private.jsonl').open('x') as stream:
        for number, kind in enumerate(KINDS):
            env, obs, proof = fixture(kind)
            native_flags = {'missed_agari_doujun': bool(env.missed_agari_doujun[0])}
            request, actions, _ = build_request(obs)
            keys = list(request['questions']['action']['criteria'])
            wanted = planned_discard(obs, kind)
            chosen = next(key for key, action in zip(keys, actions)
                          if action.action_type == ActionType.DISCARD and action.tile // 4 == wanted // 4)
            arms = [chosen]
            if kind.endswith('tsumo'):
                arms.insert(0, next(key for key, action in zip(keys, actions)
                                    if action.action_type == ActionType.TSUMO))
            else:
                original = env.clone()
                proof['original_recovery_control'] = verify_recovery(
                    original, original.get_obs_py([0])[0], kind)
                counts['original_world_recovery_controls'] += 1
            scope = decision_scope(env, obs, env.mjai_log[-1])
            visible = obs.to_dict()
            waiting = lambda p, hand: list(HandEvaluator(hand, list(visible['melds'][p])).get_waits())
            proposal = sample(decision_inventory(obs, scope), public_history(env.mjai_log),
                              visible['riichi_declared'], SEED + 100 + number, 2, 10000, waiting)
            changed_hidden = 0
            for world, allocation in enumerate(proposal['allocations']):
                fork, receipt = construct_world(env, obs, env.mjai_log[-1], allocation, True)
                fresh = fork.get_obs_py([0])[0]
                if build_request(fresh)[0] != request:
                    raise ValueError('Resampling changed the observer request/candidate menu')
                if any(bool(getattr(fork, field)[0]) != value for field, value in native_flags.items()):
                    raise ValueError('Resampling changed known observer missed-win state')
                changed = any(list(fork.hands[p]) != list(env.hands[p]) for p in (1, 2, 3))
                if not changed:
                    raise ValueError('A prior draw unexpectedly retained all actual hidden hands')
                changed_hidden += 1
                behavior = None
                if not kind.endswith('tsumo'):
                    behavior = verify_sampled_recovery(fork, fresh, kind)
                    counts['resampled_legal_ron_behavior_controls'] += 1
                outcomes = {arm: finish(fork.clone(), 0, arm, derived_seed(SEED + number, world),
                                        record_events=True) for arm in arms}
                for value in outcomes.values():
                    if value['full_match_completed'] is not True:
                        raise ValueError('A decline fixture continuation did not reach terminal')
                    counts['complete_terminal_matches'] += 1
                    counts['later_hand_reseeds'] += value['future_hand_reseeds']
                stream.write(json.dumps(dict(kind=kind, world=world, native_flags=native_flags,
                                             receipt=receipt, behavioral_control=behavior, outcomes=outcomes)) + '\n')
                stream.flush()
                counts['resampled_known_flag_and_input_controls'] += 1
            proof.update(sampled_worlds=2, changed_hidden_worlds=changed_hidden,
                         exact_public_input_and_exposed_temporary_flag=True,
                         full_terminal_matches=2 * len(arms))
            cases.append(proof)
            print(json.dumps(dict(stage='known_observer_decline_case_complete', **proof)), flush=True)
    if (counts['resampled_known_flag_and_input_controls'] != 10
            or counts['complete_terminal_matches'] != 14
            or counts['original_world_recovery_controls'] != 3
            or counts['resampled_legal_ron_behavior_controls'] != 6):
        raise ValueError('The registered complete fixture budget was not executed')
    if not all(sha(p) == h for p, h in bindings.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound audit or active training inputs changed')
    summary = dict(status='known_observer_declines_resampled_native_and_full_match_verified',
                   node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'],
                   counts=dict(counts), cases=cases, regression_tests=3,
                   active_fit_dependencies_unchanged=len(active),
                   temporary_reset_is_own_discard_in_pinned_engine=True,
                   latent_riichi_furiten_behavior_verified=True, furiten_does_not_forbid_tsumo=True,
                   arbitrary_manually_injected_flags_not_used_as_sources=True,
                   temporary_only_clear_and_unknown_swap_controls_are_diagnostic_clones_only=True,
                   synthetic_physical_fixtures_not_training_examples=True,
                   future_ordinary_wins_still_rule_forced=True,
                   nested_pursuit_teacher_still_not_implemented=True,
                   independent_full_han_fu_scoring=False, prior_not_calibrated=True,
                   new_training_labels=0, optimizer_updates=0, stage4_fit_admitted=False,
                   unseen_final_grid_unused=True, raw_worlds_events_private=True)
    write(out / 'summary-private.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.registered.resolve(), args.out.resolve())
