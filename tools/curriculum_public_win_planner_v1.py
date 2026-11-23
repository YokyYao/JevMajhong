"""One-step public-world win planner; not an admitted pursuit teacher.

Inner matches use ordinary frozen-v8 continuation, including forced later
wins. Replanning at later offers needs a separate observer-aware outer loop.
"""
import hashlib
import json
import math
import os
import socket
import statistics

from tools.curriculum_terminal_scout_stats import utility

METHOD = 'public_seed_paired_terminal_win_choice_v1'


def public_seed(request, projected_history, seed):
    if type(seed) is not int:
        raise ValueError('An explicit integer planner seed is required')
    encoded = json.dumps(dict(request=request, history=projected_history, seed=seed,
                              method=METHOD), sort_keys=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], 'little')


def choose(pairs, accept, decline):
    if accept == decline or not pairs:
        raise ValueError('Distinct complete paired arms are required')
    for pair in pairs:
        if set(pair) != {accept, decline}:
            raise ValueError('Missing or changed intervention arm')
        for result in pair.values():
            rank = result.get('rank'); top2 = result.get('top2'); net = result.get('net_from_anchor')
            if (result.get('full_match_completed') is not True or type(rank) is not int
                    or rank not in range(1, 5) or type(top2) is not bool or top2 != (rank <= 2)
                    or type(net) not in (float, int) or not math.isfinite(net)):
                raise ValueError('Complete finite terminal rank/score evidence is required')
    values = {key: statistics.mean(utility(pair[key]) for pair in pairs)
              for key in (accept, decline)}
    # No last-place-rate veto or positive-CI gate. Exact utility ties take win.
    selected = decline if values[decline] > values[accept] else accept
    return dict(selected=selected, mean_utility=values,
                paired_decline_minus_accept_utility=values[decline] - values[accept],
                worlds=len(pairs), confidence_gate_used=False,
                uncertainty_is_not_teacher_admission=True)


def plan(env, obs, client, seed, worlds=4, max_attempts=250000):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Native planning requires a scheduled GPU allocation')
    from riichienv import ActionType, HandEvaluator
    from jevmahjong.core.state import build_request
    from jevmahjong.evaluation.agents import PolicyAgent, EfficiencyAgent, CallingOpponent, guarded_indices
    from jevany.api import validate_distribution
    from tools.audit_curriculum_history_worlds import derived_seed
    from tools.curriculum_endgame import in_endgame_scope
    from tools.curriculum_win_context import build_win_request
    from tools.curriculum_decision_history import public_history
    from tools.curriculum_native_decisions import decision_scope, decision_inventory, construct_world
    from tools.curriculum_hierarchical_prior import sample, METHOD as PRIOR
    from tools.curriculum_decision_rollout import finish
    if type(worlds) is not int or not 1 <= worlds <= 16:
        raise ValueError('An explicit bounded inner-world budget is required')
    scope = decision_scope(env, obs, env.mjai_log[-1])
    request, actions, features = build_request(obs)
    if not in_endgame_scope(request['state']):
        raise ValueError('Only scheduled East4/extensions win offers are admitted')
    wins = [i for i, a in enumerate(actions) if a.action_type in (ActionType.RON, ActionType.TSUMO)]
    if len(wins) != 1:
        raise ValueError('Exactly one actual legal win candidate is required')
    keys = list(request['questions']['action']['criteria']); win = wins[0]
    if actions[win].action_type == ActionType.RON:
        alternatives = [i for i, a in enumerate(actions) if a.action_type == ActionType.PASS]
        if len(alternatives) != 1:
            raise ValueError('A ron decline needs its actual unique native PASS')
        other = alternatives[0]
    else:
        allowed = set(guarded_indices(obs, actions, features))
        discards = [i for i, a in enumerate(actions)
                    if i in allowed and a.action_type == ActionType.DISCARD]
        if not discards:
            raise ValueError('No ordinary guarded legal discard for declining tsumo')
        if len(discards) == 1:
            other = discards[0]
        else:
            # Frozen v8 still receives its ordinary input, not a new win schema.
            raw = dict(client.system_one(**request)['answers']['action']['probabilities'])
            validate_distribution(dict(raw), keys)
            other = max(discards, key=lambda i: raw[keys[i]])
    accept, decline = keys[win], keys[other]
    history = public_history(env.mjai_log)
    # Public win source/payer affects the planning seed, never model features.
    win_request = build_win_request(obs, env.mjai_log[-1])[0]
    planned_seed = public_seed(win_request, history, seed)
    visible = obs.to_dict(); seat = obs.player_id
    waits = lambda p, hand: list(HandEvaluator(hand, list(visible['melds'][p])).get_waits())
    proposal = sample(decision_inventory(obs, scope), history, visible['riichi_declared'],
                      planned_seed, worlds, max_attempts, waits)
    pairs = []; receipts = []
    for index, allocation in enumerate(proposal['allocations']):
        fork, receipt = construct_world(env, obs, env.mjai_log[-1], allocation, True)
        outcomes = {}
        for arm in (accept, decline):
            agents = [CallingOpponent() if p == (seat + 1) % 4 else EfficiencyAgent()
                      for p in range(4)]
            agents[seat] = PolicyAgent(client)
            outcomes[arm] = finish(fork.clone(), seat, arm, derived_seed(planned_seed, index),
                                   agents=agents, record_events=True)
        pairs.append(outcomes); receipts.append(receipt)
    decision = choose(pairs, accept, decline)
    return dict(method=METHOD, accept=accept, decline=decline, decision=decision,
                public_seed=planned_seed, scope=scope, prior=PRIOR,
                allocations=proposal['allocations'], constructor_receipts=receipts,
                pairs=pairs, full_terminal_arms=2 * worlds,
                utility='top2 + .02*(4-rank)/3 + .005*tanh(net_from_anchor/8000)',
                inner_worlds_independent_of_actual_hidden_hands=True,
                prior_not_calibrated=True, subsequent_inner_wins_rule_forced=True,
                ordinary_ron_PASS_or_one_guarded_tsumo_discard_only=True,
                not_all_current_legal_action_values=True, nested_outer_replanning_not_implemented=True,
                teacher_admitted=False, training_labels=False)
