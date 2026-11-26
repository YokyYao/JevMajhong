"""Public-world terminal win teacher with rank-priority and bounded replanning.

This is a candidate search teacher, not Mortal policy inference or an expert
optimality claim. Its frozen continuation is v8 plus public heuristic opponents.
"""
import math
import random

from tools.curriculum_terminal_rank_priority_v1 import choose, METHOD as OBJECTIVE
from tools.curriculum_public_win_planner_v1 import public_seed

METHOD = 'public32_rank_priority_depth1_win_replanning_soft_bootstrap_v1'


def inner_plan(env, obs, client, seed, worlds=4):
    from tools.curriculum_public_win_planner_v1 import plan
    from tools.curriculum_endgame import public_placement
    from jevmahjong.core.state import build_request
    result = plan(env, obs, client, seed=seed, worlds=worlds)
    rank = public_placement(build_request(obs)[0]['state'])['rank']
    decision = choose(result['pairs'], result['accept'], result['decline'], rank)
    return dict(result, method=METHOD, decision=decision, objective=OBJECTIVE,
        utility='lexicographic top2, lower rank, behind-only bounded net',
        inner_depth=1, inner_future_wins_forced=True,
        nested_outer_replanning_not_implemented=False, teacher_admitted=False, training_labels=False)


def propose(env, obs, client, seed, worlds=16):
    """Build public allocations without spending ordinary outer rollout arms."""
    from riichienv import ActionType, HandEvaluator
    from jevmahjong.core.state import build_request
    from jevmahjong.evaluation.agents import guarded_indices
    from jevany.api import validate_distribution
    from tools.curriculum_endgame import in_endgame_scope, public_placement
    from tools.curriculum_win_context import build_win_request
    from tools.curriculum_decision_history import public_history
    from tools.curriculum_native_decisions import decision_scope, decision_inventory
    from tools.curriculum_hierarchical_prior import sample
    if type(worlds) is not int or worlds != 16:
        raise ValueError('Exactly sixteen public worlds per independent block required')
    trigger = env.mjai_log[-1]; scope = decision_scope(env, obs, trigger)
    request, actions, features = build_request(obs)
    if not in_endgame_scope(request['state']):
        raise ValueError('Only scheduled East4/extensions, not assumed final hand')
    wins = [i for i, a in enumerate(actions) if a.action_type in (ActionType.RON, ActionType.TSUMO)]
    if len(wins) != 1: raise ValueError('Exactly one actual legal win required')
    win = wins[0]; keys = list(request['questions']['action']['criteria'])
    if actions[win].action_type == ActionType.RON:
        alternatives = [i for i, a in enumerate(actions) if a.action_type == ActionType.PASS]
        if len(alternatives) != 1: raise ValueError('Unique native ron PASS required')
        other = alternatives[0]
    else:
        allowed = set(guarded_indices(obs, actions, features))
        discards = [i for i, a in enumerate(actions) if i in allowed and a.action_type == ActionType.DISCARD]
        if not discards: raise ValueError('A tsumo decline needs an ordinary legal discard')
        if len(discards) == 1: other = discards[0]
        else:
            raw = dict(client.system_one(**request)['answers']['action']['probabilities'])
            validate_distribution(dict(raw), keys)
            other = max(discards, key=lambda i: raw[keys[i]])
    history = public_history(env.mjai_log)
    win_request = build_win_request(obs, trigger)[0]
    planned_seed = public_seed(win_request, history, seed)
    visible = obs.to_dict()
    waits = lambda p, hand: list(HandEvaluator(hand, list(visible['melds'][p])).get_waits())
    proposal = sample(decision_inventory(obs, scope), history, visible['riichi_declared'],
        planned_seed, worlds, 250000, waits)
    return dict(accept=keys[win], decline=keys[other], public_seed=planned_seed,
        initial_public_rank=public_placement(win_request['state'])['rank'],
        proposal=proposal, scope=scope)


def soft_target(pairs, accept, decline, initial_rank, seed, samples=512):
    """Paired-world bootstrap preference, retaining uncertain and fourth-risk rows."""
    if len(pairs) != 32 or samples != 512 or type(seed) is not int:
        raise ValueError('Fixed32-world paired bootstrap with512 samples required')
    point = choose(pairs, accept, decline, initial_rank)
    rng = random.Random(seed); votes = 0
    for _ in range(samples):
        sample = [pairs[rng.randrange(32)] for _ in range(32)]
        votes += choose(sample, accept, decline, initial_rank)['selected'] == decline
    p = (votes+.5)/(samples+1.)
    blocks = [choose(pairs[:16], accept, decline, initial_rank), choose(pairs[16:], accept, decline, initial_rank)]
    leave_one = [choose(pairs[:i]+pairs[i+1:], accept, decline, initial_rank)['selected'] for i in range(32)]
    if not 0 < p < 1 or not math.isfinite(p): raise ValueError('Finite positive two-arm soft mass required')
    return dict(method=METHOD, objective=OBJECTIVE, target={accept: 1-p, decline: p},
        label=decline if p > .5 else accept, pooled_decision=point,
        bootstrap_decline_votes=votes, bootstrap_samples=samples,
        independent_block_choices_agree=blocks[0]['selected'] == blocks[1]['selected'],
        every_leave_one_world_choice_agrees=all(v == point['selected'] for v in leave_one),
        fourth_place_escape_veto=False, positive_confidence_gate=False,
        uncertain_rows_filtered=False, paired_worlds_not_independent_source_families=True,
        bootstrap_preference_is_not_calibrated_optimal_action_probability=True)
