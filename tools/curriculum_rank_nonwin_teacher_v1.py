"""Bounded terminal action-value proposals and multi-action soft preferences.

Candidate proposals use only public menus/features and frozen model scores.
They are a training-budget device, not a deployment action filter or label.
"""
from collections import Counter
import hashlib
import json
import math
import random
import statistics

from tools.curriculum_terminal_rank_priority_v1 import choose as validate_pair, METHOD as OBJECTIVE

METHOD = 'public32_rank_priority_nonwin_four_candidate_bootstrap_v1'


def candidate_keys(request, probabilities, baseline, cap=4):
    criteria = request['questions']['action']['criteria']; keys = list(criteria)
    if (cap != 4 or len(keys) < 2 or baseline not in keys or set(probabilities) != set(keys)
            or any(c['type'] == 'hora' for c in criteria.values())
            or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in probabilities.values())):
        raise ValueError('Actual non-win legal menu, finite scores and registered cap4 required')
    chosen = [baseline]
    def add(key):
        if key not in chosen and len(chosen) < cap: chosen.append(key)
    kinds = {c['type'] for c in criteria.values()}
    for kind in ('none', 'reach', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'ryukyoku', 'dahai'):
        if kind in kinds and not any(criteria[k]['type'] == kind for k in chosen):
            options = [k for k in keys if criteria[k]['type'] == kind]
            add(max(options, key=lambda k: probabilities[k]))
    discards = [k for k in keys if criteria[k]['type'] == 'dahai' and isinstance(criteria[k].get('sh'), (int, float))]
    if discards:
        add(min(discards, key=lambda k: (criteria[k]['sh'], -probabilities[k])))
        threats = set(request['state'].get('riichi', []))-{request['state']['seat']}
        known_safe = [k for k in discards if isinstance(criteria[k].get('safe'), list)]
        if threats and known_safe:
            add(min(known_safe, key=lambda k: (len(threats-set(criteria[k]['safe'])), criteria[k]['sh'], -probabilities[k])))
    for key in sorted(keys, key=lambda k: -probabilities[k]): add(key)
    if not 2 <= len(chosen) <= 4: raise ValueError('Two to four distinct candidates required')
    return chosen


def propose(env, obs, client, seed):
    from riichienv import HandEvaluator
    from jevmahjong.core.state import build_request, signature
    from jevmahjong.evaluation.agents import PolicyAgent
    from jevany.api import validate_distribution
    from tools.curriculum_endgame import in_endgame_scope, public_placement
    from tools.curriculum_decision_history import public_history
    from tools.curriculum_native_decisions import decision_scope, decision_inventory
    from tools.curriculum_hierarchical_prior import sample
    request, actions, _ = build_request(obs)
    if not in_endgame_scope(request['state']) or any(c['type'] == 'hora' for c in request['questions']['action']['criteria'].values()):
        raise ValueError('Supported late non-win native scope required')
    class Capture:
        def __init__(self): self.calls = 0; self.probabilities = None
        def system_one(self, **value):
            if value != request: raise ValueError('Reference public request changed')
            self.calls += 1; result = client.system_one(**value)
            self.probabilities = result['answers']['action']['probabilities']
            return result
    capture = Capture(); selected = PolicyAgent(capture).act(obs)
    if capture.calls != 1: raise ValueError('One frozen current-reference inference required')
    keys = list(request['questions']['action']['criteria'])
    validate_distribution(dict(capture.probabilities), keys)
    match = [keys[i] for i, action in enumerate(actions) if signature(action) == signature(selected)]
    if len(match) != 1: raise ValueError('Reference has no unique actual native action')
    baseline = match[0]; candidates = candidate_keys(request, capture.probabilities, baseline)
    history = public_history(env.mjai_log); scope = decision_scope(env, obs, env.mjai_log[-1])
    encoded = json.dumps(dict(method=METHOD, request=request, history=history, seed=seed),
        sort_keys=True, separators=(',', ':'))
    planned_seed = int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], 'little')
    visible = obs.to_dict()
    waits = lambda p, hand: list(HandEvaluator(hand, list(visible['melds'][p])).get_waits())
    proposal = sample(decision_inventory(obs, scope), history, visible['riichi_declared'],
        planned_seed, 16, 250000, waits)
    return dict(baseline=baseline, candidates=candidates, public_seed=planned_seed,
        initial_public_rank=public_placement(request['state'])['rank'], proposal=proposal, scope=scope,
        action_types={k: request['questions']['action']['criteria'][k]['type'] for k in candidates})


def soft_target(worlds, candidates, initial_rank, seed):
    if len(worlds) != 32 or not 2 <= len(candidates) <= 4 or len(set(candidates)) != len(candidates):
        raise ValueError('Complete32 matched worlds and two to four ordered candidates required')
    baseline = candidates[0]
    for world in worlds:
        if set(world) != set(candidates): raise ValueError('Candidate menu differs across worlds')
    for alternative in candidates[1:]:
        validate_pair([{baseline:w[baseline], alternative:w[alternative]} for w in worlds],
            baseline, alternative, initial_rank)
    def select(pool):
        totals = {k: (sum(int(w[k]['top2']) for w in pool),
            -sum(w[k]['rank'] for w in pool),
            sum(math.tanh(w[k]['net_from_anchor']/8000) for w in pool) if initial_rank >= 3 else 0.)
            for k in candidates}
        return max(candidates, key=lambda k: totals[k])
    point = select(worlds); rng = random.Random(seed); counts = Counter()
    for _ in range(512): counts[select([worlds[rng.randrange(32)] for _ in range(32)])] += 1
    target = {k: (counts[k]+.5)/(512+.5*len(candidates)) for k in candidates}
    return dict(method=METHOD, objective=OBJECTIVE, target=target, label=max(candidates, key=target.get),
        baseline=baseline, point_choice=point, bootstrap_samples=512,
        independent_block_choices_agree=select(worlds[:16]) == select(worlds[16:]),
        every_leave_one_world_choice_agrees=all(select(worlds[:i]+worlds[i+1:]) == point for i in range(32)),
        means={k:dict(top2=statistics.mean(w[k]['top2'] for w in worlds),
            rank=statistics.mean(w[k]['rank'] for w in worlds),
            net=statistics.mean(w[k]['net_from_anchor'] for w in worlds)) for k in candidates},
        exact_placement_ties_keep_current_reference=True, no_fourth_escape_veto=True,
        no_positive_confidence_gate=True, all_uncertain_targets_retained=True,
        preferences_not_calibrated_expert_probabilities=True)
