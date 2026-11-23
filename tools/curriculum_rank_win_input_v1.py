"""Explicit public-win input and one-model terminal win/continue execution.

Ordinary normalization and complete-legal RI execution are preserved. This
research contract is enabled explicitly; no production policy is changed.
"""
import copy
import json

from tools.curriculum_win_context import SCHEMA as PUBLIC_SCHEMA, public_tile, build_win_request
from tools.curriculum_endgame import in_endgame_scope
from tools.curriculum_complete_legal_class_policy_v1 import CompleteLegalClassPolicyAgent

SCHEMA = 'rank-win-public-source-and-terminal-instruction-v1'
METHOD = 'same_model_late_win_vs_native_PASS_or_best_guarded_discard_v1'
INSTRUCTION = (
    'Choose a legal action in four-player Mahjong Soul East-only riichi mahjong. '
    'In scheduled East4 and extensions, compare taking the legal win with continuing '
    'toward terminal top two, then better terminal rank. When behind, consider the '
    'public score gap and who pays a ron. A small win can still be useful through '
    'dealer repeats or later hands. Prefer taking the win on an exact placement tie '
    'while leading. The current hand is not assumed to end the match. '
    'Consider legal yaku, honba and riichi sticks. sh=shanten; ukeire=visible '
    'improving tiles; safe=opponent seats for which a tile is genbutsu. '
    'Probabilities are relative action preferences.'
)


def checked_context(request):
    state = request['state']; context = state['win_context']
    criteria = request['questions']['action']['criteria']
    wins = {k for k, v in criteria.items() if v['type'] == 'hora'}
    if (not in_endgame_scope(state) or not isinstance(context, dict)
            or set(context) != {'schema', 'candidates'} or context['schema'] != PUBLIC_SCHEMA
            or not isinstance(context['candidates'], dict)
            or set(context['candidates']) != wins or len(wins) != 1):
        raise ValueError('Exact public late one-win source schema required')
    seat = state['seat']
    if type(seat) is not int or seat not in range(4): raise ValueError('Valid observing seat required')
    for key, value in context['candidates'].items():
        if not isinstance(value, dict) or set(value) != {'kind', 'source', 'source_seat', 'tile'}:
            raise ValueError('Only public kind/source/seat/tile win context permitted')
        kind, source, payer, tile = (value[k] for k in ('kind', 'source', 'source_seat', 'tile'))
        if type(payer) is not int or payer not in range(4) or public_tile(tile) != tile:
            raise ValueError('Canonical public win tile and source seat required')
        candidate_tile = criteria[key].get('pai')
        if candidate_tile is not None and public_tile(candidate_tile) != tile:
            raise ValueError('Win context differs from the legal candidate tile')
        if kind == 'tsumo':
            if source != 'self_draw' or payer != seat or tile not in state['hand'].split():
                raise ValueError('Self-draw source disagrees with observed own hand')
        elif kind == 'ron':
            if payer == seat or source not in ('discard', 'robbed_ankan', 'robbed_kakan'):
                raise ValueError('Ron requires another public source actor')
            if source == 'discard' and (not state['rivers'][payer].split() or state['rivers'][payer].split()[-1] != tile):
                raise ValueError('Ron discard differs from current source river')
        else: raise ValueError('Unknown legal win kind')
    return copy.deepcopy(context)


def serving_request(request, normalizer=None):
    if normalizer is None:
        from jevmahjong.policy.unified import serving_request as normalizer
    if 'win_context' not in request['state']:
        return normalizer(request)
    context = checked_context(request)
    raw = copy.deepcopy(request); raw['state'].pop('win_context')
    normalized = normalizer(raw)
    normalized['state']['win_context'] = context
    normalized['questions']['action']['instructions'] = INSTRUCTION
    return normalized


def training_record(row, normalizer=None):
    """Same visible request as inference, retaining source labels only outside it."""
    result = serving_request(row, normalizer)
    if '_meta' in row: result['_meta'] = copy.deepcopy(row['_meta'])
    for name, q in row['questions'].items():
        for field in ('label', 'target', 'src'):
            if field in q: result['questions'][name][field] = copy.deepcopy(q[field])
    return result


class RankWinPolicyClient:
    def __init__(self, model): self.model = model
    def system_one(self, **request):
        return self.model.system_one(**serving_request(request))


def continuation_keys(obs, request, actions, features):
    from riichienv import ActionType
    from jevmahjong.evaluation.agents import guarded_indices
    context = checked_context(request); win = next(iter(context['candidates'].values()))
    keys = list(request['questions']['action']['criteria'])
    if win['kind'] == 'ron':
        result = [keys[i] for i, a in enumerate(actions) if a.action_type == ActionType.PASS]
        if len(result) != 1: raise ValueError('Unique native ron PASS required')
    else:
        eligible = set(guarded_indices(obs, actions, features))
        result = [keys[i] for i, a in enumerate(actions)
                  if i in eligible and a.action_type == ActionType.DISCARD]
        if not result: raise ValueError('Ordinary guarded tsumo discard required')
    return result


def select(request, probabilities, continuation, validator=None):
    if validator is None:
        from jevany.api import validate_distribution as validator
    criteria = request['questions']['action']['criteria']; keys = list(criteria)
    values, _ = validator(dict(probabilities), keys)
    wins = [k for k, v in criteria.items() if v['type'] == 'hora']
    if (len(wins) != 1 or not continuation or len(continuation) != len(set(continuation))
            or not set(continuation) <= set(keys)-set(wins)
            or any(criteria[k]['type'] not in ('none', 'dahai') for k in continuation)):
        raise ValueError('Explicit legal win and supported continuation mask required')
    probability = dict(zip(keys, values)); win = wins[0]
    other = max(continuation, key=lambda k: probability[k])
    return win if probability[win] >= probability[other] else other


class RankWinPolicyAgent:
    def __init__(self, client, validator=None):
        self.client = client
        self.ordinary = CompleteLegalClassPolicyAgent(client, validator)
        self.choices = self.ordinary.choices; self.diagnostics = self.ordinary.diagnostics

    def act(self, obs):
        from riichienv import ActionType
        from jevmahjong.core.state import legal_candidates, build_request
        actions = legal_candidates(obs)
        wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
        if len(actions) == 1 or not wins: return self.ordinary.act(obs)
        request = build_request(obs)[0]
        if not in_endgame_scope(request['state']): return self.ordinary.act(obs)
        if len(wins) != 1 or not obs.events:
            self.diagnostics['unsupported_win_shortcuts'] += 1
            return wins[0]
        trigger = obs.events[-1]; trigger = json.loads(trigger) if isinstance(trigger, str) else trigger
        if trigger['type'] not in ('dahai', 'tsumo'):
            self.diagnostics['current_kan_win_shortcuts'] += 1
            return wins[0]
        request, actions, features = build_win_request(obs, trigger)
        allowed = continuation_keys(obs, request, actions, features)
        result = self.client.system_one(**request)
        chosen = select(request, result['answers']['action']['probabilities'], allowed)
        keys = list(request['questions']['action']['criteria'])
        action = actions[keys.index(chosen)]
        self.diagnostics['late_win_model_calls'] += 1
        self.diagnostics['late_win_declines'] += action.action_type not in (ActionType.RON, ActionType.TSUMO)
        self.choices[json.loads(action.to_mjai())['type']] += 1
        return action
