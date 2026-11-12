"""Same-hand fast-win credit and two-head policy improvement, separate from points."""
import math
from threading import RLock

import torch
from torch.nn import functional as F
from jevmahjong.policy.routing import is_call_request
from jevmahjong.policy.routing import HeadPolicyClient


def scope_for(request):
    kinds = {v['type'] for v in request['questions']['action']['criteria'].values()}
    if is_call_request(request):
        return 'calling'
    # Keep the established encoder routing: every kan-containing request uses
    # Jev Discard v1, including pon/pass alternatives in a daiminkan window.
    if kinds and kinds <= {'dahai', 'reach', 'ankan', 'daiminkan', 'kakan', 'pon', 'chi', 'none'}:
        return 'ordinary'
    return None


def hand_outcomes(log):
    """Private outcome labels only; never expose future events as model inputs."""
    hands = []
    for event in log:
        kind = event['type']
        if kind == 'start_kyoku':
            if hands and not hands[-1]['complete']:
                raise ValueError('Previous hand lacks end_kyoku')
            hands.append({'draws': 0, 'winners': [], 'complete': False})
        elif hands:
            current = hands[-1]
            if kind == 'tsumo':
                current['draws'] += 1
            elif kind == 'hora':
                if current['complete']:
                    raise ValueError('Win after hand end')
                current['winners'].append({'actor': event['actor'], 'draws': current['draws']})
            elif kind == 'end_kyoku':
                current['complete'] = True
    return hands


def fast_reward(hand, seat, draws_before, gamma):
    if not 0 < gamma < 1 or seat not in range(4) or not hand['complete']:
        raise ValueError('Completed hand, legal seat and discount required')
    if not 0 <= draws_before <= hand['draws']:
        raise ValueError('Decision cannot follow the outcome')
    wins = [w for w in hand['winners'] if w['actor'] == seat]
    if len(wins) > 1:
        raise ValueError('Duplicate winner in one hand')
    if not wins:
        return 0.
    remaining = wins[0]['draws'] - draws_before
    if remaining < 0:
        raise ValueError('Cannot credit a win preceding the decision')
    return gamma ** remaining


def credit_pair(control_log, branch_log, hand_index, seat, draws_before, gamma):
    control, branch = hand_outcomes(control_log), hand_outcomes(branch_log)
    if hand_index < 0 or hand_index >= min(len(control), len(branch)):
        raise ValueError('Intervention hand missing')
    a = fast_reward(control[hand_index], seat, draws_before, gamma)
    b = fast_reward(branch[hand_index], seat, draws_before, gamma)
    return {'control_fast_reward': a, 'branch_fast_reward': b, 'fast_advantage': b - a}


def fast_game_stats(log, seat, gamma):
    hands = hand_outcomes(log)
    if not hands or any(not h['complete'] for h in hands):
        raise ValueError('Complete hands required')
    wins = [w for h in hands for w in h['winners'] if w['actor'] == seat]
    return {'fast_reward_sum': sum(fast_reward(h, seat, 0, gamma) for h in hands),
        'winning_table_draws': sum(w['draws'] for w in wins), 'fast_hand_count': len(hands),
        'fast_hand_wins': len(wins)}




def pair_loss(head, row, kl_beta, beta, reference_margin_weight=1.):
    device = next(head.parameters()).device
    logits = head(row['decide'].to(device), row['options'].to(device))
    reference = row['reference'].to(device).clamp_min(1e-30)
    reference_logits = reference.log()
    ranking = pairwise_preference_loss(logits, reference_logits, row['action'],
        row['teacher_action'], row.get('training_advantage', row['fast_advantage']), beta, reference_margin_weight)
    eligible = row['eligible']
    logp = logits[eligible].float().log_softmax(-1)
    target = reference[eligible]; target = target / target.sum()
    kl = (target * (target.log() - logp)).sum()
    return ranking + kl_beta * kl, ranking, kl


def validate_credit(row, game, control_log, gamma):
    if row['action'] == row['teacher_action'] or row['action'] not in row['eligible']:
        raise ValueError('Intervention must change an eligible action')
    expected = credit_pair(control_log, game['mjai_log'], row['hand_index'], game['seat'],
        row['draws_before'], gamma)
    for key, value in expected.items():
        if not math.isfinite(row[key]) or abs(row[key] - value) > 1e-12:
            raise ValueError('Wrong same-hand reward: ' + key)
    if row['scope'] != scope_for(row['public_request']):
        raise ValueError('Trace representation has wrong policy scope')


def opportunity(row, category):
    if category.endswith('_in_kan'):
        from jevmahjong.policy.kan import is_kan_request
        return is_kan_request(row['public_request']) and opportunity(row,category[:-len('_in_kan')])
    for suffix in ('_open_threat', '_threat'):
        if category.endswith(suffix):
            state = row['public_request']['state']; seat = state['seat']
            if not set(state['riichi']) - {seat}: return False
            if suffix == '_open_threat':
                from riichienv import MeldType
                if not any(m['type'] != int(MeldType.Ankan) for m in state['melds'][seat]): return False
            return opportunity(row, category[:-len(suffix)])
    kinds = {v['type'] for v in row['public_request']['questions']['action']['criteria'].values()}
    if category in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan'):
        return category in kinds
    return row['scope'] == category and not kinds & {'ankan', 'daiminkan', 'kakan'}


def alternate_indices(row, category):
    while True:
        suffix=next((s for s in ('_open_threat','_threat','_in_kan') if category.endswith(s)),None)
        if suffix is None:break
        category=category[:-len(suffix)]
    eligible = [i for i in row['eligible'] if i != row['teacher_action']]
    kinds = [v['type'] for v in row['public_request']['questions']['action']['criteria'].values()]
    if category in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan') and kinds[row['teacher_action']] != category:
        eligible = [i for i in eligible if kinds[i] == category]
    if not eligible:
        raise ValueError('Missing registered non-teacher alternative')
    return eligible

def pairwise_preference_loss(logits, reference_logits, sampled, teacher, advantage, beta=1., reference_margin_weight=1.):
    if not math.isfinite(advantage) or not math.isfinite(beta) or beta <= 0:
        raise ValueError('Pairwise score weight and margin scale must be finite')
    if not math.isfinite(reference_margin_weight) or not 0 <= reference_margin_weight <= 1:
        raise ValueError('Reference margin weight must be between zero and one')
    if advantage != 0 and sampled == teacher:
        raise ValueError('An unchanged action cannot have a nonzero paired-point label')
    if advantage == 0:
        return logits.sum()*0
    margin = logits[sampled] - logits[teacher]
    reference_margin = (reference_logits[sampled] - reference_logits[teacher]).detach()
    signed_change = (1 if advantage > 0 else -1)*beta*(margin-reference_margin_weight*reference_margin)
    return abs(advantage)*F.softplus(-signed_change)
