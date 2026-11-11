"""Same-hand fast-win credit and two-head policy improvement, separate from points."""
import math
from threading import RLock

import torch
from call_policy import is_call_request
from stage2_ablation import FullStage2Client


def scope_for(request):
    kinds = {v['type'] for v in request['questions']['action']['criteria'].values()}
    if is_call_request(request):
        return 'calling'
    # Keep the established encoder routing: every kan-containing request uses
    # oldB, including simultaneous pon/pass alternatives in a daiminkan window.
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


class FastHeadClient:
    """Replace ordinary and call heads, preserving both frozen feature encoders."""
    def __init__(self, ordinary, calling, heads):
        self.ordinary, self.calling, self.heads = ordinary, calling, heads
        self.stage2 = FullStage2Client(calling)
        self.lock = RLock()

    def system_one(self, **request):
        scope = scope_for(request)
        if scope is None:
            return self.ordinary.system_one(**request)
        client = self.ordinary if scope == 'ordinary' else self.calling
        invoke = self.ordinary if scope == 'ordinary' else self.stage2
        model = client.runtime.model
        with self.lock:
            original = model.head
            try:
                model.head = self.heads[scope]
                return invoke.system_one(**request)
            finally:
                model.head = original


def pair_loss(head, row, kl_beta, beta):
    from point_pairwise import point_pairwise_loss
    device = next(head.parameters()).device
    logits = head(row['decide'].to(device), row['options'].to(device))
    reference = row['reference'].to(device).clamp_min(1e-30)
    reference_logits = reference.log()
    ranking = point_pairwise_loss(logits, reference_logits, row['action'],
        row['teacher_action'], row['fast_advantage'], beta)
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
    kinds = {v['type'] for v in row['public_request']['questions']['action']['criteria'].values()}
    if category in ('ankan', 'daiminkan', 'kakan'):
        return category in kinds
    return row['scope'] == category and not kinds & {'ankan', 'daiminkan', 'kakan'}


def alternate_indices(row, category):
    eligible = [i for i in row['eligible'] if i != row['teacher_action']]
    kinds = [v['type'] for v in row['public_request']['questions']['action']['criteria'].values()]
    if category in ('ankan', 'daiminkan', 'kakan') and kinds[row['teacher_action']] != category:
        eligible = [i for i in eligible if kinds[i] == category]
    if not eligible:
        raise ValueError('Missing registered non-teacher alternative')
    return eligible
