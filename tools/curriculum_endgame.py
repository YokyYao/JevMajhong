"""Explicit stage4 win/decline route, separate from practical guarded v8."""
import math

SCOPE='scheduled_east4_and_extensions_win_windows_v1'


def in_endgame_scope(state):
    """A late-decision scope, never a claim that this hand ends the match."""
    wind,hand=state.get('round_wind'),state.get('hand_number')
    if type(wind) is not int or wind not in range(4) or type(hand) is not int or hand not in range(1,5):
        raise ValueError('A complete public wind/hand number is required')
    return wind>0 or hand==4


def public_placement(state):
    seat=state.get('seat');scores=state.get('scores')
    if type(seat) is not int or seat not in range(4) or not isinstance(scores,list) or len(scores)!=4:
        raise ValueError('Four public scores and a valid own seat are required')
    if any(type(s) not in (int,float) or not math.isfinite(s) for s in scores):
        raise ValueError('Public scores must be finite')
    order=sorted(range(4),key=lambda p:(-scores[p],p))
    return dict(rank=order.index(seat)+1,gap_to_second=scores[order[1]]-scores[seat],
        lead_over_fourth=scores[seat]-scores[order[-1]])


class EndgamePolicyAgent:
    """Route only late legal-win windows to the same model, exactly once.

    Other windows delegate to the pinned PolicyAgent with its existing guards.
    The native menu retains ron PASS or tsumo DISCARD; no synthetic pass/yaku
    rule or score-gap threshold overrides the model. Not enabled in stage1-v3.
    """
    def __init__(self,client):
        from jevmahjong.evaluation.agents import PolicyAgent
        self.client=client;self.ordinary=PolicyAgent(client)
        self.win_windows=0;self.win_declines=0

    def act(self,obs):
        from riichienv import ActionType
        from jevmahjong.core.state import build_request,legal_candidates
        from jevany.api import validate_distribution
        actions=legal_candidates(obs)
        wins=[a for a in actions if a.action_type in (ActionType.RON,ActionType.TSUMO)]
        if len(actions)==1 or not wins:return self.ordinary.act(obs)
        request,actions,_=build_request(obs)
        if not in_endgame_scope(request['state']):return self.ordinary.act(obs)
        public_placement(request['state'])
        keys=list(request['questions']['action']['criteria'])
        result=self.client.system_one(**request)
        p,_=validate_distribution(result['answers']['action']['probabilities'],keys)
        chosen=actions[max(range(len(actions)),key=lambda i:p[i])]
        self.win_windows+=1
        self.win_declines+=chosen.action_type not in (ActionType.RON,ActionType.TSUMO)
        return chosen
