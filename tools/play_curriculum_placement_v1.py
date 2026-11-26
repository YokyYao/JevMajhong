"""Explicit-public-event full East loop for future placement candidates only."""
from collections import Counter


def play(agent,seed,seat,profile):
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.agents import EfficiencyAgent,CallingOpponent
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.training.rewards import fast_game_stats
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
    observations=env.reset();ordinary=EfficiencyAgent();caller=CallingOpponent()
    for _ in range(30000):
        if env.done():break
        event=env.mjai_log[-1]
        selected={}
        for player,obs in observations.items():
            if player==seat:selected[player]=agent.act(obs,current_public_event=event)
            else:selected[player]=(caller if profile=='calling' and player==(seat+1)%4 else ordinary).act(obs)
        observations=env.step(selected)
    if not env.done():raise RuntimeError('Truncated East match')
    log=env.mjai_log;ledger=settlement_ledger(log,env.scores());choices=Counter(e['type'] for e in log if e.get('actor')==seat)
    return dict(seed=seed,seat=seat,profile=profile,rank=env.ranks()[seat],net_points=env.scores()[seat]-25000,
        final_scores=list(env.scores()),hands=ledger['hands'],wins=choices['hora'],
        dealin_hands=hand_dealin_count(log,seat),completed=True,ledger_passed=True,
        choices={k:choices[k] for k in ('chi','pon','ankan','daiminkan','kakan','reach')},
        **fast_game_stats(log,seat,.98),mjai_log=log)

