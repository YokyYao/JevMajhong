from collections import Counter
import json
from riichienv import GameRule, RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import legal_candidates
from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.training.rewards import fast_game_stats
from .agents import EfficiencyAgent, CallingOpponent

def play(agent, strong, seed, seat, suite, gamma):
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
    observations = env.reset(); ordinary, caller = EfficiencyAgent(), CallingOpponent()
    steps = 0; opportunities = Counter()
    while not env.done():
        chosen = {}
        for p, obs in observations.items():
            policy = agent if p == seat else (strong if suite == 'strong' else caller) if p == (seat + 1) % 4 and suite != 'closed' else ordinary
            if p == seat:
                kinds = {json.loads(a.to_mjai())['type'] for a in legal_candidates(obs)}
                opportunities.update(kinds & {'ankan', 'daiminkan', 'kakan'})
            chosen[p] = policy.act(obs)
        observations = env.step(chosen); steps += 1
        if steps > 100000:
            raise RuntimeError('Evaluation did not terminate')
    log = env.mjai_log; ledger = settlement_ledger(log, env.scores())
    counts = Counter(e['type'] for e in log if e.get('actor') == seat)
    return {'suite': suite, 'seed': seed, 'seat': seat, 'score': env.scores()[seat], 'rank': env.ranks()[seat],
        'hands': ledger['hands'], 'hand_wins': counts['hora'],
        'dealins': sum(e['type'] == 'hora' and e['target'] == seat and e['actor'] != seat for e in log),
        'calls': counts['chi'] + counts['pon'], 'steps': steps,
        'point_components': ledger['components'][seat], 'settlement_ledger_passed': True,
        'kan_opportunities': dict(opportunities),
        'kan_choices': {k: counts[k] for k in ('ankan', 'daiminkan', 'kakan')},
        'final_scores': list(env.scores()), 'mjai_log': log,
        **fast_game_stats(log, seat, gamma)}
