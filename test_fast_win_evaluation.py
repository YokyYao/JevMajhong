"""Known paired outcomes must drive score/speed decisions on the complete grid."""
import copy
import unittest
from scripts.evaluate_fast_win import summarize


def example():
    cfg = {'seed_start': 100, 'seeds': 4, 'stage': 'development', 'fast_gamma': .995}
    rows = []
    for seed in range(100, 104):
        for suite in ('closed', 'mixed', 'strong'):
            for seat in range(4):
                for policy in ('reference', 'candidate'):
                    changed = policy == 'candidate'
                    scores = [25000] * 4
                    delta = [0] * 4
                    if changed:
                        delta[seat] = 100; delta[(seat + 1) % 4] = -100
                        scores = [a + b for a, b in zip(scores, delta)]
                    events = [{'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0}]
                    events += [{'type': 'tsumo'} for _ in range(30)]
                    events += [{'type': 'hora', 'actor': seat, 'target': (seat + 1) % 4, 'deltas': delta}] if changed else [
                        {'type': 'ryukyoku', 'deltas': delta}]
                    events += [{'type': 'end_kyoku'}, {'type': 'end_game'}]
                    rows.append(dict(seed=seed, suite=suite, seat=seat, policy=policy,
                        score=25100 if changed else 25000, rank=1, hands=1,
                        hand_wins=int(changed), dealins=0, calls=int(changed),
                        point_components={'win_income': 100} if changed else {'draw_settlement': 0},
                        settlement_ledger_passed=True, fast_hand_count=1, fast_hand_wins=int(changed),
                        fast_reward_sum=.995 ** 30 if changed else 0., winning_table_draws=30 if changed else 0,
                        mjai_log=events, final_scores=scores,
                        kan_opportunities={'ankan': 1}, kan_choices={'ankan': int(changed)}))
    return rows, cfg


class FastEvaluationTest(unittest.TestCase):
    def test_known_paired_speed_and_points_use_complete_whole_walls(self):
        rows, cfg = example(); result = summarize(rows, cfg)
        self.assertAlmostEqual(result['fast_reward_paired']['gain'], .995 ** 30)
        self.assertEqual(result['overall']['candidate_minus_reference']['gain'], 100)
        self.assertTrue(result['eligible_for_confirmation'])
        self.assertEqual(result['overall']['policies']['candidate']['kan_choices']['ankan'], 48)

    def test_speed_alone_with_point_loss_does_not_trigger_confirmation(self):
        rows, cfg = example()
        for r in rows:
            if r['policy'] == 'candidate':
                seat=r['seat']; other=(seat+1)%4; delta=[0]*4; delta[seat]=-200;delta[other]=200
                scores=[a+b for a,b in zip(r['final_scores'],delta)]
                r['mjai_log'] = r['mjai_log'][:-1] + [
                    {'type':'start_kyoku','scores':r['final_scores'],'kyotaku':0},
                    {'type':'hora','actor':other,'target':seat,'deltas':delta},
                    {'type':'end_kyoku'},{'type':'end_game'}]
                r.update(score=24900, point_components={'win_income':100,'ron_payments':-200},
                    final_scores=scores,hands=2,fast_hand_count=2,dealins=1)
        self.assertFalse(summarize(rows, cfg)['eligible_for_confirmation'])

    def test_missing_duplicate_and_wrong_hand_credit_rejected(self):
        rows, cfg = example()
        with self.assertRaises(ValueError):
            summarize(rows[:-1], cfg)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], cfg)
        rows[0]['fast_hand_count'] = 2
        with self.assertRaises(ValueError):
            summarize(rows, cfg)


if __name__ == '__main__':
    unittest.main()
