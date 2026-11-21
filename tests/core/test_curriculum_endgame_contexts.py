import unittest
from tools.curriculum_endgame_contexts import score_contexts,starting_scores
from tools.curriculum_endgame_ledger import ledger


class ScoreContextTests(unittest.TestCase):
    def test_all_seats_keep_named_ranks_gaps_and_total_wealth(self):
        for seat in range(4):
            for sticks in (0,2):
                contexts=score_contexts(seat,sticks)
                self.assertEqual(len(contexts),5)
                for name,context in contexts.items():
                    scores=context['scores'];order=sorted(range(4),key=lambda p:(-scores[p],p))
                    self.assertEqual(order.index(seat)+1,context['rank'])
                    self.assertEqual(sum(scores)+1000*sticks,100000)
                    gap={'third_gap3000':3000,'fourth_gap5000':5000,'fourth_gap13000':13000}.get(name)
                    if gap is not None:self.assertEqual(scores[order[1]]-scores[seat],gap)

    def test_large_stick_banks_do_not_silently_change_named_gaps(self):
        contexts=score_contexts(2,19)
        self.assertNotIn('third_gap3000',contexts)
        self.assertNotIn('fourth_gap5000',contexts)

    def test_replay_starting_scores_reverse_only_actual_accepted_deposits(self):
        history=[dict(type='start_kyoku'),dict(type='reach',actor=1),dict(type='reach_accepted',actor=1),dict(type='tsumo',actor=0)]
        initial,bank=starting_scores(history,[43000,24000,22000,10000],1)
        self.assertEqual(initial,[43000,25000,22000,10000]);self.assertEqual(bank,0)

    def test_malformed_or_nonconserving_contexts_rejected(self):
        history=[dict(type='start_kyoku'),dict(type='tsumo',actor=0)]
        for scores,sticks in (([25000]*3,0),([25000]*4,1),([25000]*4,True)):
            with self.assertRaises(ValueError):starting_scores(history,scores,sticks)


class EndgameLedgerTests(unittest.TestCase):
    def test_custom_start_deposit_and_direct_ron_financial_transfers(self):
        events=[dict(type='start_kyoku',scores=[43000,25000,22000,10000],kyotaku=0),dict(type='reach_accepted',actor=2),
            dict(type='hora',actor=2,target=1,deltas=[0,-1000,2000,0]),dict(type='end_game')]
        got=ledger(events,[43000,24000,23000,10000])
        self.assertEqual(got['riichi_deposits'],1);self.assertEqual(got['final_scores'][2],23000)

    def test_existing_sticks_can_make_small_direct_hit_reach_second(self):
        events=[dict(type='start_kyoku',scores=[41000,25000,22000,10000],kyotaku=2),
            dict(type='hora',actor=2,target=1,deltas=[0,-1000,3000,0]),dict(type='end_game')]
        got=ledger(events,[41000,24000,25000,10000])
        self.assertGreater(got['final_scores'][2],got['final_scores'][1])

    def test_unclaimed_pool_uses_native_tied_leader_priority(self):
        events=[dict(type='start_kyoku',scores=[24000]*4,kyotaku=4),dict(type='end_game')]
        got=ledger(events,[28000,24000,24000,24000])
        self.assertEqual(got['components'][0]['final_unclaimed_pool'],4000)

    def test_wrong_handover_draw_or_terminal_scores_fail(self):
        first=dict(type='start_kyoku',scores=[25000]*4,kyotaku=0)
        for events,final in (([first,dict(type='start_kyoku',scores=[26000,24000,25000,25000],kyotaku=0)],[25000]*4),
                             ([first,dict(type='ryukyoku',deltas=[1000,0,0,0])],[26000,25000,25000,25000]),
                             ([first],[26000,24000,25000,25000])):
            with self.assertRaises(ValueError):ledger(events,final)


if __name__=='__main__':unittest.main()
