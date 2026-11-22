import unittest
from types import SimpleNamespace
from tools.curriculum_decision_history import public_history,history_flags
from tools.curriculum_public_history import history_flags as win_flags


class LiveDecisionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.start=dict(type='start_kyoku')
        self.allocation=dict(opponent_hands={1:list(range(13)),2:list(range(20,33)),3:list(range(40,53))})
        self.waits=lambda p,h:[0] if p==1 else []

    def test_post_call_does_not_invent_a_draw_or_skip_the_answered_discard(self):
        raw=[self.start,dict(type='dahai',actor=3,pai='1m',tsumogiri=True),dict(type='pon',actor=0,consumed=['1m','1m'])]
        history=public_history(raw)
        self.assertEqual(history[-1],dict(type='pon',actor=0))
        got=history_flags(history,0,[False]*4,self.allocation,self.waits)
        self.assertTrue(got['opponent_temporary_furiten'][1])
        self.assertEqual(len(history),3)

    def test_post_call_missed_riichi_hypothesis_is_rejected(self):
        history=[self.start,dict(type='reach_accepted',actor=1),dict(type='dahai',actor=3,tile_type=0),dict(type='chi',actor=0)]
        self.assertFalse(history_flags(history,0,[False,True,False,False],self.allocation,self.waits)['compatible'])

    def test_win_trigger_semantics_are_unchanged(self):
        history=[self.start,dict(type='dahai',actor=2,tile_type=0)]
        args=(history,0,[False]*4,self.allocation,self.waits)
        self.assertEqual(history_flags(*args),win_flags(*args))
        history.append(dict(type='tsumo',actor=0))
        self.assertEqual(history_flags(*args),win_flags(*args))

    def test_private_fields_stripped_and_settled_prefix_rejected(self):
        history=public_history([dict(self.start,tehais=[['secret']]*4),dict(type='tsumo',actor=1,pai='9s'),dict(type='chi',actor=0)])
        self.assertNotIn('secret',str(history));self.assertNotIn('9s',str(history))
        with self.assertRaises(ValueError):public_history([self.start,dict(type='hora',actor=0)])


if __name__=='__main__':unittest.main()
