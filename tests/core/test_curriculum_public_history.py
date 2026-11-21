import unittest
from tools.curriculum_public_inventory import inventory
from tools.curriculum_public_history import public_history,history_flags,sample_history_allocations,tile_type
from tools.curriculum_public_worlds import SamplingExhausted


class HistoryPriorTests(unittest.TestCase):
    def setUp(self):
        self.allocation=dict(opponent_hands={1:list(range(13)),2:list(range(20,33)),3:list(range(40,53))})
        self.start=dict(type='start_kyoku')
        self.trigger=dict(type='tsumo',actor=0)
        self.waits=lambda player,hand:[0] if player==1 else []
    def flags(self,events,riichi=None):
        return history_flags([self.start,*events,self.trigger],0,riichi or [False]*4,self.allocation,self.waits)

    def test_redaction_never_copies_concealed_or_opponent_draw_tiles(self):
        raw=[dict(type='start_kyoku',tehais=[['secret']]*4),dict(type='tsumo',actor=1,pai='9s',secret=True),
            dict(type='dahai',actor=1,pai='5pr',tsumogiri=True),dict(type='tsumo',actor=0,pai='E')]
        got=public_history(raw)
        self.assertEqual(got[0],self.start)
        self.assertEqual(got[1],dict(type='tsumo',actor=1))
        self.assertEqual(got[2]['tile_type'],13)
        self.assertNotIn('secret',str(got));self.assertEqual(tile_type('E'),27)

    def test_temporary_furiten_from_missed_shape_clears_on_own_discard(self):
        offered=dict(type='dahai',actor=2,tile_type=0)
        self.assertTrue(self.flags([offered])['opponent_temporary_furiten'][1])
        self.assertFalse(self.flags([offered,dict(type='dahai',actor=1,tile_type=2)])['opponent_temporary_furiten'][1])

    def test_current_discard_is_not_yet_a_declined_ron(self):
        got=history_flags([self.start,dict(type='dahai',actor=2,tile_type=0)],0,[False]*4,self.allocation,self.waits)
        self.assertFalse(got['opponent_temporary_furiten'][1])

    def test_own_discard_furiten_prevents_new_missed_flags(self):
        got=self.flags([dict(type='dahai',actor=1,tile_type=0),dict(type='dahai',actor=2,tile_type=0)])
        self.assertFalse(got['opponent_temporary_furiten'][1])

    def test_missed_riichi_hypothesis_rejected_without_inventing_permanent_flag(self):
        declared=dict(type='reach_accepted',actor=1)
        offered=dict(type='dahai',actor=2,tile_type=0)
        self.assertFalse(self.flags([declared,offered],[False,True,False,False])['compatible'])
        self.assertTrue(self.flags([offered,dict(type='dahai',actor=1,tile_type=3),declared],[False,True,False,False])['compatible'])
        with self.assertRaises(ValueError):self.flags([],[False,True,False,False])

    def test_kakan_can_create_missed_flag_but_ordinary_hand_cannot_rob_ankan(self):
        self.assertTrue(self.flags([dict(type='kakan',actor=2,tile_type=0)])['opponent_temporary_furiten'][1])
        self.assertFalse(self.flags([dict(type='ankan',actor=2,tile_type=0)])['opponent_temporary_furiten'][1])

    def test_restricted_history_sampler_exhaustion_never_returns_invalid_world(self):
        pool=inventory(0,list(range(13)),[[],[],[],[]],[[],[],[],[]],[100],[False,True,False,False],'ron')
        history=[self.start,dict(type='reach_accepted',actor=1),dict(type='dahai',actor=2,tile_type=0),self.trigger]
        with self.assertRaises(SamplingExhausted) as caught:
            sample_history_allocations(pool,history,[False,True,False,False],17,1,4,lambda hand:0,self.waits)
        self.assertEqual(caught.exception.attempts,4)
        self.assertEqual(caught.exception.accepted,0)

    def test_incomplete_or_settled_history_rejected(self):
        for raw in ([dict(type='tsumo',actor=0,pai='1m')],
                    [self.start,dict(type='hora',actor=0)],
                    [self.start,dict(type='dahai',actor=1,pai='1m')]):
            with self.assertRaises(ValueError):public_history(raw)


if __name__=='__main__':unittest.main()
