import unittest
from tools.curriculum_public_inventory import inventory
from tools.curriculum_public_worlds import sample_allocations, SamplingExhausted


class PublicAllocationTests(unittest.TestCase):
    def setUp(self):
        self.pool=inventory(0,list(range(13)),[[],[20],[],[]],[[],[],[],[]],[100],
            [False,True,False,False],'ron')

    def test_complete_physical_partition_is_reproducible_and_checked(self):
        seen=[]
        def fixture(hand):
            seen.append(list(hand))
            return 0
        got=sample_allocations(self.pool,17,3,3,fixture)
        self.assertEqual(got,sample_allocations(self.pool,17,3,3,lambda hand:0))
        self.assertEqual(len(seen),3)
        for sample in got['allocations']:
            tiles=[tile for hand in sample['opponent_hands'].values() for tile in hand]+sample['unknown_nonhand_tiles']
            self.assertEqual(sorted(tiles),self.pool['unseen_tiles'])
            self.assertEqual({p:len(h) for p,h in sample['opponent_hands'].items()},self.pool['opponent_concealed_counts'])
        self.assertFalse(got['native_environment_constructed'])
        self.assertFalse(got['calibrated_public_history_posterior'])

    def test_unsatisfied_riichi_never_falls_back_to_invalid_or_partial_worlds(self):
        with self.assertRaises(SamplingExhausted) as caught:
            sample_allocations(self.pool,17,2,3,lambda hand:1)
        self.assertEqual((caught.exception.attempts,caught.exception.accepted),(3,0))

    def test_all_riichi_players_must_satisfy_constraint(self):
        self.pool['riichi_tenpai_required']=[1,2]
        calls=[]
        def fixture(hand):
            calls.append(1)
            return 0 if len(calls)%2 else 1
        with self.assertRaises(SamplingExhausted): sample_allocations(self.pool,17,1,3,fixture)
        self.assertEqual(len(calls),6)

    def test_unbounded_or_inconsistent_inventory_rejected(self):
        for count,limit in ((0,1),(1,2000001),(2,1)):
            with self.assertRaises(ValueError): sample_allocations(self.pool,17,count,limit,lambda hand:0)
        self.pool['known_tiles'].append(self.pool['unseen_tiles'][0])
        with self.assertRaises(ValueError): sample_allocations(self.pool,17,1,1,lambda hand:0)


if __name__=='__main__':unittest.main()
