import unittest
from tools.curriculum_public_inventory import inventory, from_observation


class PublicInventoryTests(unittest.TestCase):
    def test_closed_ron_tile_conservation_and_riichi_constraint(self):
        got = inventory(0,list(range(13)),[[],[20],[],[]],[[],[],[],[]],[100],
            [False,True,False,False],'ron')
        self.assertEqual(got['opponent_concealed_counts'], {1:13,2:13,3:13})
        self.assertEqual(got['riichi_tenpai_required'], [1])
        self.assertEqual(len(got['known_tiles']),15)
        self.assertEqual(len(got['unseen_tiles'])+len(got['known_tiles']),136)
        self.assertEqual(got['unknown_nonhand_tile_count'],82)
        self.assertFalse(set(got['known_tiles']) & set(got['unseen_tiles']))

    def test_called_discard_counts_once_and_kan_changes_effective_count_by_three(self):
        melds=[[],[dict(tiles=[20,21,22],called_tile=20)],[dict(tiles=[24,25,26,27],called_tile=None)],[]]
        got=inventory(0,list(range(14)),[[20],[],[],[]],melds,[100],[False]*4,'tsumo')
        self.assertEqual(got['public_called_tile_duplicates'],1)
        self.assertEqual(got['opponent_concealed_counts'],{1:10,2:10,3:13})
        self.assertEqual(len(got['known_tiles']),22)

    def test_invalid_physical_duplicates_and_wrong_own_count_rejected(self):
        for river in ([[1],[],[],[]],[[20],[20],[],[]],[[136],[],[],[]]):
            with self.assertRaises(ValueError):
                inventory(0,list(range(13)),river,[[],[],[],[]],[100],[False]*4,'ron')
        with self.assertRaises(ValueError):
            inventory(0,list(range(13)),[[],[],[],[]],[[],[],[],[]],[100],[False]*4,'tsumo')

    def test_omniscient_observation_is_rejected(self):
        class LeakingObservation:
            player_id=0
            hand=list(range(13))
            def to_dict(self): return dict(hands=[self.hand,[20],[],[]])
        with self.assertRaises(ValueError): from_observation(LeakingObservation(),'ron')


if __name__ == '__main__': unittest.main()
