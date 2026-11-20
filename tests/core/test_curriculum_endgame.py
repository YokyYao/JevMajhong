import unittest
from tools.curriculum_endgame import in_endgame_scope,public_placement


class EndgamePublicTests(unittest.TestCase):
    def test_east4_and_extensions_are_scope_not_automatic_terminal(self):
        self.assertFalse(in_endgame_scope(dict(round_wind=0,hand_number=3)))
        self.assertTrue(in_endgame_scope(dict(round_wind=0,hand_number=4)))
        self.assertTrue(in_endgame_scope(dict(round_wind=1,hand_number=1)))
        for state in ({},dict(round_wind=0,hand_number=0),dict(round_wind=True,hand_number=4)):
            with self.assertRaises(ValueError):in_endgame_scope(state)

    def test_own_third_and_fourth_gaps_and_native_tie_order(self):
        scores=[35000,28000,25000,12000]
        self.assertEqual(public_placement(dict(seat=2,scores=scores)),
            dict(rank=3,gap_to_second=3000,lead_over_fourth=13000))
        self.assertEqual(public_placement(dict(seat=3,scores=scores))['gap_to_second'],16000)
        self.assertEqual(public_placement(dict(seat=2,scores=[25000]*4))['rank'],3)
        with self.assertRaises(ValueError):public_placement(dict(seat=0,scores=[float('nan')]*4))
