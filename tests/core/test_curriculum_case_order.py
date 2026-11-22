import unittest
from tools.curriculum_case_order import ordered_seats,canonical_trace


class SimultaneousCaseOrderTests(unittest.TestCase):
    def test_multi_ron_source_selection_is_independent_of_native_map_order(self):
        first={3:'ron3',0:'ron0',2:'pass2'};second={2:'pass2',0:'ron0',3:'ron3'}
        self.assertEqual(ordered_seats(first),ordered_seats(second))
        self.assertEqual([p for p,a in ordered_seats(first) if a.startswith('ron')][0],0)

    def test_canonicalization_preserves_times_tiles_and_consumed_sets(self):
        a=dict(player=3,type=4,tile=18,consumed=[21,20]);b=dict(player=0,type=1,tile=13,consumed=[])
        c=dict(player=2,type=0,tile=7,consumed=[])
        self.assertEqual(canonical_trace([[a,b],[c]]),canonical_trace([[b,dict(a,consumed=[20,21])],[c]]))
        self.assertNotEqual(canonical_trace([[a,b],[c]]),canonical_trace([[c],[a,b]]))
        self.assertNotEqual(canonical_trace([[a,b]]),canonical_trace([[dict(a,tile=19),b]]))

    def test_invalid_native_seat_and_duplicate_actor_are_rejected(self):
        with self.assertRaises(ValueError):ordered_seats({True:'invalid'})
        a=dict(player=0,type=1,tile=13,consumed=[])
        with self.assertRaises(ValueError):canonical_trace([[a,a]])


if __name__=='__main__':unittest.main()
