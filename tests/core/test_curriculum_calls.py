import unittest

from tools.audit_curriculum_calls import is_open_hand


class CallCoverageTests(unittest.TestCase):
    def test_concealed_kan_keeps_hand_closed(self):
        for value in (3, 'ankan'):
            state = dict(seat=1, melds=[[], [{'type': value}], [], []])
            self.assertFalse(is_open_hand(state))

    def test_open_meld_with_or_without_concealed_kan_is_open(self):
        for own in ([{'type': 1}], [{'type': 3}, {'type': 1}]):
            state = dict(seat=0, melds=[own, [], [], []])
            self.assertTrue(is_open_hand(state))
