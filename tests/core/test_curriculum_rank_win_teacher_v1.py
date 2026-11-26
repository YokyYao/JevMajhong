"""Behavioral regression: allow fourth-place pursuit and accept leading ties."""
import unittest
from tools.curriculum_rank_win_teacher_v1 import soft_target


def outcome(rank, net):
    return dict(rank=rank, top2=rank <= 2, net_from_anchor=net, full_match_completed=True)


class RankTeacherTests(unittest.TestCase):
    def test_fourth_can_chase_top2_despite_worse_fourth_escape(self):
        pairs = [{'win': outcome(3, 1000), 'continue': outcome(2 if i % 4 == 0 else 4, -1000)}
                 for i in range(32)]
        result = soft_target(pairs, 'win', 'continue', 4, seed=17)
        self.assertGreater(result['target']['continue'], .9)
        self.assertEqual(result['label'], 'continue')
        self.assertFalse(result['fourth_place_escape_veto'])

    def test_leading_placement_tie_does_not_chase_net_only(self):
        pairs = [{'win': outcome(2, 1000), 'continue': outcome(2, 12000)} for _ in range(32)]
        result = soft_target(pairs, 'win', 'continue', 2, seed=17)
        self.assertEqual(result['label'], 'win')
        self.assertLess(result['target']['continue'], .01)
        self.assertAlmostEqual(sum(result['target'].values()), 1.)

    def test_evenly_conflicting_worlds_keep_both_targets(self):
        pairs = [{'win': outcome(2 if i % 2 else 4, 0),
                  'continue': outcome(4 if i % 2 else 2, 0)} for i in range(32)]
        result = soft_target(pairs, 'win', 'continue', 4, seed=17)
        self.assertGreater(result['target']['continue'], .3)
        self.assertLess(result['target']['continue'], .7)
        self.assertFalse(result['uncertain_rows_filtered'])
