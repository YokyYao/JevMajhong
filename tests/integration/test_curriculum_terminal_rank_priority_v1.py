import unittest

from tools.curriculum_terminal_rank_priority_v1 import choose
from tools.curriculum_public_win_planner_v1 import choose as scalar


def outcome(rank, net):
    return dict(rank=rank, top2=rank <= 2, net_from_anchor=net, full_match_completed=True)


class PublicRankPriority(unittest.TestCase):
    def test_leading_exact_placement_tie_takes_current_win(self):
        pairs = [{'win': outcome(2, 1000), 'pass': outcome(2, 6000)} for _ in range(4)]
        self.assertEqual(scalar(pairs, 'win', 'pass')['selected'], 'pass')
        self.assertEqual(choose(pairs, 'win', 'pass', 2)['selected'], 'win')
        self.assertEqual(choose(pairs, 'win', 'pass', 4)['selected'], 'pass')

    def test_fourth_may_risk_last_place_for_any_sampled_top2_gain(self):
        pairs = [{'win': outcome(3, 50000), 'pass': outcome(4, -50000)} for _ in range(100)]
        pairs[0]['pass'] = outcome(2, 0)
        self.assertEqual(scalar(pairs, 'win', 'pass')['selected'], 'win')
        self.assertEqual(choose(pairs, 'win', 'pass', 4)['selected'], 'pass')

    def test_terminal_rank_precedes_net_and_invalid_public_rank_fails(self):
        pairs = [{'win': outcome(3, 0), 'pass': outcome(4, 20000)} for _ in range(4)]
        self.assertEqual(choose(pairs, 'win', 'pass', 4)['selected'], 'win')
        for rank in (0, 5, True):
            with self.assertRaises(ValueError): choose(pairs, 'win', 'pass', rank)


if __name__ == '__main__': unittest.main()
