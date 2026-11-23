import copy
import unittest

from tools.curriculum_public_win_planner_v1 import choose, public_seed


def terminal(rank, net=0):
    return dict(rank=rank, top2=rank <= 2, net_from_anchor=net, full_match_completed=True)


class PublicWinPlannerTests(unittest.TestCase):
    def test_top_two_can_outweigh_more_fourths(self):
        pairs = [{'win': terminal(3), 'pass': terminal(2 if i == 0 else 4)} for i in range(4)]
        result = choose(pairs, 'win', 'pass')
        self.assertEqual(result['selected'], 'pass')
        self.assertFalse(result['confidence_gate_used'])

    def test_reaching_second_by_accepting_is_preferred(self):
        self.assertEqual(choose([{'win': terminal(2), 'pass': terminal(3, 20000)}],
                                'win', 'pass')['selected'], 'win')

    def test_exact_tie_takes_the_legal_win(self):
        self.assertEqual(choose([{'win': terminal(2), 'pass': terminal(2)}],
                                'win', 'pass')['selected'], 'win')

    def test_partial_or_inconsistent_terminal_evidence_rejected(self):
        for value in (dict(terminal(2), full_match_completed=False),
                      dict(terminal(2), top2=False), dict(terminal(2), net_from_anchor=float('nan'))):
            with self.assertRaises(ValueError):
                choose([{'win': value, 'pass': terminal(3)}], 'win', 'pass')
        with self.assertRaises(ValueError):
            choose([{'win': terminal(2)}], 'win', 'pass')

    def test_public_seed_ignores_mapping_order_but_retains_score_context(self):
        request = {'state': {'scores': [25000] * 4, 'seat': 0}}
        history = [{'type': 'start_kyoku'}, {'type': 'dahai', 'actor': 1, 'tile_type': 0}]
        reverse = {'state': {'seat': 0, 'scores': [25000] * 4}}
        self.assertEqual(public_seed(request, history, 1), public_seed(reverse, history, 1))
        changed = copy.deepcopy(request); changed['state']['scores'] = [22000, 25000, 26000, 27000]
        self.assertNotEqual(public_seed(request, history, 1), public_seed(changed, history, 1))


if __name__ == '__main__':
    unittest.main()
