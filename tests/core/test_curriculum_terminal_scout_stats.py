import unittest
from tools.curriculum_terminal_scout_stats import summarize,utility


def result(top2,rank,net):return dict(top2=top2,rank=rank,net_from_anchor=net)


class ConditionalScoutStatsTests(unittest.TestCase):
    def test_top_two_dominates_bounded_rank_point_tie_break(self):
        self.assertGreater(utility(result(True,2,-100000)),utility(result(False,3,100000)))

    def test_no_observed_success_does_not_create_zero_width_certainty(self):
        pairs=[dict(a=result(False,4,1000),b=result(False,3,3000)) for _ in range(32)]
        got=summarize(pairs,'a','b')
        self.assertTrue(got['both_arms_zero_top2_tail_unresolved'])
        self.assertTrue(got['no_observed_disagreement_is_not_equivalence'])
        self.assertLess(got['paired_discordance_wilson_bonferroni_approx95'][0],0)
        self.assertGreater(got['paired_discordance_wilson_bonferroni_approx95'][1],0)

    def test_paired_signs_and_arm_means_are_not_reversed(self):
        pairs=[dict(a=result(True,2,1000),b=result(False,3,0)),dict(a=result(False,4,0),b=result(False,3,0))]
        got=summarize(pairs,'a','b')
        self.assertEqual(got['paired_first_minus_second_top2'],.5)
        self.assertEqual(got['positive_disagreements'],1)
        self.assertEqual(got['arms']['a']['top2_rate'],.5)


if __name__=='__main__':unittest.main()
