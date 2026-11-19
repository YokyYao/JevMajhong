import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from tools.evaluate_fourth_chase_development import validate_config,candidate_agent,summarize


def outcome(rank):
    return {'rank':rank,'hands_ended_after_window':1,'dealins_after_window':int(rank==4),
        'net_points_from_window':0,'own_wins_after_window':int(rank<=2)}


def rows():
    values=[('standard',3,2,2),('standard',4,4,2),('standard',4,3,4),('standard',1,1,1),
            ('targeted',3,3,3),('targeted',4,2,2),('targeted',4,3,4)]
    return [{'cohort':cohort,'seed':i,'initial_rank':initial,'baseline':outcome(b),'candidate':outcome(c)}
            for i,(cohort,initial,b,c) in enumerate(values)]


class FourthScopeDevelopmentTests(unittest.TestCase):
    def test_same_weight_agent_changes_only_current_fourth_branch(self):
        client=Mock();agent=candidate_agent(client)
        self.assertTrue(agent.placement_aware);self.assertFalse(agent.base.placement_aware)
        self.assertIs(agent.model,client);self.assertIs(agent.base.model,client)
        client.system_one.assert_not_called()

    def test_larger_fourth_tail_does_not_veto_a_top2_gain(self):
        result=summarize(rows())
        self.assertGreater(result['fourth_window_top2_gain'],0.)
        self.assertGreater(result['fourth_rates']['candidate']['fourth_rate'],result['fourth_rates']['baseline']['fourth_rate'])
        self.assertTrue(result['eligible_for_new_confirmation'])
        self.assertTrue(result['fourth_escape_is_reporting_only'])

    def test_third_regression_still_blocks_confirmation(self):
        values=rows();values[4]['baseline']=outcome(2)
        result=summarize(values)
        self.assertFalse(result['third_no_drop_each_cohort']);self.assertFalse(result['eligible_for_new_confirmation'])

    def test_repeated_windows_do_not_count_as_independent_source_groups(self):
        values=rows();extra=copy.deepcopy(values[1]);values.append(extra)
        result=summarize(values)
        self.assertEqual(result['fourth_positions'],5);self.assertEqual(result['fourth_source_groups'],4)

    def test_registration_forbids_fresh_sources_and_an_escape_veto(self):
        path=Path(__file__).resolve().parents[2]/'configs/training/endgame-fourth-scope-consumed-development-v1.json'
        config=json.loads(path.read_text());validate_config(config)
        for change in ({'standard_seed_start':2026118900},{'fourth_escape_constraint':'veto'},
                       {'independent_confirmation':True},{'new_candidate_rollouts':166}):
            with self.assertRaises(ValueError):validate_config(dict(config,**change))


if __name__=='__main__':unittest.main()
