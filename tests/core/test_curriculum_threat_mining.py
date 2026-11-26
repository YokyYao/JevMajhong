import unittest
from tools.mine_curriculum_threat_cases import threat_stratum


class ThreatMiningTests(unittest.TestCase):
    def request(self,riichi,criteria):
        return dict(state=dict(seat=2,riichi=riichi),questions=dict(action=dict(criteria=criteria)))

    def test_own_riichi_is_not_counted_as_an_opponent_threat(self):
        menu={'a':{'type':'dahai'},'b':{'type':'ankan'}}
        self.assertIsNone(threat_stratum(self.request([2],menu),'draw'))
        self.assertEqual(threat_stratum(self.request([0,2,3],menu),'draw'),('progression_draw',2))

    def test_three_threat_call_and_one_threat_progression_are_distinct(self):
        self.assertEqual(threat_stratum(self.request([0,1,3],{'a':{'type':'pon'},'b':{'type':'none'}}),'response'),('call_response',3))
        self.assertEqual(threat_stratum(self.request([1],{'a':{'type':'dahai'},'b':{'type':'dahai'}}),'draw'),('progression_draw',1))

    def test_wins_forced_moves_and_noncall_responses_are_excluded(self):
        for menu in ({'a':{'type':'hora'},'b':{'type':'none'}},{'a':{'type':'dahai'}},{'a':{'type':'none'},'b':{'type':'none'}}):
            self.assertIsNone(threat_stratum(self.request([0],menu),'response'))


if __name__=='__main__':unittest.main()
