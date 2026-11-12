import copy
import unittest
from desktop.jev_sidecar.engine import Board, reaction_candidates, request
from desktop.jev_sidecar.tiles import parse
from jevmahjong.training.rewards import opportunity, alternate_indices


def row(hand, tile, kind):
    board = Board(parse(hand), rivers=[[], [], [], []], riichi=[1],
                  reaction={'tile':tile, 'seat':3, 'offered':[kind]})
    values = reaction_candidates(board); payload = request(board, [])
    payload['questions']['action']['criteria'] = {f'a{i}': v['action'] for i,v in enumerate(values)}
    teacher = next(i for i,v in enumerate(values) if v['action']['type']=='none')
    return dict(public_request=payload, scope='calling', eligible=list(range(len(values))), teacher_action=teacher)


class CallProfileTest(unittest.TestCase):
    def test_tanyao_acceleration_requires_closed_hand_and_preserves_candidates(self):
        r = row('3340889m57p0667s', '7s', 'chi'); before = copy.deepcopy(r)
        self.assertTrue(opportunity(r, 'chi_closed_tanyao_gain'))
        self.assertTrue(opportunity(r, 'chi_closed_threat'))
        for index in alternate_indices(r, 'chi_closed_tanyao_gain'):
            self.assertEqual(list(r['public_request']['questions']['action']['criteria'].values())[index]['type'], 'chi')
        self.assertEqual(r, before)
        r['public_request']['state']['riichi'] = []
        self.assertFalse(opportunity(r, 'chi_closed_threat'))
        self.assertTrue(opportunity(r, 'chi_closed_tanyao_gain'))
        r['public_request']['state']['melds'][0] = [{'type':1,'tiles':'5z 5z 5z'}]
        self.assertFalse(opportunity(r, 'chi_closed_tanyao_gain'))

    def test_yakuhai_profile_keeps_pass_as_control_and_is_label_independent(self):
        r = row('40m25556p1346s55z', '5z', 'pon')
        self.assertTrue(opportunity(r, 'pon_closed_yakuhai'))
        call = alternate_indices(r, 'pon_closed_yakuhai')[0]
        r['teacher_action'] = call
        self.assertTrue(opportunity(r, 'pon_closed_yakuhai'))
        self.assertEqual(list(r['public_request']['questions']['action']['criteria'].values())[
            alternate_indices(r, 'pon_closed_yakuhai')[0]]['type'], 'none')
        r = row('123456m345p67s11z', '5s', 'chi')
        self.assertFalse(opportunity(r, 'chi_closed_tanyao_gain'))


if __name__ == '__main__': unittest.main()
