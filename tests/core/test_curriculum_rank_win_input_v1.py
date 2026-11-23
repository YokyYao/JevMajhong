"""Critical public input boundaries and legal-win tie behavior."""
import copy
import unittest

from tools.curriculum_rank_win_input_v1 import serving_request, training_record, select, INSTRUCTION
from tools.curriculum_win_context import SCHEMA


def request():
    return dict(state=dict(seat=2, dealer=0, round_wind=0, hand_number=4, scores=[30000,25000,22000,23000],
        hand='1m 1m 1m 2m 3m 4m 5m 6m 7m 2p 3p 4p 5z', rivers=['', '9p', '', ''],
        melds=[[],[],[],[]], riichi=[], honba=0, sticks=0,
        win_context=dict(schema=SCHEMA, candidates={'win':dict(kind='ron', source='discard', source_seat=1, tile='9p')})),
        questions=dict(action=dict(type='choice', instructions='original',
            criteria={'pass':dict(type='none'), 'win':dict(type='hora', pai='9p')},
            label='pass', target={'pass':.5,'win':.5}, src='private-source')),
        _meta=dict(id='private-id', group_id='family', simulated_return=999999))


class RankWinInputTests(unittest.TestCase):
    def test_training_and_serving_visible_input_match_without_target_metadata(self):
        row = request(); original = copy.deepcopy(row); train = training_record(row)
        visible = serving_request(row)
        self.assertEqual(serving_request(train), visible)
        self.assertEqual(train['state'], visible['state'])
        self.assertEqual(train['questions']['action']['target'], row['questions']['action']['target'])
        self.assertNotIn('_meta', visible)
        self.assertEqual(set(visible['questions']['action']), {'type','instructions','criteria'})
        self.assertIn(INSTRUCTION, visible['questions']['action']['instructions'])
        self.assertEqual(row, original)

    def test_hidden_or_inconsistent_source_fields_are_rejected(self):
        for key, value in [('opponent_hand','1m 2m'), ('future_score', 99000)]:
            row = request(); row['state']['win_context']['candidates']['win'][key] = value
            with self.assertRaises(ValueError): serving_request(row)
        for payer in (2, True):
            row = request(); row['state']['win_context']['candidates']['win']['source_seat'] = payer
            with self.assertRaises(ValueError): serving_request(row)
        row = request(); row['state']['wall'] = ['1m']
        with self.assertRaises(ValueError): serving_request(row)
        row = request(); row['state']['rivers'][1] = '8p'
        with self.assertRaises(ValueError): serving_request(row)

    def test_exact_soft_tie_takes_win_regardless_of_menu_order(self):
        row = request()
        self.assertEqual(select(row, {'pass':.5,'win':.5}, ['pass']), 'win')
        self.assertEqual(select(row, {'pass':.6,'win':.4}, ['pass']), 'pass')
        with self.assertRaises(ValueError): select(row, {'pass':float('nan'),'win':.4}, ['pass'])
        with self.assertRaises(ValueError): select(row, {'pass':.5,'win':.5}, ['win'])
