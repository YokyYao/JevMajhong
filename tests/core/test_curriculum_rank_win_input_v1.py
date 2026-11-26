"""Critical public input boundaries and legal-win tie behavior."""
import copy
import math
import unittest

from tools.curriculum_rank_win_input_v1 import serving_request, training_record, select, INSTRUCTION
from tools.curriculum_win_context import SCHEMA


def ordinary_fixture(value):
    """Explicit stand-in; pinned-runtime equality is checked by the GPU entry."""
    if set(value['state'])-set(request()['state']) or 'win_context' in value['state']:
        raise ValueError('Only the ordinary fixture state is accepted')
    q = value['questions']['action']
    return dict(state=copy.deepcopy(value['state']), questions=dict(action={
        'type': q['type'], 'instructions': 'ordinary fixture', 'criteria': copy.deepcopy(q['criteria'])}))


def distribution_fixture(raw, keys):
    if set(raw) != set(keys): raise ValueError('Probability menu differs')
    values = [float(raw[k]) for k in keys]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values)-1) > 1e-6:
        raise ValueError('Invalid probability distribution')
    return values, sum(values)


def visible(value): return serving_request(value, normalizer=ordinary_fixture)


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
        row = request(); original = copy.deepcopy(row); train = training_record(row, normalizer=ordinary_fixture)
        public = visible(row)
        self.assertEqual(visible(train), public)
        self.assertEqual(train['state'], public['state'])
        self.assertEqual(train['questions']['action']['target'], row['questions']['action']['target'])
        self.assertNotIn('_meta', public)
        self.assertEqual(set(public['questions']['action']), {'type','instructions','criteria'})
        self.assertIn(INSTRUCTION, public['questions']['action']['instructions'])
        self.assertEqual(row, original)

    def test_hidden_or_inconsistent_source_fields_are_rejected(self):
        for key, value in [('opponent_hand','1m 2m'), ('future_score', 99000)]:
            row = request(); row['state']['win_context']['candidates']['win'][key] = value
            with self.assertRaises(ValueError): visible(row)
        for payer in (2, True):
            row = request(); row['state']['win_context']['candidates']['win']['source_seat'] = payer
            with self.assertRaises(ValueError): visible(row)
        row = request(); row['state']['wall'] = ['1m']
        with self.assertRaises(ValueError): visible(row)
        row = request(); row['state']['rivers'][1] = '8p'
        with self.assertRaises(ValueError): visible(row)

    def test_exact_soft_tie_takes_win_regardless_of_menu_order(self):
        row = request()
        self.assertEqual(select(row, {'pass':.5,'win':.5}, ['pass'], distribution_fixture), 'win')
        self.assertEqual(select(row, {'pass':.6,'win':.4}, ['pass'], distribution_fixture), 'pass')
        with self.assertRaises(ValueError): select(row, {'pass':float('nan'),'win':.4}, ['pass'], distribution_fixture)
        with self.assertRaises(ValueError): select(row, {'pass':.5,'win':.5}, ['win'], distribution_fixture)
