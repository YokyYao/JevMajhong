import copy
import unittest

from tools.curriculum_placement_input_v1 import serving_request, training_record, INSTRUCTION
from tools.curriculum_rank_win_input_v1 import serving_request as win_projection
from tools.curriculum_win_context import SCHEMA as WIN_SCHEMA


def fixture(late):
    return dict(state=dict(seat=2, dealer=0, scores=[40000,25000,22000,13000],
        round_wind=0, hand_number=4 if late else 2, hand='1m 2m',
        rivers=['','9p','',''], melds=[[],[],[],[]]),
        questions=dict(action=dict(type='choice', instructions='legacy',
            criteria=dict(a=dict(type='dahai',pai='1m'), b=dict(type='dahai',pai='2m')),
            label='b', target={'a':.25,'b':.75}, src='private')),
        _meta=dict(id='private', group_id='family', future_score=999999))


def ordinary(request):
    public=copy.deepcopy(request)
    public.pop('_meta',None)
    q=public['questions']['action']
    public['questions']['action']={k:q[k] for k in ('type','instructions','criteria')}
    public['questions']['action']['instructions']=(
        'Prioritize winning the current hand. Consider legal yaku. sh=shanten. Call note.')
    return public


class PlacementInstructionTests(unittest.TestCase):
    def test_early_inputs_unchanged_and_late_nonwin_has_one_coherent_priority(self):
        early=fixture(False)
        self.assertEqual(serving_request(early,ordinary),win_projection(early,ordinary))
        late=fixture(True); original=copy.deepcopy(late); got=serving_request(late,ordinary)
        self.assertEqual(got['state'],late['state'])
        self.assertEqual(got['questions']['action']['criteria'],late['questions']['action']['criteria'])
        self.assertIn(INSTRUCTION,got['questions']['action']['instructions'])
        self.assertNotIn('Prioritize winning the current hand',got['questions']['action']['instructions'])
        self.assertIn('sh=shanten. Call note.',got['questions']['action']['instructions'])
        self.assertEqual(late,original)

    def test_training_targets_are_retained_outside_same_visible_input(self):
        row=fixture(True); got=training_record(row,ordinary)
        self.assertEqual(serving_request(got,ordinary),serving_request(row,ordinary))
        self.assertEqual(got['_meta'],row['_meta'])
        self.assertEqual(got['questions']['action']['target'],row['questions']['action']['target'])
        self.assertNotIn('_meta',serving_request(got,ordinary))
        self.assertNotIn('target',serving_request(got,ordinary)['questions']['action'])

    def test_win_source_uses_same_placement_instruction_without_changing_payer(self):
        row=fixture(True); q=row['questions']['action']
        q.update(criteria=dict(win=dict(type='hora',pai='9p'), other=dict(type='none')),
            label='win',target={'win':.5,'other':.5})
        row['state']['win_context']=dict(schema=WIN_SCHEMA,
            candidates={'win':dict(kind='ron',source='discard',source_seat=1,tile='9p')})
        got=serving_request(row,ordinary)
        self.assertEqual(got['state']['win_context'],row['state']['win_context'])
        self.assertIn(INSTRUCTION,got['questions']['action']['instructions'])
