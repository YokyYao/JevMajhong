import copy
import json
from pathlib import Path
import unittest

from tools.collect_curriculum_endgame_cases import cohort,category


class EndgameCaseCollectionTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[2]/'configs/training/curriculum-stage4-cases-v1.json').read_text())

    def test_declared_split_is_by_complete_source_family_before_generation(self):
        sources=cohort(self.config)
        self.assertEqual(len(sources),64)
        self.assertEqual(sum(row['split']=='train' for row in sources),52)
        self.assertEqual(sum(row['split']=='validation' for row in sources),12)
        self.assertEqual(len({row['group_id'] for row in sources}),64)
        self.assertEqual(sources,cohort(copy.deepcopy(self.config)))
        for offset,row in enumerate(sources):
            self.assertEqual(row['hero_seat'],offset%4)
            self.assertEqual(row['split'],'validation' if offset%5==4 else 'train')

    def test_reserved_final_and_consumed_diagnostics_cannot_enter_source_pool(self):
        for protected in self.config['protected_source_ranges']:
            changed=copy.deepcopy(self.config)
            changed['seed_start']=protected['start']+protected['count']-1
            with self.assertRaises(ValueError):cohort(changed)

    def test_bounded_cohort_and_both_splits_required(self):
        for field,value in [('source_games',True),('source_games',513),('validation_period',1),
                            ('validation_residue',5),('seed_start',True)]:
            changed=copy.deepcopy(self.config);changed[field]=value
            with self.assertRaises(ValueError):cohort(changed)
        changed=copy.deepcopy(self.config);changed.update(source_games=5,validation_period=10,validation_residue=9)
        with self.assertRaises(ValueError):cohort(changed)

    def test_wins_take_priority_and_can_be_observed_for_other_seats(self):
        menu={'A':{'type':'hora'},'B':{'type':'dahai'},'C':{'type':'none'}}
        self.assertEqual(category(menu,'draw',True),'win')
        self.assertEqual(category(menu,'response',False),'win')

    def test_progression_and_calls_require_hero_choice_and_decline_alternative(self):
        draw={'A':{'type':'dahai'},'B':{'type':'reach'}}
        self.assertEqual(category(draw,'draw',True),'progression_draw')
        self.assertIsNone(category(draw,'draw',False))
        self.assertIsNone(category({'A':{'type':'dahai'}},'draw',True))
        for action in ('chi','pon','daiminkan'):
            menu={'A':{'type':action},'B':{'type':'none'}}
            self.assertEqual(category(menu,'response',True),'call_response')
            self.assertIsNone(category(menu,'response',False))
            self.assertIsNone(category({'A':{'type':action},'B':{'type':'hora_not_supported'}},'response',True))


if __name__=='__main__':unittest.main()
