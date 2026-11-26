import copy
import json
from pathlib import Path
import unittest
from tools.collect_curriculum_basic_teacher import sources


class BasicTeacherSourcesTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[2]/'configs/training/curriculum-basic-mortal-data-v1.json').read_text())

    def test_all_seats_and_hands_inherit_the_prospective_game_split(self):
        rows=sources(self.cfg)
        self.assertEqual(len(rows),256)
        self.assertEqual(sum(r['split']=='validation' for r in rows),51)
        self.assertEqual(len({r['group_id'] for r in rows}),256)
        self.assertEqual(rows,sources(copy.deepcopy(self.cfg)))

    def test_protected_sources_are_rejected(self):
        for protected in self.cfg['protected_source_ranges']:
            changed=copy.deepcopy(self.cfg);changed['seed_start']=protected['start']
            with self.assertRaises(ValueError):sources(changed)

    def test_no_silent_early_stop_outcome_filter_or_automatic_fit(self):
        for key,value in [('source_games',255),('no_outcome_filter',False),
                          ('all_eligible_riichi_windows',False),('student_fit_automatic',True)]:
            changed=copy.deepcopy(self.cfg);changed[key]=value
            with self.assertRaises(ValueError):sources(changed)


if __name__=='__main__':unittest.main()
