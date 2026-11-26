import copy
import json
from pathlib import Path
import unittest
from tools.collect_curriculum_advanced_teacher import category,families
from tools.collect_curriculum_basic_teacher import sources


class AdvancedTeacherTests(unittest.TestCase):
    def test_same_family_split_is_reused_and_new_independence_is_not_invented(self):
        root=Path(__file__).resolve().parents[2]
        basic=json.loads((root/'configs/training/curriculum-basic-mortal-data-v1.json').read_text())
        advanced=json.loads((root/'configs/training/curriculum-advanced-mortal-data-v1.json').read_text())
        parent={'config':basic,'sources':sources(basic)}
        self.assertEqual(families(advanced,parent),parent['sources'])
        changed=copy.deepcopy(parent);changed['sources'][0]['split']='validation'
        with self.assertRaises(ValueError):families(advanced,changed)
        changed=copy.deepcopy(advanced);changed['filter_by_success_or_selected_action']=True
        with self.assertRaises(ValueError):families(changed,parent)

    def test_win_guard_and_overlapping_kan_menus_have_explicit_precedence(self):
        self.assertIsNone(category({'hora','daiminkan','pon','none'},False))
        self.assertEqual(category({'daiminkan','pon','none'},False),'kan')
        self.assertEqual(category({'chi','pon','none'},False),'call')

    def test_post_ankan_discards_are_included_without_calling_them_open(self):
        self.assertEqual(category({'dahai'},True),'post_meld')
        self.assertIsNone(category({'dahai','reach'},False))
        self.assertIsNone(category({'none'},False))


if __name__=='__main__':unittest.main()
