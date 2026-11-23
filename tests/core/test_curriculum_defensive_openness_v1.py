import unittest
from tools.audit_curriculum_defensive_openness_v1 import structure


class DefensiveOpennessTests(unittest.TestCase):
    def test_no_meld_and_ankan_alone_stay_closed(self):
        self.assertTrue(structure([])['closed']); self.assertFalse(structure([])['any_meld'])
        for value in (3, 'ankan'):
            result = structure([{'type': value}])
            self.assertTrue(result['closed_with_ankan']); self.assertTrue(result['any_meld'])
            self.assertFalse(result['open'])

    def test_open_calls_and_mixed_ankan_remain_open(self):
        for value in (0, 1, 2, 4, 'chi', 'pon', 'daiminkan', 'kakan'):
            for ankan in ([], [{'type': 3}]):
                result = structure(ankan+[{'type': value}])
                self.assertTrue(result['open']); self.assertFalse(result['closed_with_ankan'])

    def test_unknown_or_boolean_meld_type_cannot_be_classified_as_open(self):
        for melds in (None, ['ankan'], [{'type': True}], [{'type': 9}], [{}]):
            with self.assertRaises(ValueError): structure(melds)


if __name__ == '__main__': unittest.main()
