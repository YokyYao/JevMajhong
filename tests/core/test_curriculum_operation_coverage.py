import copy
import unittest

from tools.curriculum_operation_coverage import coverage


class OperationCoverageTests(unittest.TestCase):
    def record(self, target):
        return dict(_meta=dict(id='x',group_id='g',phase='kan_window',label_type='paired_future_masked'),
            questions=dict(action=dict(criteria=dict(a=dict(type='daiminkan'),b=dict(type='pon'),c=dict(type='none')),
                target=target,label=max(target,key=target.get))))

    def test_unlabelled_operation_is_not_a_negative(self):
        result=coverage([self.record(dict(a=0.,b=.6,c=.4))])['daiminkan']
        self.assertEqual(result['operation_unlabelled_not_negative'],1)
        self.assertNotIn('operation_mass_below_half',result)

    def test_only_operation_variants_are_not_an_operation_vs_decline_contrast(self):
        row=self.record(dict(a=.6,b=.4,c=0.))
        row['questions']['action']['criteria']['b']['type']='daiminkan'
        result=coverage([row])['daiminkan']
        self.assertEqual(result['other_operations_unlabelled_not_positive_contrast'],1)
        self.assertNotIn('operation_mass_above_half',result)

    def test_labelled_contrasts_include_other_calls_and_both_directions(self):
        a=self.record(dict(a=.8,b=.2,c=0.));b=self.record(dict(a=.2,b=.8,c=0.))
        result=coverage([a,b])['daiminkan']
        self.assertEqual(result['labelled_operation_vs_other_contrast'],2)
        self.assertEqual(result['operation_mass_above_half'],1)
        self.assertEqual(result['operation_mass_below_half'],1)
        expert=copy.deepcopy(a);expert['_meta']['label_type']='expert_replay'
        expert['questions']['action'].pop('target');expert['questions']['action']['label']='b'
        self.assertEqual(coverage([expert])['daiminkan']['observed_other_action'],1)


if __name__=='__main__':unittest.main()
