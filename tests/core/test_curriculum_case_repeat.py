import copy
import unittest
from tools.verify_curriculum_case_repeat import compatible_source_specs


class SourceRepeatContractTests(unittest.TestCase):
    def spec(self):
        return dict(config=dict(out='first/data',method='stage4_mortal_grouped_score_context_inputs_v4',
            simultaneous_seat_order='ascending_absolute_seat',source_games=64),
            sources=[dict(seed=i,split='validation' if i%5==4 else 'train') for i in range(64)],source_package_sha256='fixed')

    def test_only_output_path_changes_between_repeats(self):
        a=self.spec();b=copy.deepcopy(a);b['config']['out']='second/data'
        self.assertTrue(compatible_source_specs(a,b))

    def test_different_split_source_or_sampling_rule_rejected(self):
        for mutate in (lambda b:b['sources'][0].update(split='validation'),lambda b:b['sources'][0].update(seed=1000),
                       lambda b:b['config'].update(simultaneous_seat_order='unordered')):
            a=self.spec();b=copy.deepcopy(a);mutate(b)
            with self.assertRaises(ValueError):compatible_source_specs(a,b)

    def test_different_source_or_cohort_size_rejected(self):
        for mutate in (lambda b:b.update(source_package_sha256='changed'),lambda b:b['sources'].pop()):
            a=self.spec();b=copy.deepcopy(a);mutate(b)
            with self.assertRaises(ValueError):compatible_source_specs(a,b)


if __name__=='__main__':unittest.main()
