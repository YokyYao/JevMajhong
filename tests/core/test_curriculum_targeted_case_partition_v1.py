import ast
import copy
from pathlib import Path
import unittest

from tools import curriculum_targeted_case_partition_v1 as subject
from tools import curriculum_targeted_win_resume_v2 as original
from tools import curriculum_targeted_win_resume_v3 as cached


class FixedCasePartitionTests(unittest.TestCase):
    def setUp(self):
        self.specs = [dict(id='case-'+str(index),group='family-'+str(index//3),
            split='validation',context=index%3) for index in range(27)]

    def test_disjoint_exact_coverage_of_missing_population(self):
        done = [row['id'] for row in self.specs[:9]]
        lanes = subject.partition_missing(self.specs,done,3)
        self.assertEqual([len(ids) for ids in lanes],[6,6,6])
        flattened = [rid for ids in lanes for rid in ids]
        self.assertEqual(len(flattened),len(set(flattened)))
        self.assertEqual(set(flattened),{row['id'] for row in self.specs[9:]})
        self.assertFalse(set(flattened)&set(done))

    def test_less_than_three_missing_and_all_completed(self):
        ids = [row['id'] for row in self.specs]
        self.assertEqual(subject.partition_missing(self.specs,ids[:-2],3),[[ids[-2]],[ids[-1]]])
        self.assertEqual(subject.partition_missing(self.specs,ids,3),[])

    def test_unknown_completed_ids_are_rejected(self):
        with self.assertRaises(ValueError): subject.partition_missing(self.specs,['unknown'],3)

    def test_selection_preserves_objects_original_order_and_metadata(self):
        before = copy.deepcopy(self.specs)
        selected = subject.select_specs(self.specs,['case-15','case-2','case-9'])
        self.assertEqual(selected,[self.specs[2],self.specs[9],self.specs[15]])
        self.assertTrue(all(row is self.specs[index] for row,index in zip(selected,(2,9,15))))
        self.assertEqual(self.specs,before)

    def test_duplicate_or_unknown_assignment_is_rejected(self):
        for ids in ([],['case-1','case-1'],['unknown']):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError): subject.select_specs(self.specs,ids)

    def test_every_original_case_calculation_is_preserved(self):
        path = Path(original.COLLECTOR)
        source = path.read_text()
        baseline = cached.collector_tree(source)
        adapted = subject.collector_tree(source)
        _, first = original.original_loop(baseline)
        _, second = original.original_loop(adapted)
        self.assertEqual(ast.dump(first),ast.dump(second))
        compile(adapted,str(path),'exec')
        # Reversing only this one selector must recover the entire cached
        # collector AST, including before/after-loop integrity checks.
        functions = [node for node in adapted.body if isinstance(node,ast.FunctionDef)]
        selectors = [node for node in functions[0].body if isinstance(node,ast.Assign)
            and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
            and node.targets[0].id=='specs']
        self.assertEqual(len(selectors),1)
        self.assertEqual(selectors[0].value.func.id,'select_specs')
        selectors[0].value = selectors[0].value.args[0]
        self.assertEqual(ast.dump(adapted),ast.dump(baseline))


if __name__ == '__main__': unittest.main()
