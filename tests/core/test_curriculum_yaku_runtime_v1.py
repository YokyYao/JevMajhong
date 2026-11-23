"""Regression checks for complete 3B pre-update inheritance, run on GPU jobs."""
import copy
import unittest
from tools.curriculum_yaku_runtime_v1 import require_step0, prediction_parity


class YakuWarmStartTests(unittest.TestCase):
    def proof(self):
        return dict(checked_before_first_optimizer_update=True, step0_bundle_identity='new_metadata',
            parent_bundle_identity='parent',
            exact_tensors=dict(parent_adapter_exact=True, parent_head_exact=True, adapter_tensors=372, head_tensors=4),
            all_parent_prediction_parity=dict(parent_menus=23688, child_menus=24516,
                choices_equal=True, max_probability_error=0.))

    def test_only_complete_exact_preupdate_proof_allows_different_metadata(self):
        reg = dict(curriculum=dict(parent_identity='parent'))
        require_step0(dict(identity_sha256='new_metadata'), reg, self.proof())
        for path, value in [(('exact_tensors', 'parent_adapter_exact'), False),
                            (('all_parent_prediction_parity', 'parent_menus'), 12479),
                            (('all_parent_prediction_parity', 'child_menus'), 23688),
                            (('all_parent_prediction_parity', 'max_probability_error'), float('nan'))]:
            proof = self.proof(); proof[path[0]][path[1]] = value
            with self.assertRaises(ValueError): require_step0(dict(identity_sha256='new_metadata'), reg, proof)

    def test_every_parent_menu_is_covered_and_nonfinite_predictions_fail(self):
        def row(i):
            return dict(_meta=dict(id=str(i), group_id=str(i)), state='public',
                questions=dict(action=dict(type='choice', instructions='', label='a',
                    criteria=dict(a=dict(type='dahai')))))
        parent = [row(i) for i in range(23688)]
        child = parent+[row(i) for i in range(23688, 24516)]
        old = [dict(id=r['_meta']['id'], keys=['a'], p=[1.]) for r in parent]
        new = [dict(id=r['_meta']['id'], keys=['a'], p=[1.]) for r in child]
        result = prediction_parity(parent, old, child, new)
        self.assertEqual(result['max_probability_error'], 0.)
        # A defect in the final original parent menu must not escape a prefix check.
        new[23687]['p'] = [float('nan')]
        with self.assertRaises(ValueError): prediction_parity(parent, old, child, new)
        new[23687]['p'] = [1.]
        with self.assertRaises(ValueError): prediction_parity(parent, old+[old[-1]], child, new)
        wrong = copy.deepcopy(child); wrong[23687]['questions']['action']['label'] = 'other'
        with self.assertRaises(ValueError): prediction_parity(parent, old, wrong, new)
