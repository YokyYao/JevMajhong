import copy
import unittest

from tools.curriculum_progression_runtime_v1 import require_step0


class ProgressionWarmStartTests(unittest.TestCase):
    def proof(self):
        return dict(checked_before_first_optimizer_update=True, step0_bundle_identity='new_metadata_bundle',
            parent_bundle_identity='actual_parent',
            exact_tensors=dict(parent_adapter_exact=True, parent_head_exact=True, adapter_tensors=372, head_tensors=4),
            all_parent_prediction_parity=dict(parent_menus=12479, child_menus=23688, choices_equal=True, max_probability_error=0.))

    def test_serialization_metadata_can_differ_only_with_actual_tensor_and_full_menu_proof(self):
        reg = dict(curriculum=dict(parent_identity='actual_parent'))
        require_step0(dict(identity_sha256='new_metadata_bundle'), reg, self.proof())
        for key, value in (('parent_adapter_exact', False), ('head_tensors', 3)):
            proof = self.proof(); proof['exact_tensors'][key] = value
            with self.assertRaises(ValueError): require_step0(dict(identity_sha256='new_metadata_bundle'), reg, proof)
        proof = self.proof(); proof['all_parent_prediction_parity']['max_probability_error'] = .001
        with self.assertRaises(ValueError): require_step0(dict(identity_sha256='new_metadata_bundle'), reg, proof)

    def test_parent_hash_or_partial_validation_cannot_substitute_for_real_preupdate_equivalence(self):
        reg = dict(curriculum=dict(parent_identity='actual_parent'))
        proof = self.proof(); proof['all_parent_prediction_parity']['parent_menus'] = 10
        with self.assertRaises(ValueError): require_step0(dict(identity_sha256='new_metadata_bundle'), reg, proof)
        proof = self.proof(); proof['parent_bundle_identity'] = 'other_parent'
        with self.assertRaises(ValueError): require_step0(dict(identity_sha256='new_metadata_bundle'), reg, proof)
