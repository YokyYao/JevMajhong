import copy
import unittest
from unittest.mock import patch

from scripts import evaluate_feature_confirmation_parent as adapter


class IndependentParentRegistrationTests(unittest.TestCase):
    def config(self):
        return {'stage': 'independent_confirmation', 'seed_start': 10040000,
                'seeds': 128, 'out': 'confirmation', 'paired_parent_out': 'parent'}

    def original(self):
        return ({'config': {}, 'code_sha256': {'original-evaluator': 'unchanged'},
                 'candidate_files': {'head': 'same-candidate'},
                 'parent_files': {'head': 'same-parent'}},
                {'confirmation_seed_start': 10040000, 'confirmation_seeds': 128},
                {'parent_training': 'unchanged'})

    def test_only_output_is_removed_before_original_identity_verification(self):
        cfg = self.config()
        before = copy.deepcopy(cfg)
        with patch.object(adapter, '_original_registration', return_value=self.original()) as verify:
            protocol, source, parent = adapter.confirmation_registration(cfg)
        expected = {k: v for k, v in cfg.items() if k != 'paired_parent_out'}
        verify.assert_called_once_with(expected)
        self.assertEqual(cfg, before)
        self.assertEqual(protocol['source_confirmation_config'], expected)
        self.assertEqual(protocol['config'], cfg)
        self.assertEqual(protocol['candidate_files'], {'head': 'same-candidate'})
        self.assertEqual(protocol['parent_files'], {'head': 'same-parent'})
        self.assertEqual(protocol['code_sha256']['original-evaluator'], 'unchanged')
        self.assertIn('scripts/evaluate_feature_confirmation_parent.py', protocol['code_sha256'])

    def test_wrong_stage_and_same_output_fail_before_identity_hook(self):
        for update in [{'stage': 'development'}, {'paired_parent_out': 'confirmation'}]:
            with patch.object(adapter, '_original_registration') as verify:
                with self.assertRaises(ValueError):
                    adapter.confirmation_registration({**self.config(), **update})
                verify.assert_not_called()

    def test_wall_changes_fail_even_when_original_source_check_succeeds(self):
        for update in [{'seed_start': 10040001}, {'seeds': 32}]:
            with patch.object(adapter, '_original_registration', return_value=self.original()):
                with self.assertRaises(ValueError):
                    adapter.confirmation_registration({**self.config(), **update})

    def test_original_source_rejection_propagates(self):
        with patch.object(adapter, '_original_registration', side_effect=ValueError('Frozen source changed')):
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                adapter.confirmation_registration(self.config())

    def test_entrypoint_restores_shared_registration_after_failure(self):
        before = adapter.paired.registration
        def failure():
            self.assertIs(adapter.paired.registration, adapter.confirmation_registration)
            raise RuntimeError('stopped')
        with patch.object(adapter.paired, 'main', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'stopped'):
                adapter.main()
        self.assertIs(adapter.paired.registration, before)


if __name__ == '__main__':
    unittest.main()
