import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.curriculum_transfer import canonical_protocol,curriculum_parent_loader,verify_curriculum_parent


class CurriculumTransferTests(unittest.TestCase):
    def fixture(self,root):
        (root/'train.jsonl').write_text('train\n');(root/'validation.jsonl').write_text('validation\n')
        (root/'code.py').write_text('public frozen source\n')
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(trainer_args=dict(data=str(root/'train.jsonl'),eval_data=str(root/'validation.jsonl')),
            input_sha256={name:sha(root/name) for name in ('train.jsonl','validation.jsonl')},
            code_sha256={'code.py':sha(root/'code.py')})

    def test_relative_bound_inputs_gain_verified_absolute_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original=self.fixture(root);snapshot=copy.deepcopy(original)
            actual=canonical_protocol(original,root)
            self.assertEqual(original,snapshot)
            self.assertIn('train.jsonl',actual['input_sha256'])
            self.assertEqual(actual['input_sha256']['train.jsonl'],actual['input_sha256'][str(root/'train.jsonl')])
            self.assertEqual(actual['code_sha256']['code.py'],actual['code_sha256'][str(root/'code.py')])

    def test_aliases_do_not_bypass_changed_or_unbound_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);protocol=self.fixture(root)
            unbound=copy.deepcopy(protocol);unbound['input_sha256'].pop('train.jsonl')
            with self.assertRaises(ValueError):canonical_protocol(unbound,root)
            conflict=copy.deepcopy(protocol);conflict['input_sha256'][str(root/'train.jsonl')]='0'*64
            with self.assertRaises(ValueError):canonical_protocol(conflict,root)
            (root/'train.jsonl').write_text('changed\n')
            with self.assertRaises(ValueError):canonical_protocol(protocol,root)

    def test_only_scoped_verifier_replaced_and_restored_on_failure(self):
        original=object();bind=object();module=SimpleNamespace(verified_parent=original,bind_parent=bind)
        with self.assertRaises(RuntimeError):
            with curriculum_parent_loader(module):
                self.assertIs(module.verified_parent,verify_curriculum_parent)
                self.assertIs(module.bind_parent,bind)
                raise RuntimeError('failed child')
        self.assertIs(module.verified_parent,original)
        self.assertIs(module.bind_parent,bind)

    def test_legacy_or_incomplete_parent_spec_is_not_a_fallback(self):
        with self.assertRaises(ValueError):
            verify_curriculum_parent(dict(initialization='continued',parent_initialization=dict(method='minimum_ce')))

    def test_incomplete_or_unqualified_seal_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'seal.json'
            path.write_text(json.dumps(dict(status='curriculum_stage1_final_parent_sealed',
                method='curriculum_registered_final_with_quality_v1',selection_from_minimum_CE=False,
                optimizer_resumed=False,stage_quality=dict(passed=False))))
            spec=dict(method='curriculum_registered_final_with_quality_v1',manifest=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity_sha256='unqualified',selected_step=6,
                source='not_loaded',source_sha256='not_loaded',run='not_loaded',child_data_ready='not_loaded')
            with self.assertRaises(ValueError):
                verify_curriculum_parent(dict(initialization='continued',parent_initialization=spec))
