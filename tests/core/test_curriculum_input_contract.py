import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.curriculum_input_contract import ROOT, load_context_client, seal_checkpoint, verify_checkpoint
from tools.curriculum_riichi_context import FIELD, VERSION
from test_curriculum_riichi_context import request


class Spy:
    def __init__(self):
        self.calls = []

    def system_one(self, **value):
        self.calls.append(value)
        return dict(result='unchanged-model-output')


class InputContractTests(unittest.TestCase):
    def setUp(self):
        feature = ROOT/'tools/curriculum_riichi_context.py'
        self.manifest = dict(curriculum=dict(riichi_input_schema=VERSION),
            inputs={str(feature.resolve()): hashlib.sha256(feature.read_bytes()).hexdigest()})
        self.model = dict(identity_sha256='example-model', files={'head.pt': 'example-file-hash'})

    def test_missing_or_wrong_model_contract_cannot_load(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                verify_checkpoint(temp, self.manifest, lambda _: self.model)
            seal_checkpoint(temp, self.manifest, lambda _: self.model)
            altered = dict(self.model, identity_sha256='another-model')
            with self.assertRaises(ValueError):
                verify_checkpoint(temp, self.manifest, lambda _: altered)

    def test_registered_transform_hash_and_version_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = copy.deepcopy(self.manifest)
            bad['inputs'] = {k: 'wrong' for k in bad['inputs']}
            with self.assertRaises(ValueError):
                seal_checkpoint(temp, bad, lambda _: self.model)
            bad = dict(self.manifest, curriculum=dict(riichi_input_schema='unknown-version'))
            with self.assertRaises(ValueError):
                seal_checkpoint(temp, bad, lambda _: self.model)

    def test_exactly_one_model_call_and_no_probability_override(self):
        with tempfile.TemporaryDirectory() as temp:
            seal_checkpoint(temp, self.manifest, lambda _: self.model)
            spy = Spy()
            client = load_context_client(spy, temp, self.manifest, lambda _: self.model)
            original = request()
            q = original['questions']['action']
            public = dict(state=original['state'], questions=dict(action={k: v for k, v in q.items()
                if k in ('type', 'instructions', 'criteria')}))
            self.assertEqual(client.system_one(**public), dict(result='unchanged-model-output'))
            self.assertEqual(len(spy.calls), 1)
            self.assertIn(FIELD, spy.calls[0]['questions']['action']['criteria']['a0'])
            self.assertNotIn(FIELD, original['questions']['action']['criteria']['a0'])
