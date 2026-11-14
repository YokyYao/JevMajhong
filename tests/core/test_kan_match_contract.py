import hashlib
from pathlib import Path
import tempfile
import unittest
from tools.audits.kan_comparison import kan_comparison_contract


class KanComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'baseline.pt'
        self.path.write_bytes(b'verified calling policy')
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        baseline = {'checkpoint': str(self.path), 'sha256': self.sha}
        self.config = {'confirmation_objective': 'hand_win', 'updated_scopes': ['ordinary'],
                       'candidate_activation': 'kan_windows', 'train_scopes': ['ordinary'],
                       'decision_filter': 'kan_windows', 'call_context_baseline': baseline}
        self.protocol = {'config': self.config, 'model_files': {'encoders': 'frozen'},
                         'call_context_baseline': baseline}
        self.fit = {'config': {'train_scopes': ['ordinary'], 'decision_filter': 'kan_windows'},
                    'model_files': self.protocol['model_files'], 'call_context_baseline': baseline}
        self.model = {'base_and_LoRA_frozen': True, 'trained_scopes': ['ordinary'],
                      'decision_filter': 'kan_windows', 'model_files': self.protocol['model_files'],
                      'call_context_baseline': baseline, 'kan_context': {'version': 'public-kan-context-v2'},
                      'context_replaces_calling_baseline': False}

    def check(self, config=None, protocol=None, fit=None, model=None, sha=None, activation='kan_windows'):
        return kan_comparison_contract(self.config if config is None else config,
            self.protocol if protocol is None else protocol, self.fit if fit is None else fit,
            self.model if model is None else model, self.sha if sha is None else sha, activation)

    def test_kan_parity_includes_unchanged_discard_and_calling_routes(self):
        result = self.check()
        self.assertEqual(result['candidate_component'], 'kan_context')
        self.assertEqual(result['expected_native_parity'],
                         {'ordinary': 2, 'calling': 2, 'ordinary_nonkan': 2})

    def test_calling_replacement_wrong_activation_or_head_only_candidate_rejected(self):
        for change in ({'context_replaces_calling_baseline': True}, {'kan_context': None},
                       {'decision_filter': 'all'}, {'trained_scopes': ['calling']},
                       {'base_and_LoRA_frozen': False}, {'model_files': {}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check(model=dict(self.model, **change))
        for key, value in [('updated_scopes', ['calling']), ('decision_filter', 'all'),
                           ('train_scopes', ['calling']), ('candidate_activation', 'all'),
                           ('confirmation_objective', 'points')]:
            cfg = dict(self.config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(config=cfg, protocol=dict(self.protocol, config=cfg))
        with self.assertRaises(ValueError):
            self.check(activation='call_windows')

    def test_original_play_missing_baseline_and_changed_weights_rejected(self):
        for change in ({'call_context_baseline': None},
                       {'call_context_baseline': dict(self.model['call_context_baseline'], sha256='0'*64)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check(model=dict(self.model, **change))
        cfg = dict(self.config);cfg.pop('call_context_baseline')
        with self.assertRaisesRegex(ValueError, 'Explicit stronger reference'):
            self.check(config=cfg, protocol=dict(self.protocol, config=cfg))
        with self.assertRaisesRegex(ValueError, 'Prescribed reference'):
            self.check(sha='0'*64)
        self.path.write_bytes(b'overwritten')
        with self.assertRaisesRegex(ValueError, 'Reference checkpoint changed'):
            self.check()


if __name__ == '__main__':
    unittest.main()
