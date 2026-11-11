import unittest
from pathlib import Path
import tempfile
from tools.run_experiment import freeze_source
from jevmahjong.training.sampling import stratum, weights


def row(kind, delta):
    return {'category': kind, 'action': 1, 'teacher_action': 0, 'fast_advantage': delta,
        'public_request': {'questions': {'action': {'criteria': {'a0': {'type': 'none'}, 'a1': {'type': kind}}}}}}


class SamplingTests(unittest.TestCase):
    def test_editing_main_code_keeps_existing_run_snapshot_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'jevmahjong'; package.mkdir()
            file = package / '__init__.py'; file.write_text('VERSION = 1\n')
            first = freeze_source(root)
            file.write_text('VERSION = 2\n')
            second = freeze_source(root)
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'jevmahjong/__init__.py').read_text(), 'VERSION = 1\n')
            self.assertEqual((second / 'jevmahjong/__init__.py').read_text(), 'VERSION = 2\n')

    def test_known_mass_and_direction_are_balanced_without_copying_rows(self):
        rows = [row('chi', 1.) for _ in range(3)] + [row('chi', -1.)] + [row('pon', 1.), row('pon', -1.)]
        w, report = weights(rows, {'chi': .5, 'pon': .5}, 4.)
        self.assertAlmostEqual(sum(w[:4]), sum(w[4:]))
        self.assertAlmostEqual(sum(w[:3]), w[3])
        self.assertEqual(report['copies_added'], 0)
        self.assertIn('ankan/positive', report['missing_action_directions'])

    def test_skipping_beneficial_operation_reverses_label_and_rare_weight_is_capped(self):
        r = row('daiminkan', -1.); r['teacher_action'], r['action'] = 1, 0
        self.assertEqual(stratum(r), ('daiminkan', 'positive'))
        rows = [row('chi', 1.) for _ in range(30)] + [row('daiminkan', 1.)]
        w, report = weights(rows, {'chi': .5, 'daiminkan': .5}, 4.)
        self.assertEqual(max(w), 4.)
        self.assertLess(report['effective_action_mass']['daiminkan'], .5)


if __name__ == '__main__': unittest.main()
