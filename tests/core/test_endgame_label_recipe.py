from pathlib import Path
import tempfile
import unittest

from jevmahjong.endgame.prepare import label_recipe


class MixedLabelRecipeTests(unittest.TestCase):
    def test_original_counts_and_temperatures_are_not_silently_coerced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = {}
            for tag, worlds, temperature in [('legacy4', 4, .25), ('target16', 16, .1), ('fourth32', 32, .1)]:
                original = root / tag / 'labels' / (tag + '.json')
                original.parent.mkdir(parents=True)
                original.write_text('{}')
                alias = root / 'mixed' / 'labels' / original.name
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.symlink_to(original)
                specs[tag] = {'root': str(root / tag), 'future_worlds': worlds,
                              'source_target_temperature': temperature}
                self.assertEqual(label_recipe({'label_sets': specs}, {'_label_set': tag}, alias),
                                 (worlds, temperature, tag))
            with self.assertRaises(ValueError):
                label_recipe({'label_sets': specs}, {'_label_set': 'legacy4'}, alias)
            with self.assertRaises(ValueError):
                label_recipe({'label_sets': specs}, {}, alias)

    def test_legacy_single_set_recipe_keeps_original_defaults(self):
        self.assertEqual(label_recipe({'future_worlds': 4, 'target_temperature': .25}, {}, 'unused'),
                         (4, .25, 'legacy_single_set'))


if __name__ == '__main__':
    unittest.main()
