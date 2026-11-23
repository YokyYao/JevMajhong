import copy
from fractions import Fraction
import unittest

from tools.curriculum_placement_mix_v1 import NEW_SOURCES, REPLAY_MASS, RECORDS, layout, masses


def fixture():
    # Synthetic counts for arithmetic tests, not observed production counts.
    old = {s: max(1, int(v*RECORDS)) for s, v in REPLAY_MASS.items()}
    old[max(old, key=old.get)] += RECORDS-sum(old.values())
    counts = {**old, **{s: 8+(i % 3)*8 for i, s in enumerate(NEW_SOURCES)}}
    families = {s: min(4, n) for s, n in counts.items()}
    return counts, families


class PlacementMixTests(unittest.TestCase):
    def test_exact_task_mass_and_bounded_count_only_repeat_accounting(self):
        counts, families = fixture(); before = copy.deepcopy((counts, families))
        plan = layout(counts, families)
        self.assertEqual((counts, families), before)
        self.assertEqual(len(masses()), 57)
        self.assertEqual(sum(masses().values()), 1)
        self.assertEqual(len(NEW_SOURCES), 32)
        self.assertEqual(sum(masses()[s] for s in NEW_SOURCES), Fraction(1, 5))
        self.assertGreater(plan['extra_virtual_exposures'], 0)
        self.assertEqual(plan['unique_train_records'], sum(counts.values()))
        self.assertEqual(plan['virtual_exposures_per_pass'], sum(plan['virtual_exposures_by_source'].values()))
        for source, n in counts.items():
            k = plan['virtual_repeats_per_original_by_source'][source]
            self.assertLessEqual(k, 8)
            if source in REPLAY_MASS:
                self.assertEqual(k, 1)
            weight = plan['scalar_example_weights'][source]
            self.assertLessEqual(weight, 16)
            self.assertAlmostEqual(n*k*weight/plan['virtual_exposures_per_pass'],
                                   plan['proposed_loss_mass'][source])
        self.assertFalse(plan['student_fit_admitted'])
        self.assertFalse(plan['validation_repeated'])

    def test_missing_unknown_fractional_and_changed_replay_counts_fail(self):
        counts, families = fixture()
        bad = dict(counts); bad.pop(NEW_SOURCES[0])
        with self.assertRaises(ValueError): layout(bad, families)
        bad = dict(counts); bad['unregistered_source'] = 4
        with self.assertRaises(ValueError): layout(bad, families)
        bad = dict(counts); bad[NEW_SOURCES[0]] = 8.5
        with self.assertRaises(ValueError): layout(bad, families)
        bad = dict(counts); bad[next(iter(REPLAY_MASS))] += 1
        with self.assertRaises(ValueError): layout(bad, families)

    def test_repeats_never_repair_missing_independent_family_coverage(self):
        counts, families = fixture(); families[NEW_SOURCES[0]] = 1
        with self.assertRaisesRegex(ValueError, 'independent TRAIN families'):
            layout(counts, families)

    def test_too_sparse_source_requires_data_instead_of_unbounded_repeats(self):
        counts, families = fixture(); counts[NEW_SOURCES[0]] = 2; families[NEW_SOURCES[0]] = 2
        with self.assertRaisesRegex(ValueError, 'repeat cap8'):
            layout(counts, families)

