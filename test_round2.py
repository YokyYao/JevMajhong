import unittest

from scripts.sample_round2 import select
from scripts.train_round2 import action_summary


class RoundTwoTests(unittest.TestCase):
    def test_unique_mixture_is_deterministic_and_keeps_both_action_types(self):
        records = [{"_meta": {"id": str(i)}, "questions": {"action": {
            "label": "a0", "criteria": {"a0": {"type": "dahai" if i < 100 else "none"}}}}}
                   for i in range(140)]
        a = select(records, 100, .85, seed=42)
        b = select(list(reversed(records)), 100, .85, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len({r["_meta"]["id"] for r in a}), 100)
        self.assertEqual(sum(r["questions"]["action"]["criteria"]["a0"]["type"] == "dahai" for r in a), 85)

    def test_insufficient_unique_pool_fails_instead_of_resampling(self):
        record = {"_meta": {"id": "a"}, "questions": {"action": {
            "label": "a0", "criteria": {"a0": {"type": "dahai"}}}}}
        with self.assertRaisesRegex(ValueError, "Insufficient unique pool"):
            select([record], 100, .85)

    def test_action_diagnostics_use_target_type_and_label_index(self):
        rows = [{"id": "a", "question": "action", "p": [.1, .9], "label": 1},
                {"id": "b", "question": "action", "p": [.8, .2], "label": 1},
                {"id": "c", "question": "action", "p": [.9, .1], "label": 0}]
        result = action_summary(rows, {"a": "dahai", "b": "dahai", "c": "reach"})
        self.assertEqual(result["dahai"]["n"], 2)
        self.assertEqual(result["dahai"]["accuracy"], .5)
        self.assertEqual(result["reach"]["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
